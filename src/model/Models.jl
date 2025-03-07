module Models

# type exports
export  Model, 
        Layer, 
        ModelStructure, 
        ParameterisedFunction,
        NormalisingFlowLayer,
        VariationalModel,
        LaplaceModel,
        MCMC_Model,
        DegenerateModel

# function exports
export  diagonal_gaussian_prior_creator,
        PlanarFlowLayer,
        RadialFlowLayer,
        inverse_flow,
        softmax,
        binary_log_likelihood,
        multi_class_log_likelihood,
        regression_log_likelihood,
        log_density,
        pred,
        prune_diagonal_gaussian_proportion!,
        prune_diagonal_gaussian_CI!,
        produce_degenerate

using Statistics
using LinearAlgebra
using Zygote
using Distributions
using LogExpFunctions
using ..HelperFunctions

# struct for specifying a dense neural network layer.
struct Layer
    n::Int                # number of neurons in layer
    activation::Function    # activation function
end

abstract type Model end

# struct for creating model architectures.
struct ModelStructure
    layers::Vector{Layer}
    n_inputs::Int
    n_total_params::Int         # total number of weights in the model
end

# struct for parameterised priors
# * function is θ -> x -> Type parameterised over θ
mutable struct ParameterisedFunction
    θ::Vector{Float64}
    func::Function
end

# normalising flow is vector of layers
mutable struct NormalisingFlowLayer
    D::Int
    n_params::Int
    func::Function                      # θ -> x -> Type
    jacobian_determinant::Function      # θ -> x -> Type
    inverse_func::Function              # θ -> x -> Type
end

# struct for creating variational models.
mutable struct VariationalModel <: Model
    structure::ModelStructure
    θ::Vector{Float64}
    weight_sampler::Function
    log_prior::ParameterisedFunction
    log_likelihood::Function
    log_posterior::Function

    normalising_flow::Vector{NormalisingFlowLayer}
    n_variational_params::Int
    n_flow_params::Int
end

# struct for creating Laplace models.
mutable struct LaplaceModel <: Model
    structure::ModelStructure
    θ::Vector{Float64}                  # w_MAP then log L
    log_prior::ParameterisedFunction    # log prior on weights
    log_likelihood::Function            # likelihood of data for weights
end

mutable struct MCMC_Model <: Model
    structure::ModelStructure
    log_prior::ParameterisedFunction    # log prior on weights
    log_likelihood::Function            # likelihood of data for weights
end

mutable struct DegenerateModel <: Model
    structure::ModelStructure
    θ::Vector{Float64}
end

##########################################################################
####                            Functions                             ####
##########################################################################

function produce_degenerate(layers::Vector{Layer}, n_inputs::Int)
    n_weights = n_inputs * layers[1].n +
        sum([layers[i].n * layers[i + 1].n for i in 1:length(layers) - 1])
    n_params = n_weights + sum(layers .|> (x -> x.n))
    DegenerateModel(
        ModelStructure(layers, n_inputs, n_params),
        randn(n_params)
    )
end

# function for creating diagonal gaussian parametrised priors
function diagonal_gaussian_prior_creator(n_params::Int; weight = 10) 
    return ParameterisedFunction(
        [zeros(n_params); weight * ones(n_params)], 
        θ -> w -> logpdf(MvNormal(
            θ[1:n_params], 
            Diagonal(log.(1 .+ exp.(θ[n_params + 1:end])) .^ 2)
        ), w)
    )
end

# h is smooth non-linearity
function PlanarFlowLayer(D::Int)
    h = tanh
    function ψ(w::AbstractArray{Float64}, b::Float64)
        function f(z)
            x = w'z + b
            gradient(y -> h(y), x)[1] * w
        end
    end

    return NormalisingFlowLayer(
        D,
        2D + 1,
        θ -> (z -> z + (θ[1:D] .* h(θ[D+1:2D]' * z + θ[2D+1]))),
        θ -> (z -> abs(1 + θ[1:D]'ψ(θ[D+1:2D], θ[2D+1])(z))),
        x -> throw("No analytical inverse for Planar Transform")
    )
end

function RadialFlowLayer(D::Int)
    h = (α, r) -> 1/(α + r)
    h′ = (a, r) -> -1/((a + r) ^ 2)

    function abs_det(θ::AbstractArray{Float64})
        α = abs(θ[1])
        β = -α + log(1 + exp(θ[2]))
        z₀ = θ[3:D+2]
        function f(z::AbstractArray{Float64})
            r = norm(z - z₀)
            t1 = ((1 + β * h(α, r)) ^ (D - 1))
            t2 = 1 + β * h(α, r) + β * h′(α, r) * r
            return t1 * t2
        end
    end

    function func(θ::AbstractArray{Float64})
        α = abs(θ[1])
        β = -α + log(1 + exp(θ[2]))
        z₀ = θ[3:D+2]
        function f(z::AbstractArray{Float64})
            r = norm(z - z₀)
            return z + β * h(α, r) * (z - z₀)
        end
    end

    function inverse_func(θ::AbstractArray{Float64})
        α = abs(θ[1])
        β = -α + log(1 + exp(θ[2]))
        z₀ = θ[3:D+2]
        function f(z′::AbstractArray{Float64})
            r′ = norm(z′ - z₀)
            r = ((r′ - α - β) + sqrt(((α + β - r′) ^ 2) + 4r′ * α)) / 2
            return z₀ + (z′ - z₀) / (1 + β * h(α, r))
        end
    end

    return NormalisingFlowLayer(
        D,
        D + 2,
        func,
        abs_det,
        inverse_func
    )
end

function inverse_flow(zₙ::AbstractArray{Float64}, 
        flow_params::AbstractArray{Float64}, 
        flow::AbstractArray{NormalisingFlowLayer})
    foldl(
        ((m, param_index, i), l) -> (
            set_col_matrix_expr(m, i + 1, l.inverse_func(
                flow_params[param_index:param_index+l.n_params - 1]
            )(m[:,i])),
            param_index + l.n_params,
            i + 1
        ),
        reverse(flow),
        init=([zₙ zeros(length(zₙ), length(flow))], 1, 1)
    )[1]
end

function softmax(z::AbstractArray{Float64})
    z .-= maximum(z)
    z′ = exp.(z)
    return z′./ sum(z′)
end

# log likelihood for binary classification
function binary_log_likelihood(m::Model, w::AbstractArray, 
        X::AbstractMatrix{Float64}, y::BitVector)
    ŷ = pred(m.structure, w, X)' # vector of Float64
    return sum(y .* HelperFunctions.s_log.(ŷ) + (1 .- y) .* 
        HelperFunctions.s_log.(1 .- ŷ))
end

# log likelihood for multi-class classification
function multi_class_log_likelihood(m::Model, w::AbstractArray{Float64}, 
        X::AbstractMatrix{Float64}, y::AbstractArray{Int})
    ŷ = pred(m.structure, w, X) # matrix of Float64, column samples
    normalised_ŷ = mapslices(softmax, ŷ, dims=1)
    return sum(HelperFunctions.s_log.(normalised_ŷ[y]))
end

# log likelihood for mono-target regression
function regression_log_likelihood(m::Model, w::AbstractArray, 
        X::AbstractMatrix{Float64}, y::AbstractArray{Float64})
    ŷ = pred(m.structure, w, X)'[:,1]
    error_diff = ŷ - y
    return -(error_diff' * error_diff)
end

# log P(D|w)P(w)
function log_density(m::Model, w::AbstractArray, 
        X::AbstractMatrix{Float64}, y::AbstractArray; coef = 1)
    mapped_log_likelihood = (m, X, y) -> (w -> m.log_likelihood(m, w, X, y))
    log_prior = m.log_prior.func(m.log_prior.θ)

    # println("$(mean(log_prior.(eachcol(w)))), $(mean(mapped_log_likelihood(m, X, y).(eachcol(w))))\n")
    # return mean(mapped_log_likelihood(m, X, y).(eachcol(w)))

    return coef * mean(log_prior.(eachcol(w))) + 
        mean(mapped_log_likelihood(m, X, y).(eachcol(w)))
end

# forward pass of model architecture for data X and weights.
function pred(s::ModelStructure, weights::AbstractArray, 
        X::AbstractArray)

    # initialise forward pass state
    output = X                      # previous layer output
    next_i_w = 1                    # next index to start from for weights
    next_i_b = 0                    # next index to start from back for biases
    w_offset = s.n_inputs # last layer dimension

    # iterate over the layers of the network
    for i in 1:length(s.layers)
        # get weights and biases of current layer
        layer_weights = weights[
            next_i_w:next_i_w + w_offset * s.layers[i].n - 1
        ]
        layer_weights = reshape(layer_weights, (s.layers[i].n, w_offset))
        
        biases = weights[end - next_i_b - s.layers[i].n + 1: end - next_i_b]
        next_i_b = next_i_b + s.layers[i].n

        # update forward pass state
        output = s.layers[i].activation.(layer_weights * output .+ biases)
        next_i_w = next_i_w + w_offset * s.layers[i].n
        w_offset = s.layers[i].n
    end
    return output
end

# remove proportion of highest variance weights' variational parameters 
function prune_diagonal_gaussian_proportion!(m; proportion = 0.9)
    θ = m.θ[1:m.n_variational_params]
    log_σ = θ[m.structure.n_total_params + 1:end]
    sorted = zip(log_σ, 1:m.structure.n_total_params) |> collect |> sort
    # zero everything below this
    cutoff_i = ceil(proportion * m.structure.n_total_params)

    # indexes of params to be zeroed
    to_zero = (x -> x[2]).(sorted)[1:Int(cutoff_i)]
    m.θ[to_zero] .= 0 
    m.θ[to_zero .+ m.structure.n_total_params] .= -1000
end

# remove parameters for weights that are not significantly different
# from 0
function prune_diagonal_gaussian_CI!(m; CI_probability = 0.9)
    function f(μ, log_σ)
        cdf(Normal(μ, exp(log_σ)), 0) > (1 - CI_probability) / 2
    end

    function g(offset, m)
        function h(i)
            if f(m.θ[i], m.θ[1 + offset])
                m.θ[i], m.θ[1 + offset] = 0, -1000
            end
        end
    end
        
    g(m.structure.n_total_params, m).(1:m.structure.n_total_params)
end

end