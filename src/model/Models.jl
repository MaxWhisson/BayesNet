module Models

# type exports
export  Model, 
        Layer,
        DenseLayer,
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
        produce_degenerate,
        simple_apply_grad,
        produceMultiModel

using Statistics
using LinearAlgebra
using Zygote
using Distributions
using LogExpFunctions
using ..HelperFunctions
using Optimisers

abstract type Layer end

abstract type Model end

# struct for specifying a dense neural network layer.
struct DenseLayer <: Layer
    n::Int                # number of neurons in layer
    activation::Function    # activation function
end

# struct for specifying a residual neural network layer.
struct ResidualLayer <: Layer
    n1::Int
    n2::Int
    activation::Function
end

# struct for creating model architectures.
struct ModelStructure
    layers::Vector{Layer}
    n_inputs::Int
    n_total_params::Int         # total number of weights in the model
end

# struct for parameterised priors
# * function is θ -> x -> Type parameterised over θ
mutable struct ParameterisedFunction
    optimiser_state
    optimiser_rule
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
    apply_grad::Function
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
    apply_grad::Function
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
    apply_grad::Function
    structure::ModelStructure
    θ::Vector{Float64}
end

##########################################################################
####                            Functions                             ####
##########################################################################

function simple_apply_grad(m::Models.Model, g)
    m.θ = m.θ .- g
end

function produce_degenerate(layers::Vector, n_inputs::Int)
    n_weights = n_inputs * layers[1].n +
        sum([layers[i].n * layers[i + 1].n for i in 1:length(layers) - 1])
    n_params = n_weights + sum(layers .|> (x -> x.n))
    DegenerateModel(
        simple_apply_grad,
        ModelStructure(layers, n_inputs, n_params),
        init_means(layers, n_inputs)
    )
end

# function for creating diagonal gaussian parametrised priors
function diagonal_gaussian_prior_creator(n_params::Int; weight = log(0.1)) 
    θ = [zeros(n_params); weight]
    rule = Optimisers.Adam(0.1)
    state = Optimisers.init(rule, θ)
    return ParameterisedFunction(
        rule,
        state,
        θ,
        θ -> w -> -exp.(weight) * w'w
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
        X::AbstractMatrix{Float64}, y::AbstractArray{Float64})
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
    N = length(y)
    error_diff = ŷ - y

    # TODO
    # σ2 = (1/N) * (error_diff' * error_diff)
    σ2 = 0.01

    return -(1/(2 *σ2)) * (error_diff' * error_diff)
end

# log P(D|w)P(w)
function log_density(m::Model, w::AbstractArray, 
        X::AbstractMatrix{Float64}, y::AbstractArray; coef = 1)
    mapped_log_likelihood = (m, X, y) -> (w -> m.log_likelihood(m, w, X, y))
    log_prior = m.log_prior.func(m.log_prior.θ)

    # println("$(mean(log_prior.(eachcol(w)))), $(mean(mapped_log_likelihood(m, X, y).(eachcol(w))))\n")

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
        if typeof(s.layers[i]) == DenseLayer
            layer_weights = weights[next_i_w:next_i_w + w_offset * s.layers[i].n - 1]
            layer_weights = reshape(layer_weights, (s.layers[i].n, w_offset))
            biases = weights[end - next_i_b - s.layers[i].n + 1: end - next_i_b]
            next_i_w = next_i_w + w_offset * s.layers[i].n
            next_i_b = next_i_b + s.layers[i].n

            # update forward pass state
            output = s.layers[i].activation.(layer_weights * output .+ biases)
            w_offset = s.layers[i].n
            
        elseif typeof(s.layers[i]) == ResidualLayer
            layer_weights_1 = weights[next_i_w:next_i_w + w_offset * s.layers[i].n1 - 1]
            layer_weights_1 = reshape(layer_weights_1, (s.layers[i].n1, w_offset))
            biases_1 = weights[end - next_i_b - s.layers[i].n1 + 1: end - next_i_b]
            next_i_w = next_i_w + w_offset * s.layers[i].n1
            next_i_b = next_i_b + s.layers[i].n1

            layer_weights_2 = weights[next_i_w:next_i_w + s.layers[i].n1 * s.layers[i].n2 - 1]
            layer_weights_2 = reshape(layer_weights_2, (s.layers[i].n2, w_offset))
            biases_2 = weights[end - next_i_b - s.layers[i].n2 + 1: end - next_i_b]
            next_i_w = next_i_w + s.layers[i].n1 * s.layers[i].n2
            next_i_b = next_i_b + s.layers[i].n2

            output_temp = s.layers[i].activation.(layer_weights_1 * output .+ biases_1)
            output = output + (layer_weights_2 * output_temp .+ biases_2)
            # don't need to change w_offset
        end
    end
    return output
end

end