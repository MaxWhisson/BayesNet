module Models

# type exports
export  Model, 
        Layer, 
        ModelStructure, 
        ParameterisedFunction,
        NormalisingFlowLayer,
        VariationalModel,
        LaplaceModel,
        MCMC_Model

# function exports
export  diagonal_gaussian_prior_creator,
        PlanarFlowLayerCreator,
        RadialFlowLayer,
        softmax,
        binary_log_likelihood,
        multi_class_log_likelihood,
        regression_log_likelihood,
        log_density,
        pred

using Statistics
using LinearAlgebra
using Zygote
using Distributions
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
end

# struct for creating variational models.
mutable struct VariationalModel <: Model
    structure::ModelStructure
    θ::Vector{Float64}
    weight_sampler::Function
    log_prior::ParameterisedFunction
    log_likelihood::Function
    log_posterior::Function

    variational_param_generator::Function

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

##########################################################################
####                            Functions                             ####
##########################################################################

# function for creating diagonal gaussian parametrised priors
function diagonal_gaussian_prior_creator(n_params) 
    return ParameterisedFunction(
        [zeros(n_params);ones(n_params)], 
        θ -> w -> logpdf(MvNormal(θ[1:n_params], Diagonal(log.(1 .+ exp.(θ[n_params + 1:end])))), w)
    )
end

# h is smooth non-linearity
function PlanarFlowLayerCreator(h::Function)
    function PlanarFlowLayer(D::Int)
        function ψ(w,b)
            function f(z)
                x = w'z + b
                gradient(y -> h(y), x)[1] * w
            end
        end

        return NormalisingFlowLayer(
            D,
            2D + 1,
            θ -> (z -> z + (θ[1:D] .* h(θ[D+1:2D]' * z + θ[2D+1]))),
            θ -> (z -> abs(1 + θ[1:D]'ψ(θ[D+1:2D], θ[2D+1])(z)))
        )
    end
end

function RadialFlowLayer(D::Int)
    h = (a, r) -> 1/(a + r)

    function abs_det(θ)
        function f(z)
            r = norm(z - θ[3:D + 2])
            t1 = ((1 + θ[1] * h(θ[2], r)) ^ (D - 1)) 
            t2 = (1 + θ[1] * h(θ[2], r) + 
                θ[1] * gradient(z -> h(θ[2], z), r)[1] * r
            ) 
            return t1 * t2
        end
    end

    return NormalisingFlowLayer(
        D,
        D + 2,
        θ -> (z -> z + θ[1] * h(θ[2], norm(z - θ[3:D + 2])) .* (z - θ[3:D + 2])),
        abs_det
    )
end

function softmax(z)
    z′ = exp.(z)
    return z′./ sum(z′)
end

# log likelihood for binary classification
function binary_log_likelihood(m::Model, w::AbstractArray{Float64}, X::Matrix{Float64}, 
        y::BitVector)
    ŷ = pred(m.structure, w, X)' # vector of Float64
    return sum(y .* HelperFunctions.s_log.(ŷ) + (1 .- y) .* HelperFunctions.s_log.(1 .- ŷ))
end

# log likelihood for multi-class classification
function multi_class_log_likelihood(m::Model, w::AbstractArray{Float64}, X::Matrix{Float64}, 
        y::Vector{Int})
    ŷ = pred(m.structure, w, X) # matrix of Float64, column samples
    normalised_ŷ = mapslices(softmax, ŷ, dims=1)
    return sum(HelperFunctions.s_log.(normalised_ŷ[y]))
end

# log likelihood for generalised multi-target regression
function regression_log_likelihood(m::Model, w::AbstractArray{Float64}, X::Matrix{Float64},
        y::Matrix{Float64})
    col_outer_product = x -> x * x'
    col_gaussian_exponent = Σ -> (x -> x' * Σ * x)

    ŷ = pred(m.structure, w, X) # matrix of Float64, column samples
    mean_diff = ŷ .- mean(ŷ, dims=2)
    sample_Σ = sum(col_outer_product.(eachcol(mean_diff))) / size(ŷ)[2]
    error_diff = ŷ .- y

    return sum(-col_gaussian_exponent(sample_Σ).(eachcol(error_diff)))
end

# log P(D|w)P(w)
function log_density(m, w, X, y; coef = 1)
    mapped_log_likelihood = (m, X, y) -> (w -> m.log_likelihood(m, w, X, y))
    log_prior = m.log_prior.func(m.log_prior.θ)
    return coef * log_prior.(eachcol(w)) + mapped_log_likelihood(m, X, y).(eachcol(w))
end

# forward pass of model architecture for data X and weights.
function pred(s::ModelStructure, weights::AbstractArray{Float64}, X::Matrix{Float64})
    # initialise forward pass state
    output = X                      # previous layer output
    next_i_w = 1                    # next index to start from for weights
    next_i_b = 0                    # next index to start from back for biases
    w_offset = s.n_inputs # last layer dimension

    # iterate over the layers of the network
    for i in 1:length(s.layers)
        # get weights and biases of current layer
        layer_weights = weights[next_i_w:next_i_w + w_offset * s.layers[i].n - 1]
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

# produce new model with (0, 0) Gaussians for diagonal model
# TODO consider full Gaussians

# remove proportion of highest variance weights' variational parameters 
function prune_weights_proportion(m; proportion = 0.9)
    θ = m.θ[1:m.n_variational_params]
    log_σ = θ[m.structure.n_total_params + 1:end]
    # TODO zip with indexes and sort
end

# remove parameters for weights that are not significantly different
# from 0
function prune_weights_CI(m; CI_probability = 0.9)
    # TODO
end

end