using Statistics
using LinearAlgebra
using Zygote

function trace(x, y)
    println(x)
    y
end

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

abstract type AbstractMCMC_Model <: Model end

mutable struct MCMC_Model <: AbstractMCMC_Model
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
        randn(2 * n_params) .|> abs, 
        θ -> w -> logpdf(MvNormal(θ[1:n_params], Diagonal(θ[n_params + 1:end])), w)
    )
end

# h is smooth non-linearity
function PlanarFlowLayer(D::Int, h::Function)
    ψ = (w,b) -> (z -> gradient(h, w'z + b)[1] * w)
    return NormalisingFlowLayer(
        D,
        2D + 1,
        θ -> (z -> z + (θ[1:D] .* h(θ[D+1:2D]' * z + θ[2D+1]))),
        θ -> (z -> abs(1 + θ[1:D]'ψ(θ[D+1:2D], θ[2D+1])(z)))
    )
end

function RadialFlowLayer(D::Int)
    h = (a, r) -> 1/(a + r)
    return NormalisingFlowLayer(
        D,
        D + 2,
        θ -> (z -> z + θ[1] * h(θ[2], norm(z - θ[1:D])) .* (z - θ[1:D])),
        θ -> (z -> ((1 + θ[1] * h(θ[2], (r = norm(z - θ[1:D])))) ^ (D - 1)) * 
            (1 + θ[1] * h(θ[2], r) + 
                θ[1] * gradient(z -> h(θ[2], r), z) * r
            ) 
        )
    )
end

function softmax(z)
    z′ = exp.(z)
    return z′./ sum(z′)
end

# log likelihood for binary classification
function binary_log_likelihood(m::Model, w::AbstractArray{Float64}, X::Matrix{Float64}, 
        y::BitVector)
    ŷ = pred(m, w, X)' # vector of Float64
    return sum(y .* log.(ŷ .+ eps()) + (1 .- y) .* log.(1 .- ŷ .+ eps()))
end

# log likelihood for multi-class classification
function multi_class_log_likelihood(m::Model, w::AbstractArray{Float64}, X::Matrix{Float64}, 
        y::Vector{Int})
    ŷ = pred(m, w, X) # matrix of Float64, column samples
    normalised_ŷ = mapslices(softmax, ŷ, dims=1)
    return sum(log.(normalised_ŷ[y] .+ eps()))
end

# log likelihood for generalised multi-target regression
function regression_log_likelihood(m::Model, w::AbstractArray{Float64}, X::Matrix{Float64},
        y::Matrix{Float64})
    col_outer_product = x -> x * x'
    col_gaussian_exponent = Σ -> (x -> x' * Σ * x)

    ŷ = pred(m, w, X) # matrix of Float64, column samples
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
function pred(m::Model, weights::AbstractArray{Float64}, X::Matrix{Float64})
    # initialise forward pass state
    output = X                      # previous layer output
    next_i_w = 1                    # next index to start from for weights
    next_i_b = 0                    # next index to start from back for biases
    w_offset = m.structure.n_inputs # last layer dimension

    # iterate over the layers of the network
    for i in 1:length(m.structure.layers)
        # get weights and biases of current layer
        layer_weights = weights[next_i_w:next_i_w + w_offset * m.structure.layers[i].n - 1]
        layer_weights = reshape(layer_weights, (m.structure.layers[i].n, w_offset))
        
        biases = weights[end - next_i_b - m.structure.layers[i].n + 1: end - next_i_b]
        next_i_b = next_i_b + m.structure.layers[i].n

        # update forward pass state
        output = m.structure.layers[i].activation.(layer_weights * output .+ biases)
        next_i_w = next_i_w + w_offset * m.structure.layers[i].n
        w_offset = m.structure.layers[i].n
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