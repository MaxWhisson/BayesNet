module Models

# type exports
export  Model,
        ModelStructure, 
        ParameterisedFunction,
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
        produceMultiModel,
        init_means,
        count_params

using Statistics
using LinearAlgebra
using Zygote
using Distributions
using LogExpFunctions
using Optimisers

using ..HelperFunctions
using ..Layer:NNLayer
using ..Normalising

using ..Planar
using ..Radial

using ..DenseLayer
using ..ResidualLayer

abstract type Model end

# struct for creating model architectures.
struct ModelStructure
    layers::Vector{NNLayer}
    n_inputs::Int
    n_total_params::Int         # total number of weights in the model
end

# struct for parameterised priors
# * function is θ -> x -> Type parameterised over θ
mutable struct ParameterisedFunction
    optimiser_rule
    optimiser_state
    θ::Vector{Float64}
    func::Function
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
    n_params = count_params(layers, n_inputs)
    DegenerateModel(
        simple_apply_grad,
        ModelStructure(layers, n_inputs, n_params),
        init_means(layers, n_inputs)
    )
end

# function for creating diagonal gaussian parametrised priors
function diagonal_gaussian_prior_creator(n_params::Int; weight = log(0.1),
        hyper_weight = 0.1, rule = Optimisers.Adam(0.01)) 
    θ = [zeros(n_params); weight * ones(n_params)]
    state = Optimisers.init(rule, θ)
    return ParameterisedFunction(
        rule,
        state,
        θ,
        θ -> w -> -((w - θ[1:n_params])' * 
            (exp.(θ[n_params + 1:end]) .* (w - θ[1:n_params])) +
            hyper_weight * θ[1:n_params]' * θ[1:n_params])
    )
end

function log_softmax(ŷ_item, y_item)
    max_val = maximum(ŷ_item)
    ((ŷ_item)[y_item] - max_val) - 
        log(sum(exp.(ŷ_item .- max_val)))
end

# log likelihood for binary classification
function binary_log_likelihood(m::Model, w::AbstractArray, 
        X::AbstractMatrix{Float64}, Y::AbstractMatrix{Float64})
    ŷ = pred(m.structure, w, X)

    return sum(Y .* HelperFunctions.s_log.(ŷ) + (1 .- Y) .* 
        HelperFunctions.s_log.(1 .- ŷ))
end

# log likelihood for multi-class classification
function multi_class_log_likelihood(m::Model, w::AbstractArray{Float64}, 
        X::AbstractMatrix{Float64}, y::AbstractArray{Int})
    ŷ = pred(m.structure, w, X) # matrix of Float64, column samples
    return sum(log_softmax.(eachcol(ŷ), y))
end

# log likelihood for mono-target regression
function regression_log_likelihood(m::Model, w::AbstractArray, 
        X::AbstractMatrix{Float64}, Y::AbstractMatrix{Float64}; 
        τ::Float64 = 100.0)
    ŷ = pred(m.structure, w, X)
    error_diff = ŷ - Y

    return -τ * (error_diff * error_diff')[1,1]
end

# log P(D|w)P(w)
function log_density(m::Model, W::AbstractArray, 
        X::AbstractMatrix{Float64}, Y::AbstractArray; coef = 1)
    mapped_log_likelihood = (m, X, Y) -> (w -> m.log_likelihood(m, w, X, Y))
    log_prior = m.log_prior.func(m.log_prior.θ)

    return mean(mapped_log_likelihood(m, X, Y).(eachcol(W))) #+
        coef * mean(log_prior.(eachcol(W)))
end

# forward pass of model architecture for data X and weights.
function pred(s::ModelStructure, weights::AbstractArray, 
        X::AbstractArray)

    # initialise forward pass state
    output = X                      # previous layer output
    w_index = 1                     # next index to start from for weights
    last_output_n = s.n_inputs      # last layer dimension

    # iterate over the layers of the network
    for l in s.layers
        (w_index, layer_weights) = extract_parameters(l, weights, w_index, last_output_n)
        output = forward(l, layer_weights, output)
        last_output_n = output_dimension(l)
    end
    return output
end

function count_params(layers::Vector, input_n::Int)
    foldl(
        ((n, last_n), l) -> (
            n + n_weights(l, last_n), 
            output_dimension(l)
        ),
        layers,
        init = (0, input_n)
    )[1]
end

function init_means(layers::Vector, input_n::Int)
    means = Vector(undef, length(layers))

    for i in 1:length(layers)
        if i == 1
            output_n = input_n
        else
            output_n = output_dimension(layers[i - 1])
        end

        means[i] = initialise_parameters(layers[i], output_n)
    end

    foldl(
        (acc, w) -> [acc;w],
        means,
        init = []
    ) |> vec
end

end