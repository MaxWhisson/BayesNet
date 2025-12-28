module Models

# type exports
export  EvalStrategy,
        Model,
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
        produceMultiModel

using Statistics
using LinearAlgebra
using Zygote
using Distributions
using LogExpFunctions
using Optimisers

using ..Layer:NNLayer
using ..Normalising

using ..Planar
using ..Radial

using ..DenseLayer
using ..ResidualLayer
using ..LSTMLayer

import ..HelperFunctions: trace, deep_foldl, set_vector_elem!
using ..ModelTypes: 
        Model, 
        EvalStrategy,
        ModelStructure,
        ParameterisedFunction,
        VariationalModel,
        LaplaceModel,
        MCMC_Model,
        DegenerateModel

using ..ModelFunctions: init_means,
        count_params

abstract type Model end

EvalStrategy = Vector{Tuple{Int, Vector{Int}}}

# struct for creating model architectures.
struct ModelStructure
    layers::Vector{<:NNLayer}
    evaluation::EvalStrategy
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

# simple evaluation strategy for generic feed-forward NN
function create_simple_evaluation(l::Vector{<:NNLayer})
    [(i, [i == 1 ? -1 : i - 1]) for i in 1:length(l)]
end

function simple_apply_grad(m::Models.Model, g)
    m.θ = m.θ .- g
end

function produce_degenerate(layers::Vector, n_inputs::Int)
    evalOrder = create_simple_evaluation(layers)
    n_params = count_params(layers, evalOrder, [n_inputs])
    DegenerateModel(
        simple_apply_grad,
        ModelStructure(layers, evalOrder, n_inputs, n_params),
        init_means(layers, evalOrder, [n_inputs])
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
    ŷ = pred(m.structure, w, X)[1][end]
    error_diff = ŷ - Y

    return -τ * (error_diff * error_diff')[1,1]
end

# log P(D|w)P(w)
function log_density(m::Model, W::AbstractArray, 
        X::AbstractMatrix{Float64}, Y::AbstractArray; coef = 1)
    mapped_log_likelihood = (m, X, Y) -> (w -> m.log_likelihood(m, w, X, Y))
    log_prior = m.log_prior.func(m.log_prior.θ)

    return mean(mapped_log_likelihood(m, X, Y).(eachcol(W)))
        coef * mean(log_prior.(eachcol(W)))
end

function get_layer_output(li::Int, layers::Vector{<:NNLayer}, outputs::AbstractVector, 
        passState::AbstractArray, evaluationStatus::AbstractVector)
    return evaluationStatus[li] == 1 ? outputs[li] : get_layer_state(layers[li], passState[li])
end

function pred(s::ModelStructure, weights::AbstractArray, X::AbstractMatrix)
    initState = [repeat(init_state(l), 0, size(X,2)) for l in s.layers]
    pred(s, weights, X, initState)
end

function evaluateLayer(
        s::ModelStructure, 
        outputs::AbstractVector, 
        passState::AbstractVector{<:AbstractArray{Float64}}, 
        evaluationStatus, 
        initState,
        i::Int64, 
        X, 
        weights, 
        w_index)
    inputs = s.evaluation[i][2] .|> (layer_i -> 
        layer_i == 0 ? X :
        get_layer_output(layer_i, s.layers, outputs, passState, evaluationStatus))
    
    eval_layer_i = s.evaluation[i][1]
    (w_index, layer_weights) = extract_parameters(
        s.layers[eval_layer_i], 
        weights, 
        w_index, 
        size.(inputs, 1)
    )

    (forwardOutput, newState) = forward(
        s.layers[eval_layer_i], 
        layer_weights, 
        inputs, 
        initState[eval_layer_i]
    )

    (
        w_index,
        [passState[1:eval_layer_i - 1];[newState];passState[eval_layer_i + 1:end]],
        [outputs[1:eval_layer_i - 1];[forwardOutput];outputs[eval_layer_i + 1:end]],
        [evaluationStatus[1:eval_layer_i - 1];1;evaluationStatus[eval_layer_i + 1:end]]
    )
end

function pred(s::ModelStructure, weights::AbstractArray,
        X::AbstractMatrix, initState::AbstractVector{<:AbstractVector{Float64}})
    res = deep_foldl(
        ((
            w_index,
            passState,
            outputs,
            evaluationStatus
        ), i) -> evaluateLayer(
            s, 
            outputs,
            passState, 
            evaluationStatus, 
            initState,
            i, 
            X,
            weights,
            w_index
        ),
        eachindex(s.evaluation),
        (
            1, 
            fill(zeros(0), length(s.layers))::Vector{<:AbstractArray{Float64}},
            fill(zeros(0), length(s.layers))::Vector{<:AbstractArray{Float64}},
            zeros(length(s.layers))
        )
    )[[3, 2]]
    return res
end

end