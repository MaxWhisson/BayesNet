module ModelTypes

export  Model,
        EvalStrategy,
        ModelStructure,
        ParameterisedFunction,
        VariationalModel,
        LaplaceModel,
        MCMC_Model,
        DegenerateModel

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
    
end