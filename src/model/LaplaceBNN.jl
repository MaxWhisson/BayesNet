module LaplaceBNN

# function exports
export  BuildLaplaceModel,
        MAP_loss_fn,
        fit_covariance!,
        fit_gaussian!

# dependencies:
using LinearAlgebra
using Distributions
using Optimisers
using Random

using ..Models
using ..Training

# constructor for Laplace models
function BuildLaplaceModel(priorCreator::Function, log_likelihood::Function, 
    n_inputs::Int, layers::Vector{Layer})

    n_weights = n_inputs * layers[1].n +
        sum([layers[i].n * layers[i + 1].n for i in 1:length(layers) - 1])
    n_params = n_weights + sum(layers .|> (x -> x.n))

    return Models.LaplaceModel(
        Models.ModelStructure(layers, n_inputs, n_params),
        randn(n_params),        # weights
        priorCreator(n_params),
        log_likelihood
    )
end

function MAP_loss_fn(m::Models.LaplaceModel, X_batch::AbstractMatrix{Float64},
        y_batch::AbstractArray, args::Training.TrainingParameters)
    return log_density(m, m.θ[1:m.structure.n_total_params], X_batch, y_batch)
end

# find Hessian 
function fit_covariance!(m::Models.LaplaceModel, X::AbstractMatrix{Float64}, 
        y::AbstractArray, args::Training.TrainingParameters)
    H = hessian(Params([m.θ])) do
        log_density(m, m.θ[1:m.structure.n_total_params], X_batch, y_batch)
    end
    m.θ[m.structure.n_total_params + 1:end] = reshape(H, (1,:))
end

function fit_gaussian!(m, X, y, train_params::TrainingParameters)
    args = TrainArgs(train_params, MAP_loss_fn)
    # learn first portion of m.θ (MAP)
    Training.train!(m, X, y, args)
    fit_covariance!(m, X, y, args)
end

end