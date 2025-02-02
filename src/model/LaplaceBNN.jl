module LaplaceBNN

# function exports
export  LaplaceModel,
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
function LaplaceModel(prior::ParameterisedFunction, log_likelihood::Function, 
    n_inputs::Int, is_diagonal::Bool, layers::Vector{Layer})

    n_weights = n_inputs * layers[1].n +
        sum([layers[i].n * layers[i + 1].n for i in 1:length(layers) - 1])
    n_params = n_weights + length(layers)

    return LaplaceModel(
        ModelStructure(layers, n_inputs, n_weights_and_biases),
        randn(n_params),
        prior,
        log_likelihood
    )
end

function MAP_loss_fn(m, X_batch, y_batch, args)
    return log_density(m, m.θp[1:m.structure.n_total_params], X_batch, y_batch)
end

# find Hessian 
function fit_covariance!(m, X, y, args)
    H = hessian(Params([m.θ])) do
        log_density(m, m.θp[1:m.structure.n_total_params], X_batch, y_batch)
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