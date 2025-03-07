module Training

# type exports
export  TrainingParameters,
        TrainArgs,
        PRIOR_OPTIMISATION

# function exports
export  train!

using Optimisers
using Random
using Zygote
using Statistics

import ..Models

@enum PRIOR_OPTIMISATION NONE=1 PARALLEL=2 BLOCK=3

Base.@kwdef struct TrainingParameters
    n_samples = 10
    max_epoch = 200
    batch_size = 20
    optimiser_rule = Optimisers.Adam() 
    prior_optimisation_strategy = NONE
    random_seed = -1
end

struct TrainArgs
    loss_fn::Function
    training_params::TrainingParameters
end

function update_parameters!(m, X_batch, y_batch, args, i, M, optimiser_state)
    # calculate gradients for mini-batch:
    ∇θ = gradient(
        () -> args.loss_fn(m, X_batch, y_batch, args.training_params, i, M),
        Params([m.θ])
    )[m.θ]

    # optionally calculate gradients for prior of model
    if (args.training_params.prior_optimisation_strategy == PARALLEL)
        # TODO calculate grads for m.log_prior.θ and update
    end

    # and update with optimiser:
    (optimiser_state, Δθ) = 
        Optimisers.apply!(args.training_params.optimiser_rule, optimiser_state, m.θ, ∇θ)
    m.θ = m.θ .- Δθ
    return optimiser_state
end

# train model 'm' on data 'X' and 'y'
function train!(m::Models.Model, X::Matrix{Float64}, y::AbstractArray, args::TrainArgs)
    # for replicating results
    args.training_params.random_seed != -1 && Random.seed!(args.training_params.random_seed)

    optimiser_state = Optimisers.init(args.training_params.optimiser_rule, m.θ)
    no_batches = Int64(floor(length(y) / args.training_params.batch_size))
    allLosses = Vector(undef, args.training_params.max_epoch)
    losses = Vector(undef, no_batches)

    @info "Training"
    for epoch in 1:args.training_params.max_epoch
        indexes = shuffle(1:length(y))
        X, y = X[:,indexes], y[indexes]
        for batch_i in 0:no_batches - 1
            start_index = 1 + batch_i * args.training_params.batch_size
            end_index = (batch_i + 1) * args.training_params.batch_size

            X_batch = X[:, start_index:end_index]
            y_batch = y[start_index:end_index]

            optimiser_state = update_parameters!(m, X_batch, y_batch, args, batch_i + 1, no_batches, optimiser_state)
            losses[batch_i + 1] = args.loss_fn(m, X_batch, y_batch, args.training_params, batch_i + 1, no_batches)
        end
        allLosses[epoch] = sum(losses) 
        if (epoch % 20 == 0) && (length(losses) > 0)
            @info "Mean loss of epoch $(epoch): $(mean(losses))"
        end
    end
    return allLosses
end

end