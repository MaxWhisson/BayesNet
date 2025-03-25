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
    optimiser_rule = Optimisers.Adam(0.1) 
    prior_optimisation_strategy = NONE
    random_seed = -1
end

struct TrainArgs
    loss_fn::Function
    training_params::TrainingParameters
end

function update_parameters!(ms::Vector, X_batch, y_batch, args, 
        i, M, optimiser_state)
    # calculate gradients for mini-batch:
    ∇θs = gradient(
        () -> args.loss_fn(ms, X_batch, y_batch, args.training_params, i, M),
        Params((m -> m.θ).(ms))
    )

    # optionally calculate gradients for prior of model
    if (args.training_params.prior_optimisation_strategy == PARALLEL)
        ∇θs_prior = gradient(
            () -> args.loss_fn(ms, X_batch, y_batch, args.training_params, i, M),
            Params((m -> m.log_prior.θ).(ms))
        )
        for m_i in 1:length(m)
            (ms[m_i].log_prior.optimiser_state, Δθ_prior) = Optimisers.apply!(
                ms[m_i].log_prior.optimiser_rule, 
                ms[m_i].log_prior.optimiser_state, 
                ms[m_i].log_prior.θ, 
                ∇θs_prior[ms[m_i].log_prior.θ]
            )
            ms[m_i].log_prior.θ = ms[m_i].log_prior.θ .- Δθ_prior
        end
    end

    # and update with optimiser:
    for m_i in 1:length(ms)
        (optimiser_state[m_i], Δθ) = Optimisers.apply!(
            args.training_params.optimiser_rule, 
            optimiser_state[m_i], 
            ms[m_i].θ, 
            ∇θs[ms[m_i].θ]
        )
        ms[m_i].apply_grad(ms[m_i], Δθ)
    end
    return optimiser_state
end

# train model 'm' on data 'X' and 'y'
function train!(ms::Vector, X::Matrix{Float64}, y::AbstractArray, args::TrainArgs)
    # for replicating results
    args.training_params.random_seed != -1 && Random.seed!(args.training_params.random_seed)

    optimiser_state = [Optimisers.init(args.training_params.optimiser_rule, m.θ) for m in ms]
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

            optimiser_state = update_parameters!(ms, X_batch, y_batch, args, batch_i + 1, no_batches, optimiser_state)
            losses[batch_i + 1] = args.loss_fn(ms, X_batch, y_batch, args.training_params, batch_i + 1, no_batches)
        end
        allLosses[epoch] = sum(losses) 
        if (epoch % 1 == 0) && (length(losses) > 0)
            @info "Mean loss of epoch $(epoch): $(mean(losses))"
        end
    end
    return allLosses
end

end