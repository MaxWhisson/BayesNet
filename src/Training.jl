module Training

# type exports
export  TrainingParameters,
        TrainArgs

# function exports
export  train!

using Optimisers

import ..Models

@enum PRIOR_OPTIMISATION NONE=1 PARALLEL=2 BLOCK=3

Base.@kwdef struct TrainingParameters
    n_samples = 1
    max_epoch = 50
    batch_size = 10
    optimiser_rule = Optimisers.Rprop() 
    prior_optimisation_strategy = NONE
    random_seed = -1
end

struct TrainArgs
    loss_fn::Function
    training_params::TrainingParameters
end

function update_parameters!(m, X_batch, y_batch, args::TrainArgs, i, M)
    # calculate gradients for mini-batch:
    ∇θ = gradient(Params([m.θ])) do
        args.loss_fn(m, X_batch, y_batch, args, i, M)
    end

    # optionally calculate gradients for prior of model
    if (args.training_params.prior_optimisation_strategy == PARALLEL)
        # TODO calculate grads for m.log_prior.θ and update
    end

    # and update with optimiser:
    (optimiser_state, ∇θ) = 
        Optimisers.apply!(args.training_params.optimiser_rule, optimiser_state, m.θ, ∇θ)
    m.θ = m.θ .- ∇θ
end

# train model 'm' on data 'X' and 'y'
function train!(m::Models.Model, X::Matrix{Float64}, y::Vector, args::TrainArgs)
    # for replicating results
    args.training_params.random_seed != -1 && Random.seed!(args.training_params.random_seed)

    optimiser_state = Optimisers.init(args.training_params.optimiser_rule, m.θ)
    no_batches = Int64(floor(length(y) / args.batch_size))
    losses = Vector(undef, no_batches)

    @info "Training"
    for epoch in 1:args.train_params.max_epoch
        indexes = shuffle(1:length(y))
        X, y = X[indexes,:], y[indexes]
        for batch_i in 0:no_batches - 1
            start_index = 1 + batch_i * args.training_params.batch_size
            end_index = (batch_i + 1) * args.training_params.batch_size

            X_batch = X[start_index:end_index, :]
            y_batch = y[start_index:end_index]

            update_parameters!(m, X_batch, y_batch, args.training_params, batch_i + 1, no_batches)
            losses[batch_i + 1] = args.loss_fn(m, X_batch, y_batch, args)
        end
        @info "Mean loss of epoch $(epoch): $(mean(losses))"
    end
end

end