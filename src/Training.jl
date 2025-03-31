module Training

# type exports
export  TrainingParameters,
        TrainArgs

# function exports
export  train!

using Optimisers
using Random
using Zygote
using Statistics
using LinearAlgebra
using Distributions

import ..Models

Base.@kwdef struct TrainingParameters
    n_samples = 10
    max_epoch = 200
    batch_size = 20
    optimiser_rule = Optimisers.Adam(0.1) 
    prior_optimisation_strategy = "none"
    random_seed = -1
    is_natural = "none"
end

struct TrainArgs
    loss_fn::Function
    training_params::TrainingParameters
end

function compute_Fisher_diagonal(m;n_samples = 20)
    n_params = m.structure.n_total_params
    diag_mus = exp.(m.θ[n_params + 1:end]) .^ -2
    diag_sigmas = ones(n_params) .* 2
    # return diagm([diag_mus;diag_sigmas])
    return diagm(ones(n_params * 2))
end

function update_parameters!(ms::Vector, X_batch, y_batch, args, 
        i, M, optimiser_state)

    uses_natural_gd = args.training_params.is_natural

    # optionally calculate gradients for prior of model
    if (args.training_params.prior_optimisation_strategy == "parallel")
        ∇θs_prior = gradient(
            () -> args.loss_fn(ms, X_batch, y_batch, args.training_params, i, M),
            Params((m -> m.log_prior.θ).(ms))
        )
        for m_i in 1:length(ms)
            (ms[m_i].log_prior.optimiser_state, Δθ_prior) = Optimisers.apply!(
                ms[m_i].log_prior.optimiser_rule, 
                ms[m_i].log_prior.optimiser_state, 
                ms[m_i].log_prior.θ, 
                clean_grad.(∇θs_prior[ms[m_i].log_prior.θ])
            )
            ms[m_i].log_prior.θ = ms[m_i].log_prior.θ .- Δθ_prior
        end
    end

    # calculate gradients for mini-batch:
    ∇θs = gradient(
        () -> args.loss_fn(ms, X_batch, y_batch, args.training_params, i, M),
        Params((m -> m.θ).(ms))
    )

    # and update with optimiser:
    for m_i in 1:length(ms)
        if uses_natural_gd == "diagonal"
            println("a")
            ∇θs[ms[m_i].θ] = (compute_Fisher_diagonal(ms[m_i]) \ ∇θs[ms[m_i].θ])
        end
        (optimiser_state[m_i], Δθ) = Optimisers.apply!(
            args.training_params.optimiser_rule, 
            optimiser_state[m_i], 
            ms[m_i].θ, 
            clean_grad.(∇θs[ms[m_i].θ])
        )
        ms[m_i].apply_grad(ms[m_i], Δθ)
    end

    return optimiser_state
end

function clean_grad(g)
    g == nothing ? 0.0 : g
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