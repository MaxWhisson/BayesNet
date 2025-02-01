using LinearAlgebra
using Random
using Distributions
using Zygote

include("Models.jl")

# constructor for MCMC models
function MCMC_Model(prior::ParameterisedFunction, log_likelihood::Function, n_inputs::Int, 
        layers::Vector{Layer})

    n_weights = n_inputs * layers[1].n +
        sum([layers[i].n * layers[i + 1].n for i in 1:length(layers) - 1])

    return MCMC_Model(
        ModelStructure(layers, n_inputs, n_weights_and_biases),
        prior,
        log_likelihood
    )
end

# produce weight samples
# TODO every nth sample and burn in
function adaptive_MCMC_sampler(m, X, y, n_samples; α′ = 0.25, pᵦ = 0.02)
    weights = Matrix{Float64}(undef, m.n_weights, n_samples)
    w_sample = randn(m.n_weights)

    L = Matrix(I, (m.n_weights, m.n_weights))
    rule = Optimisers.Rprop()
	optimiserState = Optimisers.init(rule, L)

    β = 1

    for i in 1:n_samples
        ϵ = rand(MvNormal(zeros(length(L)), I))
        w_sample′ = w_sample + (L * ϵ)

        # equivalent to log(P(w'|D)/P(w|D))
        α = log_density(m, w_sample′, X, y) - log_density(m, w_sample, X, y)
        (w_sample, accepted) = rand(Uniform()) <= exp(α) ? 
            (w_sample′, true) : 
            (w_sample, false)
        weights[:,i] = w_sample

        ∇L = gradient(Params([L])) do
            β * tr(L) + min(0, α)
        end

        (optimiserState, grad) = Optimisers.apply!(rule, optimiserState, L, ∇L)
        L = L .- grad

        β = β(1 + pᵦ(accepted - α′))
    end
    return weights
end

# Langevin dynamics gradient based sampler
function Langevin_dynamics_MCMC_sampler(m, X, y, args; a = 0.1, 
        b = 0.1, γ = 0.75)

    batch_size = args.training_params.batch_size
    w = randn(m.n_weights)
    no_batches = Int64(floor(length(y) / batch_size))
    weights = Matrix{Float64}(undef, m.n_weights, args.train_params.max_epoch * no_batches)
    t = 1

    for epoch in 1:args.train_params.max_epoch
        indexes = shuffle(1:end)
        X, y = X[indexes,:], y[indexes]
        for batch_i in 0:no_batches - 1
            start_index = 1 + batch_i * batch_size
            end_index = (batch_i + 1) * batch_size

            X_batch = X[start_index:end_index, :]
            y_batch = y[start_index:end_index]

            log_prior_grad = gradient(Params([w])) do
                m.log_prior.func(m.log_prior.θ)(w)
            end

            log_likelihood_grad = gradient(Params([w])) do
                m.log_likelihood(m, w, X_batch, y_batch)
            end

            ϵₜ = a * (b + t) ^ (-γ)
            ηₜ = randn(length(θ)) .* ϵₜ # spherical Gaussian
            θ -= ϵₜ/2 .* (log_prior_grad[1] + ((length(y) / batch_size) .* log_likelihood_grad[1]))
            weights[:,t] = θ
            t += 1
        end
    end
    return weights
end