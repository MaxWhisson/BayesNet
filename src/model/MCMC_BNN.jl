module MCMC_BNN

# function exports
export  Build_MCMC_Model,
        adaptive_MCMC_sampler,          # exposed for testing
        langevin_dynamics_MCMC_sampler, # exposed for testing
        sample_posterior_MCMC,
        predict_MCMC,
        proportion_accepted

using LinearAlgebra
using Random
using Distributions
using Zygote
using Optimisers

using ..Training
using ..Models
using ..HelperFunctions

# constructor for MCMC models
function Build_MCMC_Model(priorCreator::Function, log_likelihood::Function,
        n_inputs::Int, layers::Vector{Layer})

    n_weights = n_inputs * layers[1].n +
        sum([layers[i].n * layers[i + 1].n for i in 1:length(layers) - 1])
    n_params = n_weights + sum(layers .|> (x -> x.n))

    return MCMC_Model(
        ModelStructure(layers, n_inputs, n_params),
        priorCreator(n_params),
        log_likelihood
    )
end

# produce weight samples
function adaptive_MCMC_sampler(m::Models.MCMC_Model, 
        X::AbstractMatrix{Float64}, y::AbstractArray; n_samples::Int = 22000,
        α′::Float64 = 0.25, pᵦ::Float64 = 0.02)

    weights = Matrix{Float64}(undef, m.structure.n_total_params, n_samples)
    w_sample = randn(m.structure.n_total_params)

    L = HelperFunctions.init_L_diagonal_cov(m.structure.n_total_params) .|> 
        exp .|> 
        (x -> log(1 + x)) |> 
        (x -> HelperFunctions.to_lower_triangular(
            x, m.structure.n_total_params
        ))
    β = 1

    rule = Optimisers.Adam()
    optimState = Optimisers.init(rule, L[:])
    accepted_count = 0

    for i in 1:n_samples
        ϵ = randn(m.structure.n_total_params)
        w_sample′ = w_sample + (L * ϵ)

        # equivalent to log(P(w'|D)/P(w|D))
        α = log_density(m, w_sample′, X, y) - log_density(m, w_sample, X, y)

        ∇L = β .* diagm(1 ./ diag(L))
        if exp(α) < 1
            ∇L += jacobian(wₜ -> (
                log_density(m, wₜ, X, y) * ϵ'
            ), w_sample′)[1] |> LowerTriangular |> diag |> diagm
        end

        (optimState, grad) = Optimisers.apply!(rule, optimState, L[:], ∇L[:])
        L = reshape(L[:] .+ grad, size(L))

        (w_sample, accepted) = rand(Uniform()) <= exp(α) ? 
            (w_sample′, true) : (w_sample, false)
        weights[:,i] = w_sample
        accepted_count += accepted

        β = β * (1 + pᵦ * (accepted - α′))

        if i % 1000 == 0
            @info "acceptance rate after sample $(i) is $(accepted_count / i), β is $(β)"
        end
    end
    return weights
end

# Langevin dynamics gradient based sampler
function langevin_dynamics_MCMC_sampler(m::Models.MCMC_Model, 
        X::AbstractMatrix{Float64}, y::AbstractArray, 
        args::Training.TrainingParameters)

    batch_size = args.batch_size
    w = randn(m.structure.n_total_params)
    no_batches = Int64(floor(length(y) / batch_size))
    weights = Matrix{Float64}(
        undef, 
        m.structure.n_total_params, 
        args.max_epoch * no_batches
    )
    t = 1

    N = length(y)
    n = batch_size

    optimiser_rule = Optimisers.Adam() 
    optimiser_state = Optimisers.init(optimiser_rule, w)

    for epoch in 1:args.max_epoch
        indexes = shuffle(1:length(y))
        X, y = X[:,indexes], y[indexes]
        for batch_i in 0:no_batches - 1
            start_index = 1 + batch_i * batch_size
            end_index = (batch_i + 1) * batch_size

            X_batch = X[:,start_index:end_index]
            y_batch = y[start_index:end_index]

            ∇w = gradient(Params([w])) do
                -log_density(m, w, X, y, coef = (n/N))
            end[w]

            (optimiser_state, Δw) = Optimisers.apply!(optimiser_rule, optimiser_state, w, ∇w)
            ϵₜ = Δw ./ ∇w
            ηₜ = randn(length(w)) .* ϵₜ # spherical Gaussian
            w = w .- Δw + ηₜ
            weights[:,t] = w
            
            t += 1
        end
        if epoch % 20 == 0
            @info "finished epoch $(epoch)"
        end
    end
    return weights
end

function predict_MCMC(m::Models.MCMC_Model, X::AbstractMatrix{Float64}, 
        W::AbstractMatrix{Float64})
    mean((w -> pred(m.structure, w, X)).(eachcol(W)))
end

function proportion_accepted(weights::AbstractMatrix{Float64})
    foldl(
        (count, i) -> count + (weights[:,i - 1] != weights[:,i]),
        2:size(weights)[2],
        init = 0
    ) / size(weights)[2]
end

end