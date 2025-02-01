using Random
using LinearAlgebra

# sample diagonal Gaussian variational θ n_samples times.
function diagonal_gaussian_sampler(θ, n_samples, n_params)
    μ, log_σ = θ[1:n_params], θ[n_params + 1:end]
    return μ .+ (exp.(log_σ) .* randn(n_params, n_samples))
end

# sample full Gaussian variational θ n_samples times.
function full_gaussian_sampler(θ, n_samples, n_params)
    μ, L = θ[1:n_params], reshape(exp.(θ[n_params + 1:end]), (n_params, n_params))
    return μ .+ (L * randn(n_params, n_samples))
end