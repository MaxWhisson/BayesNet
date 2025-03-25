module Samplers

# function exports
export  diagonal_gaussian_sampler, 
        full_gaussian_sampler,
        unit_gaussian_sampler

using Random
using LinearAlgebra

using ..HelperFunctions

# sample diagonal Gaussian variational θ n_samples times.
function diagonal_gaussian_sampler(θ, n_samples, n_params)
    μ, log_σ = θ[1:n_params], θ[n_params + 1:end]
    return μ .+ (exp.(log_σ) .* randn(n_params, n_samples))
end

# sample full Gaussian variational θ n_samples times.
function full_gaussian_sampler(θ, n_samples, n_params)
    μ, L♭ = θ[1:n_params], exp.(θ[n_params + 1:end])
    L = HelperFunctions.to_lower_triangular(L♭, n_params)
    return μ .+ (L * randn(n_params, n_samples))
end

# sample unit multivariate Gaussian, don't need θ
function unit_gaussian_sampler(θ, n_samples, n_params)
    return randn(n_params, n_samples)
end

end