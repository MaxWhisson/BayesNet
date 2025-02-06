module VI_BNN

# function exports
export  VariationalFullGaussianModel,
        VariationalDiagonalGaussianModel,
        log_diagonal_gaussian_posterior,
        log_full_gaussian_posterior,
        gaussian_entropy,
        propagate_matrix_opp,
        set_col_matrix_expr,
        flow_transforms,
        uniform_complexity_cost,
        exponential_complexity_cost,
        variational_free_energy_creator

# dependencies:
using LinearAlgebra
using Distributions
using Optimisers
using Random

using ..Samplers
using ..Models
using ..Training
using ..HelperFunctions

# constructor for Gaussian Variational models
function VariationalGaussianModel(prior_creator::Function, log_likelihood::Function, 
        n_inputs::Int, is_diagonal::Bool, layers::Vector{Models.Layer}; 
        normalising_flow = [],
        variational_generator = (X, m) -> m.θ[1:m.n_variational_params])

    n_weights = n_inputs * layers[1].n +
        sum([layers[i].n * layers[i + 1].n for i in 1:length(layers) - 1])
    n_params = n_weights + sum(layers .|> (x -> x.n))

    n_variational_params = is_diagonal ? n_params * 2 : n_params + HelperFunctions.triangular(n_params)

    instantiated_flow = map(f -> f(n_params), normalising_flow) 
    n_flow_params = instantiated_flow != [] ? sum((x -> x.n_params).(instantiated_flow)) : 0

    sampler = is_diagonal ? diagonal_gaussian_sampler : full_gaussian_sampler
    posterior = is_diagonal ? log_diagonal_gaussian_posterior : log_full_gaussian_posterior

    return VariationalModel(
        ModelStructure(layers, n_inputs, n_params),
        abs.(randn(n_variational_params + n_flow_params)), # TODO replace this with function call
        sampler,
        prior_creator(n_params),
        log_likelihood,
        posterior,

        variational_generator,

        instantiated_flow,
        n_variational_params,
        n_flow_params
    )
end

# simpler constructor for model architectures with full Gaussian weights.
function VariationalFullGaussianModel(log_likelihood::Function, n_inputs::Int, 
    layers::Vector{Models.Layer}; normalising_flow = [])
VariationalGaussianModel(
    diagonal_gaussian_prior_creator,
    log_likelihood, 
    n_inputs, 
    false, 
    layers, 
    normalising_flow = normalising_flow
)
end

# simpler constructor for model architectures with diagonal Gaussian weights.
function VariationalDiagonalGaussianModel(log_likelihood::Function, n_inputs::Int, 
    layers::Vector{Models.Layer}; normalising_flow = [])
VariationalGaussianModel(
    diagonal_gaussian_prior_creator,
    log_likelihood, 
    n_inputs, 
    true, 
    layers, 
    normalising_flow = normalising_flow
)
end

# evaluate log density of samples on diagonal gaussian
function log_diagonal_gaussian_posterior(m, samples)
    variational_params = m.θ[1:m.n_variational_params]
    n_params = m.structure.n_total_params

    log_posterior_pdf = sample -> logpdf(MvNormal(
        variational_params[1:n_params], 
        Diagonal(log.(1 .+ exp.(variational_params[n_params + 1:end])) .^ 2)
    ), sample)
    return log_posterior_pdf.(eachcol(samples))
end

# evaluate log density of samples on full gaussian
function log_full_gaussian_posterior(m, samples)
    variational_params = m.θ[1:m.n_variational_params]
    n_params = m.structure.n_total_params
    L = HelperFunctions.to_lower_triangular(log.(1 .+ exp.(variational_params[n_params + 1:end])))

    log_posterior_pdf = sample -> logpdf(MvNormal(
        variational_params[1:n_params], 
        L * L'
    ), sample)
    return log_posterior_pdf.(eachcol(samples))
end

# closed form solution to integral of variational posterior for Gaussians as
# used by BBVI.
function gaussian_entropy(log_σ::Vector{Float64}, dims::Int)
	return 0.5 * dims * (1 + log(2π) + sum(log_σ))
end

function uniform_complexity_cost(M, i)
    return 1/M
end

function exponential_complexity_cost(M, i)
    return (2 ^ (M - i)) / (2 ^ M - 1)
end

function encoder_creator(structure::Models.ModelStructure)
    function f(X, m)
        pred(structure, m.θ[1:m.n_variational_params], X)
    end
    return f
end

function transform_sample(m::Models.VariationalModel, flow_params::Vector{Float64})
    function f(w)
        param_index = 1
        for i in 1:length(m.normalising_flow)
            w = m.normalising_flow[i].func(
                flow_params[param_index:param_index + m.normalising_flow[i].n_params - 1]
            )(w)
            param_index += m.normalising_flow[i].n_params
        end
        return w
    end
end

function sum_log_jacobian(m, flow_params)
    function f(w)
        log_jacobian_sum = 0
        param_index = 1
        for i in 1:length(m.normalising_flow)
            w = m.normalising_flow[i].func(
                flow_params[param_index:param_index + m.normalising_flow[i].n_params - 1]
            )(w)
            log_jacobian_sum += m.normalising_flow[i].jacobian_determinant(
                flow_params[param_index:param_index + m.normalising_flow[i].n_params - 1]
            )(w) .|> abs .|> log
            param_index += m.normalising_flow[i].n_params
        end
        return log_jacobian_sum
    end
end

# VI training function to optimise
# in the case of encoder architectures, m.variational_params is the encoder weights
function variational_free_energy_creator(is_closed_form_gaussian::Bool, coef_func::Function; 
        uses_encoder = false)
    function f(m, X, y, args, i, M)
        coef = coef_func(M, i)

        # samples from approximate posterior
        variational_params = m.variational_param_generator(X, m)
        flow_params = m.θ[m.n_variational_params + 1:end]

        w₀ = m.weight_sampler(variational_params, args.n_samples, m.structure.n_total_params)

        if (m.normalising_flow != [])
            samples = transform_sample(m, flow_params).(eachcol(w₀))
            samples = reduce(hcat, samples)
            flows_E = mean(sum_log_jacobian(m, flow_params).(eachcol(w₀)))
        else
            samples = w₀
            flows_E = 0
        end

        if (is_closed_form_gaussian)
            variational_expectation = -gaussian_entropy(
                variational_params[m.structure.n_total_params + 1:end], 
                m.structure.n_total_params
            )
        else
            variational_expectation = mean(m.log_posterior(m, w₀))
        end

        log_joint_distribution = mean(log_density(m, samples, X, y, coef = coef))

        return coef * (variational_expectation - flows_E) - log_joint_distribution
    end
end

end