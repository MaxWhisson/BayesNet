module VI_BNN

# function exports
export  VariationalFullGaussianModel,
        VariationalDiagonalGaussianModel,
        VariationalUnitGaussianModel,
        log_diagonal_gaussian_posterior,
        log_full_gaussian_posterior,
        log_unit_gaussian_posterior,
        gaussian_entropy,
        propagate_matrix_opp,
        set_col_matrix_expr,
        flow_transforms,
        uniform_complexity_cost,
        exponential_complexity_cost,
        variational_free_energy_creator,
        normalising_flow_density,
        transform_sample,
        sum_log_jacobian,
        predict_VI

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
function VariationalGaussianModel(prior_creator::Function, 
        log_likelihood::Function, n_inputs::Int, is_diagonal::Bool, 
        layers::Vector{Models.Layer}; 
        normalising_flow::AbstractArray = [])

    n_weights = n_inputs * layers[1].n +
        sum([layers[i].n * layers[i + 1].n for i in 1:length(layers) - 1])
    n_params = n_weights + sum(layers .|> (x -> x.n))

    n_variational_params = is_diagonal ? 
        n_params * 2 : 
        n_params + HelperFunctions.triangular(n_params)

    instantiated_flow = map(f -> f(n_params), normalising_flow) 
    n_flow_params = instantiated_flow != [] ? 
        sum((x -> x.n_params).(instantiated_flow)) : 0

    sampler = is_diagonal ? 
        diagonal_gaussian_sampler : 
        full_gaussian_sampler
    posterior = is_diagonal ? 
        log_diagonal_gaussian_posterior : 
        log_full_gaussian_posterior

    return VariationalModel(
        ModelStructure(layers, n_inputs, n_params),
        randn(n_variational_params + n_flow_params),
        sampler,
        prior_creator(n_params),
        log_likelihood,
        posterior,

        instantiated_flow,
        n_variational_params,
        n_flow_params
    )
end

function VariationalUnitGaussianModel(log_likelihood::Function, 
        n_inputs::Int, layers::Vector{Models.Layer}; 
        normalising_flow::AbstractArray = [])

    n_weights = n_inputs * layers[1].n +
        sum([layers[i].n * layers[i + 1].n for i in 1:length(layers) - 1])
    n_params = n_weights + sum(layers .|> (x -> x.n))

    instantiated_flow = map(f -> f(n_params), normalising_flow) 
    n_flow_params = instantiated_flow != [] ? 
        sum((x -> x.n_params).(instantiated_flow)) : 0

    return VariationalModel(
        ModelStructure(layers, n_inputs, n_params),
        # [100, 0, 2, 2],
        randn(n_flow_params),
        Samplers.unit_gaussian_sampler,
        diagonal_gaussian_prior_creator(n_params),
        log_likelihood,
        log_unit_gaussian_posterior,

        instantiated_flow,
        0,
        n_flow_params
    )
end

# simpler constructor for model architectures with full Gaussian weights.
function VariationalFullGaussianModel(log_likelihood::Function, n_inputs::Int,
        layers::Vector{Models.Layer}; 
        normalising_flow::AbstractArray = [])
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
function VariationalDiagonalGaussianModel(log_likelihood::Function, 
        n_inputs::Int, layers::Vector{Models.Layer}; 
        normalising_flow::AbstractArray = [])
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
function log_diagonal_gaussian_posterior(m::Models.Model, 
        samples::AbstractMatrix{Float64})
    variational_params = m.θ[1:m.n_variational_params]
    n_params = m.structure.n_total_params

    log_posterior_pdf = sample -> logpdf(MvNormal(
        variational_params[1:n_params], 
        Diagonal(log.(1 .+ exp.(variational_params[n_params + 1:end])) .^ 2)
    ), sample)
    return log_posterior_pdf.(eachcol(samples))
end

#evaluate log density on unit gaussian
function log_unit_gaussian_posterior(m::Models.Model, 
        samples::AbstractMatrix{Float64})
    n_params = m.structure.n_total_params
    log_posterior_pdf = sample -> logpdf(MvNormal(zeros(n_params), I), sample)
    return log_posterior_pdf.(eachcol(samples))
end

# evaluate log density of samples on full gaussian
function log_full_gaussian_posterior(m::Models.Model, 
        samples::AbstractMatrix{Float64})
    variational_params = m.θ[1:m.n_variational_params]
    n_params = m.structure.n_total_params
    L = HelperFunctions.to_lower_triangular(
        log.(1 .+ exp.(variational_params[n_params + 1:end]))
    )

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

function uniform_complexity_cost(M::Int, i::Int)
    return 1/M
end

function exponential_complexity_cost(M::Int, i::Int)
    return (2 ^ (M - i)) / (2 ^ M - 1)
end

function normalising_flow_density(m::Models.Model, 
        flow_params::AbstractArray{Float64})
    transformation = transform_sample(m.normalising_flow, flow_params)
    sum_log_jacobian_fn = sum_log_jacobian(m.normalising_flow, flow_params)
    function density(w₀)
        samples = transformation.(eachcol(w₀))
        samples = reduce(hcat, samples)
        variational_expectation = mean(m.log_posterior(m, w₀))
        flows_E = mean(sum_log_jacobian_fn.(eachcol(w₀)))

        res = exp(variational_expectation - flows_E)
        return res
    end
end

function transform_sample(
        normalising_flow::AbstractArray{Models.NormalisingFlowLayer}, 
        flow_params::Vector{Float64})
    function f(w)
        param_index = 1
        for i in 1:length(normalising_flow)
            w = normalising_flow[i].func(
                flow_params[
                    param_index:param_index + normalising_flow[i].n_params - 1
                ]
            )(w)
            param_index += normalising_flow[i].n_params
        end
        return w
    end
end

function sum_log_jacobian(
        normalising_flow::AbstractArray{Models.NormalisingFlowLayer}, 
        flow_params::AbstractArray{Float64})
    function f(w::AbstractArray{Float64})
        log_jacobian_sum = 0
        param_index = 1
        for i in 1:length(normalising_flow)
            layer_params = flow_params[
                param_index:param_index + normalising_flow[i].n_params - 1
            ]
            log_jacobian_sum += (normalising_flow[i].jacobian_determinant(
                layer_params
            )(w) |> abs |> log)
            w = normalising_flow[i].func(layer_params)(w)
            param_index += normalising_flow[i].n_params
        end
        return log_jacobian_sum
    end
end

function evaluate_normalising_flow(m::Models.Model, 
        w₀::AbstractArray{Float64})
    flow_params = m.θ[m.n_variational_params + 1:end]
    if (m.normalising_flow != [])
        samples = transform_sample(
            m.normalising_flow, flow_params
        ).(eachcol(w₀))
        samples = reduce(hcat, samples)
        
        flows_E = mean(sum_log_jacobian(
            m.normalising_flow, flow_params
        ).(eachcol(w₀)))
    else
        samples = w₀
        flows_E = 0
    end
    return (samples, flows_E)
end

function evaluate_variational_posterior_expectation(m::Models.Model, 
        is_closed_form_gaussian::Bool, w₀::AbstractArray{Float64})
    if (is_closed_form_gaussian)
        return -gaussian_entropy(
            variational_params[m.structure.n_total_params + 1:end], 
            m.structure.n_total_params
        )
    end
    return mean(m.log_posterior(m, w₀))
end

# VI training function to optimise
function variational_free_energy_creator(is_closed_form_gaussian::Bool, 
        coef_func::Function; 
        custom_log_density::Tuple{Bool, Function} = (false, x->x))
    function f(m::Models.Model, X::AbstractMatrix{Float64}, 
            y::AbstractArray, args::Training.TrainingParameters, 
            i::Int, M::Int)
        coef = coef_func(M, i)

        # samples from approximate posterior
        variational_params = m.θ[1:m.n_variational_params]
        w₀ = m.weight_sampler(
            variational_params, 
            args.n_samples, 
            m.structure.n_total_params
        )

        # normalising flow then E_Q[Q()]
        (samples, flows_E) = evaluate_normalising_flow(m, w₀)  
        variational_expectation = evaluate_variational_posterior_expectation(
            m, is_closed_form_gaussian, w₀)

        if (custom_log_density[1])
            log_joint_distribution = mean(custom_log_density[2](samples))
        else
            log_joint_distribution = mean(
                log_density(m, samples, X, y, coef = coef)
            )
        end

        return coef * (variational_expectation - flows_E) - 
            log_joint_distribution
    end
end

function sample_model(m::Models.Model, n_samples::Integer)
    W = m.weight_sampler(
        m.θ[1:m.n_variational_params], 
        n_samples,
        m.structure.n_total_params,
    )
    if (m.normalising_flow != [])
        flow_params = m.θ[m.n_variational_params + 1:end]
        f = transform_sample(m.normalising_flow, flow_params)
        W = reduce(hcat, f.(eachcol(W)))
    end
    return W
end

function AL_datapoint_uncertainty(m::Models.Model, W::AbstractMatrix{Float64},
        x::AbstractArray{Float64})
    coef1 = 1 / size(W)[2]
    coef2 = 1 / (size(W)[2] - 1)

    preds = map(w -> pred(m.structure, w, x), eachcol(W))
    ŷ = coef1 * sum(preds, dims=2)
    f = pred -> (pred - ŷ) * (pred - ŷ)'
    Σ = coef2 * sum(map(f, eachcol(preds)))
    return det(Σ)
end

function find_uncertainties(m::Models.Model, X::AbstractMatrix{Float64}, 
        y::AbstractArray, U::AbstractMatrix{Float64}, 
        args::Training.TrainingParameters, n_active_samples::Int)
    Training.train!(m, X, y, args)
    W = sample_model(m, n_active_samples)

    uncertainties = map(x -> AL_datapoint_uncertainty(m, W, x), eachcol(U))
    return uncertainties
end

function active_learning(m::Models.Model, X::AbstractMatrix{Float64}, 
        y::AbstractArray, U::AbstractMatrix{Float64}, 
        oracle::Function, args::Training.TrainArgs; 
        n_active_samples::Int = 10, threshold::Float64 = -Inf)
    max_uncertainty = Inf
    init_x_length = size(X)[2]

    while (U != []) && (max_uncertainty > threshold)
        uncertainties = find_uncertainties(m, X, y, U, args, n_active_samples)
        max_uncertainty_i = argmax(uncertainties)[2]
        max_uncertainty = uncertainties[max_uncertainty_i]
        Ux_max = U[:,max_uncertainty_i]

        U = U[1:end, 1:end .!= max_uncertainty_i]
        X = [X Ux_max]
        y = [y oracle(Ux_max)]
    end

    return X[init_x_length + 1:end]
end

function predict_VI(m::Models.Model, X::AbstractMatrix{Float64}; n_samples::Int = 1)
    W = m.weight_sampler(m.θ, n_samples, m.structure.n_total_params)
    f = transform_sample(m.normalising_flow, m.θ[m.n_variational_params + 1:end])
    mean((w -> pred(m.structure, f(w), X)).(eachcol(W)))
end

end