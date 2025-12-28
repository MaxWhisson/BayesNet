module VI_BNN

# function exports
export  VariationalFullGaussianModel,
        VariationalDiagonalGaussianModel,
        VariationalUnitGaussianModel,
        VariationalGaussianModel,
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
        predict_VI,
        active_learning!,
        Langevin_Stein_apply_gradient,
        create_variational_operator,
        Langevin_Stein_objective,
        KL_divergence_objective

# dependencies:
using LinearAlgebra
using Distributions
using Optimisers
using Random
using Zygote
using SpecialFunctions

using ..Samplers
using ..Models
using ..Training
using ..HelperFunctions
using ..ModelFunctions: init_means,
        count_params

function VariationalUnitGaussianModel(log_likelihood::Function, 
        n_inputs::Int, layers::Vector; 
        normalising_flow::AbstractArray = [])

    n_weights = n_inputs * layers[1].n +
        sum([layers[i].n * layers[i + 1].n for i in 1:length(layers) - 1])
    n_params = n_weights + sum(layers .|> (x -> x.n))

    instantiated_flow = map(f -> f(n_params), normalising_flow) 
    n_flow_params = instantiated_flow != [] ? 
        sum((x -> x.n_params).(instantiated_flow)) : 0

    return VariationalModel(
        Models.simple_apply_grad,
        ModelStructure(layers, n_inputs, n_params),
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

# constructor for Gaussian Variational models
function VariationalGaussianModel(prior_creator::Function, 
        log_likelihood::Function, n_inputs::Int, is_diagonal::Bool, 
        layers::Vector; 
        normalising_flow::AbstractArray = [], 
        init_param_fn = (n ->(n[1], randn(n[1] + n_f[2]))))

    n_params = count_params(layers, [n_inputs])

    instantiated_flow = map(f -> f(n_params), normalising_flow) 
    n_flow_params = instantiated_flow != [] ? 
        sum((x -> x.n_params).(instantiated_flow)) : 0

    sampler = is_diagonal ? 
        diagonal_gaussian_sampler : 
        full_gaussian_sampler
    posterior = is_diagonal ? 
        log_diagonal_gaussian_posterior : 
        log_full_gaussian_posterior

    (n_variational_params, init_vals) = init_param_fn(n_params, n_flow_params)
    return VariationalModel(
        Models.simple_apply_grad,
        ModelStructure(layers, n_inputs, n_params),
        init_vals,
        sampler,
        prior_creator(n_params),
        log_likelihood,
        posterior,

        instantiated_flow,
        n_variational_params,
        n_flow_params
    )
end

# simpler constructor for model architectures with full Gaussian weights.
function VariationalFullGaussianModel(log_likelihood::Function, n_inputs::Int,
        layers::Vector; hyper_weight = 0.1,
        normalising_flow::AbstractArray = [])

    function init_param_fn_full(n, n_f)
        init_L = -20ones(HelperFunctions.triangular(n))
        t = 0
        for i in 1:n
            t += i
            init_L[t] = -5
        end

        layer_means_init = init_means(layers, Models.create_simple_evaluation(layers), [n_inputs])
        normalising_init = randn(n_f)

        n + HelperFunctions.triangular(n), [layer_means_init;init_L;normalising_init]
    end

    VariationalGaussianModel(
        n_params -> diagonal_gaussian_prior_creator(n_params, hyper_weight = hyper_weight),
        log_likelihood, 
        n_inputs, 
        false, 
        layers, 
        normalising_flow = normalising_flow,
        init_param_fn = init_param_fn_full
    )
end

# simpler constructor for model architectures with diagonal Gaussian weights.
function VariationalDiagonalGaussianModel(log_likelihood::Function, 
        n_inputs::Int, layers::Vector; hyper_weight = 0.1,
        normalising_flow::AbstractArray = [])

    function init_param_fn_diag(n, n_f)
        init_log_σ = ones(n) * -5
        layer_means_init = init_means(layers, Models.create_simple_evaluation(layers), [n_inputs])
        normalising_init = randn(n_f)

        2 * n, [layer_means_init;init_log_σ;normalising_init]
    end

    VariationalGaussianModel(
        n_params -> diagonal_gaussian_prior_creator(n_params, hyper_weight = hyper_weight),
        log_likelihood, 
        n_inputs, 
        true, 
        layers, 
        normalising_flow = normalising_flow,
        init_param_fn = init_param_fn_diag
    )
end

# evaluate log density of samples on diagonal gaussian
function log_diagonal_gaussian_posterior(m::Models.Model, 
        samples::AbstractArray{Float64})
    variational_params = m.θ[1:m.n_variational_params]
    n_params = m.structure.n_total_params

    Σ_diag = exp.(variational_params[n_params + 1:end]) .^ 2

    μ = variational_params[1:n_params]
    log_posterior_pdf = x -> -log((2π) ^ (length(x/2))) - 0.5(((x - μ) .* Σ_diag))' * (x - μ)

    return log_posterior_pdf.(eachcol(samples))
end

# evaluate log density of samples on full gaussian
function log_full_gaussian_posterior(m::Models.Model, 
        samples::AbstractArray{Float64})
    variational_params = m.θ[1:m.n_variational_params]
    n_params = m.structure.n_total_params
    L = HelperFunctions.to_lower_triangular(
        exp.(variational_params[n_params + 1:end]), n_params
    )

    log_posterior_pdf = sample -> logpdf(MvNormal(
        variational_params[1:n_params], 
        L * L'
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

# closed form solution to integral of variational posterior for Gaussians as
# used by BBVI.
function gaussian_entropy(log_σ::Vector{Float64}, dims::Int)
	return 0.5 * dims * (1 + log(2π)) + sum(log_σ)
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
        variational_params = m.θ[1:m.n_variational_params]
        return -gaussian_entropy(
            variational_params[m.structure.n_total_params + 1:end], 
            m.structure.n_total_params
        )
    end
    return mean(m.log_posterior(m, w₀))
end

# VI training function to optimise
function variational_free_energy_creator(is_closed_form_gaussian::Bool, 
        coef_func::Function; λ::Float64 = 1.0, λγ::Float64 = 100.0,
        α₀::Float64 = 12.0, β₀::Float64 = 0.1, is_adaptive_regression::Bool = false,
        custom_log_density::Tuple{Bool, Function} = (false, x->x))

    function f(ms::Vector, X::AbstractMatrix{Float64}, 
            y::AbstractArray, args::Training.TrainingParameters, 
            i::Int, M::Int)
        m = ms[1]
        coef_reweighting = λ * coef_func(M, i)
        variational_params = m.θ[1:m.n_variational_params]

        retVal = 0

        # adaptive regression case
        if is_adaptive_regression
            ατ = exp(m.θ[end - 1])
            βτ = exp(m.θ[end])

            # likelihood
            function sample_evaluation(m, sample, X, y)
                ŷ = Models.pred(m.structure, sample, X)'
                digamma(ατ) - log(βτ) - ((ατ) / (βτ)) * 
                    sum((y - ŷ) .^ 2) - log(2π)
            end
            m.log_likelihood = sample_evaluation

            α₁, β₁ = ατ, βτ
            α₂, β₂ = α₀, β₀

            retVal += (α₁ * log(β₁/β₂) - (loggamma(α₁) - loggamma(α₂)) +
                (α₁ - α₂) * digamma(α₁) - (β₁ - β₂) * (α₁/β₁)) * λγ
        end

        w₀ = m.weight_sampler(
            variational_params, 
            args.n_samples, 
            m.structure.n_total_params
        )

        # normalising flow then E_Q[Q(w)]
        (samples, flows_E) = evaluate_normalising_flow(m, w₀)
        variational_expectation = evaluate_variational_posterior_expectation(
            m, is_closed_form_gaussian, w₀)

        # evaluate log joint density log(p(D,w))
        if (custom_log_density[1])
            log_joint_density = custom_log_density[2]
        else
            log_joint_density = log_density
        end

        log_joint_distribution = log_joint_density(m, samples, X, y, 
            coef = coef_reweighting)

        # # for debugging
        # println("$(τ), $(α^τ / β^τ), $(α^τ), $(β^τ), $(α₀), $(β₀) $(retVal)")
        # println(coef_reweighting * (variational_expectation - flows_E))
        # println(-log_joint_distribution)
        # println(retVal)
        # println()

        return coef_reweighting * (variational_expectation - flows_E) - 
            log_joint_distribution + retVal
    end
end

# simplified batch Bayes by Backprop objective without normalising flows
function KL_divergence_objective(f::Models.DegenerateModel, m::Models.Model, 
        i::Int, M::Int, coef_func::Function)
    function g(X::AbstractMatrix{Float64}, w::AbstractArray, y::AbstractArray)
        coef = coef_func(M, i)
        coef * mean(m.log_posterior(m, w)) - log_density(m, w, X, y, coef = coef)
    end
end

function Langevin_Stein_objective(f::Models.DegenerateModel, m::Models.Model, 
        i::Int, M::Int, coef_func::Function)
    function g(X::AbstractMatrix{Float64}, w::AbstractArray, y::AbstractArray)
        ∇w = gradient(Params([w])) do
            log_density(m, w, X, y, coef = coef_func(M, i))
        end[w]

        z = reshape(w, length(w), 1)
        ∇f = jacobian(r -> pred(f.structure, f.θ, r), z)[1] |> tr

        (∇w' * pred(f.structure, f.θ, reshape(w, length(w), 1)))[1,1] + ∇f
    end
end

# VI Operator Objective
function create_variational_operator(operator::Function; 
        t::Function=(x -> x^2), coef_func::Function=uniform_complexity_cost)
    function variational_operator_objective(ms::Vector,
            X::AbstractMatrix{Float64}, y::AbstractArray, 
            args::Training.TrainingParameters, i::Int, M::Int)

        m = ms[1]
        f = ms[2]
        
        W = m.weight_sampler(
            m.θ, 
            args.n_samples, 
            m.structure.n_total_params
        )

        operator_objective = operator(f, m, i, M, coef_func)
        res = (w -> operator_objective(X, w, y)).(eachcol(W))
        mean(res) |> t
    end
end

function Langevin_Stein_apply_gradient(ms::Vector, g)
    m = ms[1]
    f = ms[2]
    mg = g[1:m.n_variational_params + m.n_flow_params]
    fg = g[m.n_variational_params + m.n_flow_params + 1:end]

    m.θ = m.θ .- mg
    f.θ = f.θ .+ fg
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

    preds = map(w -> pred(m.structure, w, reshape(x, (:,1))[:]), eachcol(W))
    ŷ = coef1 * sum(preds)

    f = p -> (p - ŷ) * (p - ŷ)'
    Σ = coef2 * sum(f.(preds))

    return tr(Σ)
end

function update_online_diagonal_gaussian_posterior!(
        m::Models.VariationalModel, n_active_samples::Int, 
        x::AbstractArray{Float64}, y::AbstractArray{Float64})
    n_params = m.structure.n_total_params
    function log_expectation_of_exponential(θ)
        W = m.weight_sampler(θ, n_active_samples, n_params)
        log(mean(
            (w -> exp(m.log_likelihood(m, w, reshape(x, length(x), 1), y))).(eachcol(W))
        ))
    end

    Σ_diag = exp.(m.θ[n_params + 1:2n_params]) .^ 2

    g = gradient(
        θ -> log_expectation_of_exponential(vcat(θ, m.θ[n_params + 1:2n_params])),
        m.θ[1:n_params]
    )[1]

    H = hessian(
        θ -> log_expectation_of_exponential(vcat(θ, m.θ[n_params + 1:2n_params])),
        m.θ[1:n_params]
    )

    m.θ[1:n_params] += 0.1 * Σ_diag .* g
    m.θ[n_params + 1:2n_params] = (
        Σ_diag .+ 
        Σ_diag .^ 2 .* diag(H)
    ) .^ 0.5 .|> log
end

function find_uncertainties(m::Models.VariationalModel, 
        U::AbstractMatrix{Float64}, n_active_samples::Int)
    W = sample_model(m, n_active_samples)

    uncertainties = map(x -> AL_datapoint_uncertainty(m, W, x), eachcol(U))
    return uncertainties
end

function active_learning!(m::Models.VariationalModel, 
        X::AbstractMatrix{Float64}, Y::AbstractMatrix{Float64}, 
        U::AbstractMatrix{Float64}, oracle::Function, 
        args::Training.TrainArgs, iterations::Int; n_active_samples::Int = 20, 
        threshold::Float64 = -Inf, retrains = true)
    max_uncertainty = Inf
    init_x_length = size(X)[2]

    i = 1
    while true
        uncertainties = find_uncertainties(m, U, n_active_samples)
        max_uncertainty_i = argmax(uncertainties)
        max_uncertainty = uncertainties[max_uncertainty_i]
        Ux_max = U[:,max_uncertainty_i]
        Uy_max = oracle(Ux_max)[1]

        U = U[1:end, 1:end .!= max_uncertainty_i]
        X = [X Ux_max]
        Y = [Y Uy_max]
        
        i = i + 1

        if retrains
            Training.train!([m], X, Y, args)
        else
            update_online_diagonal_gaussian_posterior!(m, n_active_samples,
                Ux_max, [Uy_max])
        end

        if !((U != []) && (max_uncertainty > threshold) && i <= iterations)
            return (X, Y)
        end
    end
end

function predict_VI(m::Models.VariationalModel, X::AbstractMatrix{Float64}; 
        n_samples::Int = 1)
    W = m.weight_sampler(m.θ, n_samples, m.structure.n_total_params)
    f = transform_sample(m.normalising_flow, m.θ[m.n_variational_params + 1:end])
    mean((w -> pred(m.structure, f(w), X)).(eachcol(W)))
end

end