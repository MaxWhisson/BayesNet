# dependencies:
using LinearAlgebra
using Distributions
using Optimisers
using Random

include("Samplers.jl")
include("Models.jl")
include("../Training.jl")

# constructor for Gaussian Variational models
function VariationalGaussianModel(prior_creator::Function, log_likelihood::Function, 
        n_inputs::Int, is_diagonal::Bool, layers::Vector{Layer}; normalising_flow = [])

    n_weights = n_inputs * layers[1].n +
        sum([layers[i].n * layers[i + 1].n for i in 1:length(layers) - 1])
    n_params = n_weights + length(layers)

    n_variational_params = is_diagonal ? n_params * 2 : n_params + (n_params ^ 2)
    n_flow_params = normalising_flow != [] ? sum((x -> x.n_params).(normalising_flow)) : 0

    sampler = is_diagonal ? diagonal_gaussian_sampler : full_gaussian_sampler
    posterior = is_diagonal ? log_diagonal_gaussian_posterior : log_full_gaussian_posterior

    return VariationalModel(
        ModelStructure(layers, n_inputs, n_params),
        abs.(randn(n_variational_params + n_flow_params)), # TODO replace this with function call
        sampler,
        prior_creator(n_params),
        log_likelihood,
        posterior,

        normalising_flow,
        n_variational_params,
        n_flow_params
    )
end

# simpler constructor for model architectures with full Gaussian weights.
function VariationalFullGaussianModel(log_likelihood::Function, n_inputs::Int, 
    layers::Vector{Layer}; normalising_flow = [])
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
    layers::Vector{Layer}; normalising_flow = [])
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
        Diagonal(exp.(variational_params[n_params + 1:end]) .^ 2)
    ), sample)
    return log_posterior_pdf.(eachcol(samples))
end

# evaluate log density of samples on full gaussian
function log_full_gaussian_posterior(m, samples)
    variational_params = m.θ[1:m.n_variational_params]
    n_params = m.structure.n_total_params
    L = reshape(exp.(variational_params[n_params + 1:end]), (n_params, n_params))

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

function propagate_matrix_opp(matrix1, matrix2, i, f)                                                                                                                                                                                                                                                 
    matrix1[:,i + 1] = f(matrix2[:, i])                                                                                                                                                                                                                     
    return matrix1
end

function set_col_matrix_expr(matrix, i, col)
    matrix[:,i] = col
    return matrix
end

function flow_transforms(m::VariationalModel, flow_params::Vector{Float64}, w::Vector{Float64})
    transforms = foldl(
        ((i, param_index, transformation_matrix, jacobian_determinants), layer) -> (
            i + 1, 
            param_index + layer.n_params + 1,
            propagate_matrix_opp(
                transformation_matrix,
                transformation_matrix,
                i, 
                layer.func(
                    flow_params[param_index:param_index + layer.n_params]
                )
            ),
            propagate_matrix_opp(
                jacobian_determinants,
                transformation_matrix,
                i, 
                layer.jacobian_determinant(
                    flow_params[param_index:param_index + layer.n_params]
                )
            )
        ), 
        m.normalising_flow; 
        init = (1, 1, [w zeros(length(w), length(w))], zeros(length(w), length(w) + 1))
    )
    return transforms[3][:,end], sum(transforms[4], dims = 2)
end

function uniform_complexity_cost(M, i)
    return 1/M
end

function exponential_complexity_cost(M, i)
    return (2 ^ (M - i)) / (2 ^ M - 1)
end

# VI training function to optimise
function variational_free_energy_creator(is_closed_form_gaussian::Bool, coef_func::Function)
    function f(m, X, y, args, i, M)
        coef = coef_func(M, i)

        # samples from approximate posterior
        variational_params = m.θ[1:m.n_variational_params]
        flow_params = m.θ[m.n_variational_params + 1:end]

        w₀ = m.weight_sampler(variational_params, args.n_samples, m.structure.n_total_params)

        if (m.normalising_flow != [])
            transform_vals = ((m, params) -> w -> 
                flow_transforms(m, params, w))(m, flow_params).(eachcol(w₀))
                samples = foldl(
                    ((i, matrix), col) -> (i + 1, set_col_matrix_expr(matrix, i, col)),
                    (x->x[1]).(transform_vals), 
                    (1, Matrix(undef, size(w₀)))
                )
            flows_E = mean(ln.((x->x[2])(transform_vals)))
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
            variational_expectation = mean(m.log_posterior(m, samples))
        end

        return coef * (variational_expectation - flows_E) - 
            mean(log_density(m, samples, X, y, coef = coef))
    end
end