module LaplaceBNN

# function exports
export  BuildLaplaceModel,
        create_MAP_loss_fn,
        create_fit_covariance,
        fit_gaussian!,
        get_approximating_distribution,
        predict_Laplace

# dependencies:
using LinearAlgebra
using Distributions
using Optimisers
using Random
using Zygote

using ..Models
using ..Training
using ..HelperFunctions

using ..ModelFunctions: count_params

# constructor for Laplace models
function BuildLaplaceModel(priorCreator::Function, log_likelihood::Function, 
    n_inputs::Int, layers::Vector)

    evalOrder = create_simple_evaluation(layers)
    n_params = count_params(layers, evalOrder, [n_inputs])

    return Models.LaplaceModel( 
        Models.simple_apply_grad,
        Models.ModelStructure(layers, n_inputs, n_params),
        randn(n_params + HelperFunctions.triangular(n_params)),        # weights
        priorCreator(n_params),
        log_likelihood
    )
end

function create_MAP_loss_fn(
        ;log_density_fn::Tuple{Bool, Function} = (false,log_density),
        n_samples::Int = 10)
    function MAP_loss_fn(ms::Vector, X::AbstractMatrix{Float64},
            y::AbstractArray, args::Training.TrainingParameters, i::Int, 
            M::Int)
        m = ms[1]
        return -log_density_fn[2](m, m.θ[1:m.structure.n_total_params], X, y, coef = 1/M)
    end
end

# find Hessian
function create_fit_covariance(
        ;log_density_fn::Tuple{Bool, Function} = (false,log_density))
    function fit_covariance!(m::Models.LaplaceModel, 
            X::AbstractMatrix{Float64}, y::AbstractArray)
        H_inv = hessian(
            θ -> -log_density_fn[2](m, θ, X, y),
            m.θ[1:m.structure.n_total_params]
        ) |> inv

        lower_H = LowerTriangular(H_inv)
        H_inv = zeros(size(H_inv)) + lower_H + lower_H' - diagm(diag(H_inv))

        @assert issymmetric(H_inv) "Inverse Hessian isn't symmetric..."
        @assert isposdef(H_inv) "Inverse Hessian isn't positive definite..."

        L = HelperFunctions.flatten_triangular(
            cholesky(H_inv).L, 
            m.structure.n_total_params
        )
        m.θ[m.structure.n_total_params + 1:end] = reshape(L, (1,:))
    end
end

function fit_gaussian!(m::Models.LaplaceModel, X::AbstractMatrix{Float64},
        y::AbstractArray, train_params::TrainingParameters;
        custom_log_density::Tuple{Bool, Function} = (false, log_density))
    args = TrainArgs(
        create_MAP_loss_fn(log_density_fn = custom_log_density), 
        train_params
    )
    # learn first portion of m.θ (MAP)
    Training.train!([m], X, y, args)
    create_fit_covariance(log_density_fn = custom_log_density)(m, X, y)
end

function get_approximating_distribution(m::Models.LaplaceModel)
    L = HelperFunctions.to_lower_triangular(
        m.θ[m.structure.n_total_params + 1:end],
        m.structure.n_total_params
    )

    MvNormal(
        m.θ[1:m.structure.n_total_params], 
        L * L'
    )
end

function predict_Laplace(m::Models.LaplaceModel, X::AbstractMatrix{Float64}
        ;n_samples::Int = 1)
    approximating_distribution = get_approximating_distribution(m)
    W = rand(approximating_distribution, n_samples)
    mean((w -> pred(m.structure, w, X)).(eachcol(W)))
end

end