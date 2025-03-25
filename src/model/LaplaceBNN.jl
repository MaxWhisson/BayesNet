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

# constructor for Laplace models
function BuildLaplaceModel(priorCreator::Function, log_likelihood::Function, 
    n_inputs::Int, layers::Vector)

    n_weights = n_inputs * layers[1].n +
        sum([layers[i].n * layers[i + 1].n for i in 1:length(layers) - 1])
    n_params = n_weights + sum(layers .|> (x -> x.n))

    return Models.LaplaceModel( 
        Models.ModelStructure(layers, n_inputs, n_params),
        randn(n_params + HelperFunctions.triangular(n_params)),        # weights
        priorCreator(n_params),
        log_likelihood
    )
end

function create_MAP_loss_fn(
        ;log_density_fn::Tuple{Bool, Function} = (false,log_density),
        n_samples::Int = 10)
    function MAP_loss_fn(m::Models.LaplaceModel, X::AbstractMatrix{Float64},
            y::AbstractArray, args::Training.TrainingParameters, i::Int, 
            M::Int)
        
        if !log_density_fn[1]
            return -log_density_fn[2](m, m.θ[1:m.structure.n_total_params], X, y, coef = 1/M)
        end
        
        return -log_density_fn[2](
            rand(get_approximating_distribution(m), n_samples)
        )
    end
end

# find Hessian
function create_fit_covariance(
        ;log_density_fn::Tuple{Bool, Function} = (false,log_density), 
        n_samples::Int = 10)
    function fit_covariance!(m::Models.LaplaceModel, 
            X::AbstractMatrix{Float64}, y::AbstractArray)
        H = hessian(
            θ -> !log_density_fn[1] ?
                    -log_density_fn[2](m, θ, X, y) : 
                    -log_density_fn[2](
                        rand(get_approximating_distribution(m), n_samples)
                    ),
            m.θ[1:m.structure.n_total_params]
        )

        H_inv = inv(H)
        # println(minimum(diag(inv(H))))

        lower_H = LowerTriangular(H_inv)
        H_inv = zeros(size(H_inv)) + lower_H + lower_H' - diagm(diag(H_inv))
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
    Training.train!(m, X, y, args)
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