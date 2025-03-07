using Distributions
using LogExpFunctions
using LinearAlgebra

using BayesNet

leaky_relu = x -> max(x,0.1)

function generate_binary_clusters(c1::Vector{Float64}, c2::Vector{Float64})
    x1,x2 = MvNormal(c1, I), MvNormal(c2, I)
    X = [rand(x1, 100) rand(x2, 100)]
    y = [zeros(100); ones(100)]
    return X, y .|> Bool
end

function generate_regression_data()
    x = rand(Uniform(-5, 5), (1, 100))
    f = x -> 2x + 5randn()
    return x, f.(x)
end

function generate_multi_clusters(centres...)
    n = 100
    xs = [MvNormal(c, I) for c in centres]
    X = foldl(hcat, [rand(x, n) for x in xs])
    y = [div(i - 1, n) + 1 for i in 1:size(X)[2]]
    return X, y .|> Int
end

function make_binary_Laplace_model(n_in, layers)
    BayesNet.BuildLaplaceModel(
        BayesNet.diagonal_gaussian_prior_creator,
        BayesNet.binary_log_likelihood,
        n_in,
        layers
    )
end

function make_test_VI_model(D::Int, is_diagonal::Bool, n_out::Int; log_l = BayesNet.binary_log_likelihood)
    f = is_diagonal ? 
        BayesNet.VariationalDiagonalGaussianModel : 
        BayesNet.VariationalFullGaussianModel
    return f(
        log_l, 
        D,
        [
            BayesNet.Layer(10, leaky_relu), 
            BayesNet.Layer(5, leaky_relu), 
            BayesNet.Layer(n_out, logistic)
        ]
    )
end

function make_test_VI_regression_model(D::Int, is_diagonal::Bool, n_out::Int)
    f = is_diagonal ? 
        BayesNet.VariationalDiagonalGaussianModel : 
        BayesNet.VariationalFullGaussianModel
    return f(
        BayesNet.regression_log_likelihood, 
        D,
        [BayesNet.Layer(5, leaky_relu), BayesNet.Layer(n_out, x -> x)]
    )
end

function make_test_binary_MCMC(D)
    BayesNet.Build_MCMC_Model(
        BayesNet.diagonal_gaussian_prior_creator,
        BayesNet.binary_log_likelihood,
        D,
        [
            BayesNet.Layer(5, leaky_relu), 
            BayesNet.Layer(1, logistic), 
        ]
    )
end