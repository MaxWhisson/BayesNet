using LogExpFunctions
using Distributions
using BayesNet

include("../test_helper_functions/sample_models.jl")

##########################################################################
####                              Tests                               ####
##########################################################################

# D is the dimension of the parameter vector here
function test_PlanarFlowLayer_func_size_1(D)
    z = ones(D)
    flow_layer = BayesNet.PlanarFlowLayer(D)
    size(flow_layer.func(randn(flow_layer.n_params))(z)) == size(z)
end

function test_PlanarFlowLayer_jacobian_determinant_size_1(D)
    z = ones(D)
    flow_layer = BayesNet.PlanarFlowLayer(D)
    flow_layer.jacobian_determinant(randn(flow_layer.n_params))(z)
    true # just checking that the function doesn't cause a matrix dimension error
end

function test_RadialFlowLayer_func_size_1(D)
    z = ones(D)
    flow_layer = BayesNet.RadialFlowLayer(D)
    size(flow_layer.func(randn(flow_layer.n_params))(z)) == size(z)
end

function test_RadialFlowLayer_jacobian_determinant_size_1(D)
    z = ones(D)
    flow_layer = BayesNet.PlanarFlowLayer(D)
    flow_layer.jacobian_determinant(randn(flow_layer.n_params))(z)
    true # just checking that the function doesn't cause a matrix dimension error
end

function test_flow_chain_1(D)
    z = ones(D)
    flow_layers = [
        BayesNet.BayesNet.PlanarFlowLayer(D), 
        BayesNet.RadialFlowLayer(D), 
        BayesNet.RadialFlowLayer(D)
    ]
    size(foldl((acc, l) -> l.func(randn(l.n_params))(acc), flow_layers, init = z)) == size(z)
end

function test_diagonal_gaussian_prior(n_params)
    f = BayesNet.diagonal_gaussian_prior_creator(n_params)
    typeof(f.func(f.θ)(randn(n_params))) == Float64
end

function test_softmax()
    BayesNet.softmax([92.1;12.3;0.2;1.2]) |> sum ≈ 1
end

function test_binary_log_likelihood()
    m = make_test_VI_model(2, true, 1)
    X, y = generate_binary_clusters(float.([2,1]), float.([1,2]))
    w = randn(m.structure.n_total_params)
    res = exp(BayesNet.binary_log_likelihood(m, w, X, y))
    (res <= 1) && (res >= 0)
end

function test_multi_class_log_likelihood()
    m = make_test_VI_model(2, true, 3)
    X, y = generate_multi_clusters([1,2], [1,2], [3,3])
    w = randn(m.structure.n_total_params)
    res = exp(BayesNet.multi_class_log_likelihood(m, w, X, y))
    (res <= 1) && (res >= 0)
end

function test_regression_log_likelihood()
    m = make_test_VI_regression_model(1, true, 1)
    X, y = generate_regression_data()
    w = randn(m.structure.n_total_params)
    res = BayesNet.regression_log_likelihood(m, w, X, y) # probably 0
    true
end

function test_log_density()
    m = make_test_VI_model(2, true, 3, log_l = BayesNet.multi_class_log_likelihood)
    X, y = generate_multi_clusters([1,2], [1,2], [3,3])
    BayesNet.log_density(m, randn(m.structure.n_total_params, 5), X, y, coef = 1)
    true
end

function test_pred_1()
    m = make_test_VI_model(2, true, 3, log_l = BayesNet.multi_class_log_likelihood)
    X, y = generate_multi_clusters([1,2], [1,2], [3,3])
    res = BayesNet.pred(m.structure, randn(m.structure.n_total_params), X)
    size(res)[2] == length(y)
end

function test_pred_2()
    m = make_test_VI_regression_model(1, true, 1)
    X, y = generate_regression_data()
    res = BayesNet.pred(m.structure, randn(m.structure.n_total_params), X)
    size(res) == size(y)
end

##########################################################################
####                            Test sets                             ####
##########################################################################

@testset "Models Tests" begin
    @testset "Normalising flows" begin
        @test test_PlanarFlowLayer_func_size_1(2)
        @test test_PlanarFlowLayer_func_size_1(5)
        @test test_RadialFlowLayer_func_size_1(2)
        @test test_RadialFlowLayer_func_size_1(8)
        @test test_PlanarFlowLayer_jacobian_determinant_size_1(2)
        @test test_RadialFlowLayer_jacobian_determinant_size_1(2)
        @test test_flow_chain_1(2)
        @test test_flow_chain_1(5)
    end

    @testset "log priors" begin
        @test test_diagonal_gaussian_prior(3)
        @test test_diagonal_gaussian_prior(20)
    end

    @testset "misc" begin
        @test test_softmax()
        @test test_log_density()
        @test test_pred_1()
        @test test_pred_2()
    end

    @testset "log likelihoods" begin
        @test test_binary_log_likelihood()
        @test test_multi_class_log_likelihood()
        @test test_regression_log_likelihood()
    end

    @testset "post training processing" begin

    end
end