using BayesNet

function test_Laplace_model_constructor()
    m = make_binary_Laplace_model(
        2,
        [BayesNet.DenseLayer(5, leaky_relu), BayesNet.DenseLayer(1, logistic)]
    )
    condition1 = typeof(m) == BayesNet.LaplaceModel
    condition2 = m.structure.n_total_params == 21
    condition1 && condition2
end

function test_MAP_loss_fn()
    m = make_binary_Laplace_model(
        1,
        [BayesNet.DenseLayer(5, leaky_relu), BayesNet.DenseLayer(1, logistic)]
    )
    X = [1.0 1.1]
    y = [1.0;0.0]
    typeof(BayesNet.create_MAP_loss_fn()([m], X, y, BayesNet.TrainingParameters(), 0, 0)) == Float64
end

function test_fit_covariance()
    m = make_binary_Laplace_model(
        1,
        [BayesNet.DenseLayer(5, leaky_relu), BayesNet.DenseLayer(1, logistic)]
    )
    before = m.θ[:]
    X = [1.0 1.1]
    y = [1.0;0.0]
    BayesNet.create_fit_covariance()(m, X, y)
    before != m.θ
end

function test_fit_gaussian()
    m = make_binary_Laplace_model(
        1,
        [BayesNet.DenseLayer(5, leaky_relu), BayesNet.DenseLayer(1, logistic)]
    )
    before = m.θ[:]
    X = [1.0 1.1]
    y = [1.0;0.0]
    BayesNet.fit_gaussian!(m, X, y, BayesNet.TrainingParameters(batch_size=2))
    before != m.θ
end

@testset "Laplace" begin
    @test test_Laplace_model_constructor()
    @test test_MAP_loss_fn()
    @test test_fit_covariance()
    @test test_fit_gaussian()
end