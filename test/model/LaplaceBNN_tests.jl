using BayesNet

function test_Laplace_model_constructor()
    m = make_binary_Laplace_model(
        2,
        [BayesNet.Layer(5, leaky_relu), BayesNet.Layer(1, logistic)]
    )
    condition1 = typeof(m) == BayesNet.LaplaceModel
    condition2 = m.structure.n_params = 21
    condition1 && condition2
end

function test_MAP_loss_fn()
    # TODO
end

@testset "Laplace" begin
    @test test_Laplace_model_constructor()
end