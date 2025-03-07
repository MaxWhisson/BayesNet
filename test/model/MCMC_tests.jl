using BayesNet

include("../test_helper_functions/sample_models.jl")

function test_adaptive_MCMC()
    m = make_test_binary_MCMC(2)
    X, y = generate_binary_clusters(float.([2,1]), float.([1,2]))
    weight_samples = BayesNet.adaptive_MCMC_sampler(m, X, y, 100)
    (size(weight_samples)[1] == 2 * 5 + 5 + 5 + 1) && (size(weight_samples)[2] == 100)
end

function test_Langevin_dynamics_MCMC_sampler()
    m = make_test_binary_MCMC(2)
    X, y = generate_binary_clusters(float.([2,1]), float.([1,2]))
    weight_samples = BayesNet.langevin_dynamics_MCMC_sampler(
        m, X, y, BayesNet.TrainingParameters()
    )
    length(size(weight_samples)) == 2
end

@testset "MCMC tests" begin
    @test test_adaptive_MCMC()
    @test test_Langevin_dynamics_MCMC_sampler()
end