using BayesNet

function test_diagonal_gaussian_sampler_1()
    means = [1,2,3,4,5] .|> float
    res = BayesNet.diagonal_gaussian_sampler([means;-500 * ones(5)], 2, 5)
    (res[:,1] ≈ means) && (res[:,2] ≈ means)
end

function test_full_gaussian_sampler_1()
    means = [1,2,3,4,5] .|> float
    res = BayesNet.full_gaussian_sampler([means;-500 * ones(BayesNet.triangular(5))], 2, 5)
    (res[:,1] ≈ means) && (res[:,2] ≈ means)
end

@testset "Samplers tests" begin
    @testset "Gaussian samplers" begin
        @test test_diagonal_gaussian_sampler_1()
        @test test_full_gaussian_sampler_1()
    end
end