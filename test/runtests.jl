using Test

@testset "All Tests" begin
    @testset "VI tests" begin
        include("model/VI_BNN_tests.jl")
    end

    @testset "Laplace tests" begin

    end

    @testset "MCMC tests" begin

    end

    @testset "sampler tests" begin
        include("model/Samplers_tests.jl")
    end

    @testset "training tests" begin

    end

    @testset "model tests" begin
        include("model/Models_tests.jl")
    end
end
