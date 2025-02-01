using Test

@testset "All Tests" begin
    @testset "VI tests" begin
        include("model/VI_BNN_tests.jl")
    end

    @testset "Model tests" begin
        include("model/Models_tests.jl")
    end
end
