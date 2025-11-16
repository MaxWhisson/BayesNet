using BayesNet

r = x -> max(0.1x, x)

function test_forwardSingleInput()
    l = BayesNet.Dense(3, r)
    layerParams = [
        [1.0 2.0 1.0;1.0 1.0 2.0],
        [1.0, 2.0]
    ]
    inputs = [
        [2.3 1.3;1.3 4.5;1.4 4.6]
    ]
    state = zeros(0,0)
    (output, newState) = BayesNet.forward(l, layerParams, inputs, state)

    expectedOutput = [
        (1 * 2.3 + 2 * 1.3 + 1 * 1.4 + 1) (1 * 1.3 + 2 * 4.5 + 1 * 4.6 + 1);
        (1 * 2.3 + 1 * 1.3 + 2 * 1.4 + 2) (1 * 1.3 + 1 * 4.5 + 2 * 4.6 + 2)
    ]
    
    condition1 = newState == zeros(0, 2)
    condition2 = output ≈ expectedOutput

    condition1 && condition2
end

@testset "DenseLayer Tests" begin
    @test test_forwardSingleInput()
end