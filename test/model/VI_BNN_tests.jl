using Distributions
using LogExpFunctions

using BayesNet
include("../test_helper_functions/sample_models.jl")

##########################################################################
####                              Tests                               ####
##########################################################################

function test_set_col_matrix_expr_1(i)
    function f()
        m = zeros(3,5)
        col = [1;1;1]
        m_out = BayesNet.set_col_matrix_expr(m, i, col)
        for j in 1:5
            if ((i == j) && !(m_out[:,j] == col)) return false end
            if ((i != j) && !(m_out[:,j] == [0;0;0])) return false end
        end
        return true
    end
end

function test_propagate_matrix_opp_basic()
    m1 = zeros(2,3)
    m2 = ones(2,3)
    m_out = BayesNet.propagate_matrix_opp(m1, m2, 1, (x -> x.+2))
    return m_out[:,2] == [3;3]
end

function test_propagate_matrix_opp_chain()
    m1 = zeros(2,3)
    m_out_1 = BayesNet.propagate_matrix_opp(m1, m1, 1, (x -> x.+2))
    m_out_2 = BayesNet.propagate_matrix_opp(m_out_1, m_out_1, 2, (x -> x.+2))
    return m_out_2[:,3] == [4;4]
end

function test_uniform_coef_1()
    return BayesNet.uniform_complexity_cost(1,1) ≈ 1
end

function test_uniform_coef_2()
    return BayesNet.uniform_complexity_cost(5,2) ≈ 0.2
end

function test_exponential_complexity_cost_1()
    return BayesNet.exponential_complexity_cost(1,1) ≈ 1
end

function test_exponential_complexity_cost_2()
    return BayesNet.exponential_complexity_cost(5,2) ≈ (8/31)
end

function test_gaussian_entropy_1()
    return BayesNet.gaussian_entropy(float.(log.([1,1])), 2) ≈ 1 + log(2π)
end

function test_gaussian_entropy_2()
    return BayesNet.gaussian_entropy(float.(log.([ℯ,ℯ])), 4) ≈ 2 * (1 + log(2π)) + 2
end

function test_log_diagonal_gaussian_posterior_size_correct(D)
    m = make_test_VI_model(D, true, 1)
    variational_params = m.θ[1:m.n_variational_params]
    samples = m.weight_sampler(variational_params, 20, m.structure.n_total_params)
    return size(samples) == (m.structure.n_total_params, 20)
end

function test_log_full_gaussian_posterior_size_correct(D)
    m = make_test_VI_model(D, false, 1)
    variational_params = m.θ[1:m.n_variational_params]
    samples = m.weight_sampler(variational_params, 20, m.structure.n_total_params)
    return size(samples) == (m.structure.n_total_params, 20)
end

function test_flow_transforms_1(n_inputs)
    m = make_test_VI_model(n_inputs, true, 1)
    D = m.structure.n_total_params # number of weights
    flow = [
        BayesNet.PlanarFlowLayer(D), 
        BayesNet.RadialFlowLayer(D), 
        BayesNet.RadialFlowLayer(D)
    ]
    sample = randn(D)
    params = randn((D + 2) + (D + 2) + (2D + 1))
    transform = BayesNet.transform_sample(flow, params)
    log_jac_det = BayesNet.sum_log_jacobian(flow, params)

    wₖ = transform(sample)
    jacobian_det_sum = log_jac_det(sample)

    (length(wₖ) == D) && (typeof(jacobian_det_sum) == Float64)
end

function test_flow_transforms_2(n_inputs)
    m = make_test_VI_model(n_inputs, true, 1)
    D = m.structure.n_total_params # number of weights
    flow = [
        BayesNet.RadialFlowLayer(D)
    ]

    sample = randn(D)
    params = randn((D + 2) + (D + 2) + (2D + 1))
    transform = BayesNet.transform_sample(flow, params)
    log_jac_det = BayesNet.sum_log_jacobian(flow, params)

    wₖ = transform(sample)
    jacobian_det_sum = log_jac_det(sample)

    (length(wₖ) == D) && (typeof(jacobian_det_sum) == Float64)
end

function test_flow_transforms_3()
    n_inputs = 1
    flow = [
        BayesNet.RadialFlowLayer(1)
    ]

    sample = [0.0]
    params = [100.0, 0.541324854612918, 1.0]

    res1 = flow[1].jacobian_determinant(params)(sample) |> abs |> log
    log_jac_det = BayesNet.sum_log_jacobian(flow, params)
    jacobian_det_sum = log_jac_det(sample)

    res1 ≈ jacobian_det_sum
end

function test_variational_free_energy()
    f = BayesNet.variational_free_energy_creator(false, BayesNet.exponential_complexity_cost)
    m = make_test_VI_model(2, true, 1)
    X, y = generate_binary_clusters(float.([2,1]), float.([1,2]))

    L = f([m], X, y, BayesNet.TrainingParameters(), 1, 1)
    typeof(L) == Float64
end

function test_equivalent_posterior_densities(D)
    m1 = BayesNet.VariationalDiagonalGaussianModel(
        x -> x,
        D,
        [
            # BayesNet.Dense(20, x -> x),
            BayesNet.Dense(1, x -> x)
        ]
    )

    m2 = BayesNet.VariationalFullGaussianModel(
        x -> x,
        D,
        [
            # BayesNet.Dense(20, x -> x),
            BayesNet.Dense(1, x -> x)
        ]
    )

    m1.θ[1:m1.structure.n_total_params] = zeros(m1.structure.n_total_params)
    m2.θ[1:m1.structure.n_total_params] = zeros(m2.structure.n_total_params)

    m1.θ[m1.structure.n_total_params + 1:end] = ones(m1.structure.n_total_params) * log(ℯ - 1)
    init_L = -20ones(BayesNet.triangular(m2.structure.n_total_params))
    global t = 0
    for i in 1:m2.structure.n_total_params
        global t += i
        init_L[t] = log(ℯ - 1)
    end
    m2.θ[m2.structure.n_total_params + 1:end] = init_L

    samples = 5randn(m1.structure.n_total_params, 2)

    res1 = BayesNet.log_diagonal_gaussian_posterior(m1, samples)
    res2 = BayesNet.log_full_gaussian_posterior(m2, samples)

    res1 ≈ res2
end

##########################################################################
####                            Test sets                             ####
##########################################################################

@testset verbose = true "VI functions" begin
    @testset "Helper Functions" begin
        @testset "set_col_matrix_expr()" begin
            @test test_set_col_matrix_expr_1(1)()
            @test test_set_col_matrix_expr_1(5)()
        end

        @testset "propagate_matrix_opp()" begin
            @test test_propagate_matrix_opp_basic()
            @test test_propagate_matrix_opp_chain()
        end

        @testset "complexity_cost_coefficients()" begin
            @test test_uniform_coef_1()
            @test test_uniform_coef_2()
            @test test_exponential_complexity_cost_1()
            @test test_exponential_complexity_cost_2()
        end

        @testset "Gaussian entropy" begin
            @test test_gaussian_entropy_1()
            @test test_gaussian_entropy_2()
        end
    end

    @testset "normalising flows" begin
        @test test_flow_transforms_1(2)
        @test test_flow_transforms_1(7)
        @test test_flow_transforms_2(2)
        @test test_flow_transforms_3()
    end

    @testset "probability distributions" begin
        @test test_log_diagonal_gaussian_posterior_size_correct(2)
        @test test_log_diagonal_gaussian_posterior_size_correct(10)
        @test test_log_full_gaussian_posterior_size_correct(4)
        @test test_log_full_gaussian_posterior_size_correct(7)
        @test test_equivalent_posterior_densities(1)
    end

    @testset "loss functions" begin
        @test test_variational_free_energy()
    end
end