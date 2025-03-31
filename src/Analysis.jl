module Analysis

export  plot_normalising_flow_vector_field!,
        accuracy,
        prune_diagonal_gaussian_proportion!,
        prune_diagonal_gaussian_CI!

using Plots
using Statistics
using Distributions

using ..Models
using ..VI_BNN

# simple plotting for 2D 
function plot_normalising_flow_vector_field!(
        m::Models.Model, 
        plt::Plots.Plot;
        xlim::Tuple{Float64, Float64} = (-2,2),
        ylim::Tuple{Float64, Float64} = (-2,2),
        axisSamples::Int = 20)
    xs = range(xlim[1], xlim[2], axisSamples)
    ys = range(ylim[1], ylim[2], axisSamples)
    all_xs = [x for x in xs for y in ys]
    all_ys = [y for x in xs for y in ys]

    flow_params = m.θ[m.n_variational_params+1:end]
    f = VI_BNN.transform_sample(m.normalising_flow, flow_params)

    quiver!(plt, all_xs, all_ys, quiver=(((x, y) -> f([x;y]) - [x;y])), 
        xlim=xlim, ylim=ylim)
end

function accuracy(y, preds)
    return 1 - mean(((x1, x2) -> !(x1 ≈ x2)).(y, preds))
end

# remove proportion of highest variance weights' variational parameters 
function prune_diagonal_gaussian_proportion!(m; proportion = 0.5)
    θ = m.θ[1:m.n_variational_params]
    μ = θ[1:m.structure.n_total_params]
    log_σ = θ[m.structure.n_total_params + 1:end]

    sorted = zip(
        abs.(μ) ./ exp.(log_σ), 
        1:m.structure.n_total_params
    ) |> collect |> sort
    # zero everything below this
    cutoff_i = ceil(proportion * m.structure.n_total_params)

    # indexes of params to be zeroed
    to_zero = (x -> x[2]).(sorted)[1:Int(cutoff_i)]
    m.θ[to_zero] .= 0 
    m.θ[to_zero .+ m.structure.n_total_params] .= -Inf
end

# remove parameters for weights that are not significantly different
# from 0
function prune_diagonal_gaussian_CI!(m; CI_probability = 0.9)
    function f(μ, log_σ)
        cdf(Normal(μ, exp(log_σ)), 0) > 1 - CI_probability / 2
    end

    function g(offset, m)
        function h(i)
            if f(m.θ[i], m.θ[1 + offset])
                m.θ[i], m.θ[1 + offset] = 0, -Inf
            end
        end
    end
        
    g(m.structure.n_total_params, m).(1:m.structure.n_total_params)
end

end