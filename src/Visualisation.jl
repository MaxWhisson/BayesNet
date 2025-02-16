module Visualisation

export  plot_normalising_flow_vector_field!

using Plots

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

    quiver!(plt, all_xs, all_ys, quiver=(((x, y) -> f([x;y]) - [x;y])), xlim=xlim, ylim=ylim)
end

end