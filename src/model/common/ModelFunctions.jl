module ModelFunctions

export  init_means

import ..ModelTypes: Model,
        EvalStrategy,
        ModelStructure,
        ParameterisedFunction,
        VariationalModel,
        LaplaceModel,
        MCMC_Model,
        DegenerateModel

function init_means(layers::Vector{<:NNLayer}, evalOrder::EvalStrategy, input_ns::Vector{Int})
    means = Vector(undef, length(layers))

    for i in eachindex(evalOrder)
        outputs_n = evalOrder[i][2] .|> (x -> x < 0 ? input_ns[abs(x)] : output_dimension(layers[x]))
        means[i] = initialise_parameters(layers[i], outputs_n)
    end

    foldl(
        (acc, w) -> [acc;w],
        means,
        init = []
    ) |> vec
end

function count_params(layers::Vector, evalOrder::EvalStrategy, input_ns::Vector{Int})
    foldl(
        ((n, last_i), (li, inputIndexes)) -> (
            n + n_weights(layers[li], [j < 0 ? input_ns[abs(j)] : output_dimension(layers[j]) for j in inputIndexes]), 
            last_i + 1
        ),
        evalOrder,
        init = (0, 0)
    )[1]
end

end