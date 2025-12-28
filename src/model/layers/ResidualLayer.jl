module ResidualLayer
    export  Residual,
            forward,
            initialise_parameters,
            output_dimension,
            n_weights,
            extract_parameters,
            init_state,
            get_layer_state

    import ..Layer:NNLayer, 
        forward, 
        initialise_parameters, 
        output_dimension, n_weights, 
        extract_parameters,
        init_state,
        get_layer_state

    import Flux.glorot_uniform

    using ..ModelTypes: Model, 
        EvalStrategy,
        ModelStructure

    using ..ModelFunctions: init_means,
            count_params

    using ..HelperFunctions: deep_foldl

    # struct for specifying a residual neural network layer.
    struct Residual <: NNLayer
        inner::ModelStructure
    end

    function forward(l::Residual, layer_params::AbstractVector, 
            inputs::AbstractArray, initState::AbstractMatrix{Float64})
        (res, state) = pred(l.inner, layer_params, sum(inputs), initState)
        return (sum(inputs) + res, state)
    end

    function __extract_params(l::NNLayer, params::AbstractVector, pos::Int, 
            extracted_i::Int, in_ns::Vector{Int}, current_extraction::AbstractVector)
        (res, newPos) = extract_parameters(l, params, pos, in_ns)
        return (
            newPos,
            [current_extraction[1:extracted_i];res;current_extraction[extracted_i + length(res) + 1:end]],
            extracted_i + length(res) + 1
        )
    end

    function extract_parameters(l::Residual, params::AbstractVector, param_pos::Int, 
            in_ns::Vector{Int})
        n_total_params = get_n_params(l, in_ns)
        deep_foldl(
            ((param_i, extracted_i, current_params), (layer_i, layer_ins_i)) -> 
                __extract_parameters(
                    l.inner.layers[layer_i], 
                    params, 
                    param_i,
                    extracted_i, 
                    [j < 0 ? in_ns[abs(j)] : output_dimension(l.inner.layers[j]) for j in layer_ins_i],
                    current_params
                ),
            l.inner.evaluation,
            (param_pos, 0, [zeros(0) for i in 1:n_total_params])
        )[1,2]
    end

    # ns_in must be single element
    function initialise_parameters(l::Residual, ns_in::Vector{Int})
        init_means(l.inner.layers, l.inner.evaluation, ns_in)
    end

    function output_dimension(l::Residual)
        return output_dimension(l.inner.layers[end])
    end

    function n_weights(l::Residual, ns_in::Vector{Int})
        count_params(l.inner.layers, l.inner.evaluation, ns_in)
    end

    function init_state(s::Substructure)::Vector{Float64}
        reduce(vcat, s.inner.layers .|> (l -> init_state(l)))
    end

    function get_layer_state(l::Residual, state::AbstractVector)
        return get_layer_state(l.inner, state)
    end

    function get_n_params(l::LSTM, in_ns::Vector{Int})
        deep_foldl(
            (count, (layer_i, layer_ins_i)) -> count + get_n_params(
                l.inner.layers[layer_i], 
                [i < 0 ? in_ns[abs(i)] : output_dimension(l.layers[i]) for i in layer_ins_i]
            ),
            l.inner.evaluation,
            0
        )
    end
end