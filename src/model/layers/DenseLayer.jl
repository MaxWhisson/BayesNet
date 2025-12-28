module DenseLayer
    export  Dense,
            forward,
            initialise_parameters,
            output_dimension,
            n_weights,
            extract_parameters,
            init_state,
            get_layer_state,
            get_n_params

    import ..Layer:NNLayer, 
        forward, 
        initialise_parameters, 
        output_dimension, n_weights, 
        extract_parameters,
        init_state,
        get_layer_state,
        get_n_params
    
    import Flux.glorot_uniform
    
    import ..HelperFunctions: trace, deep_foldl

    # struct for specifying a dense neural network layer.
    struct Dense <: NNLayer
        n::Int                  # number of neurons in layer
        activation::Function    # activation function
    end

    function forward(l::Dense, layer_params::AbstractVector, 
            inputs::AbstractVector, _::AbstractMatrix{Float64})
        return ([
            l.activation.(layer_params[2(i - 1) + 1] * inputs[i] .+ layer_params[2i])
            for i in eachindex(inputs)
        ] |> sum, zeros(0, size(inputs[1], 2)))
    end

    function initialise_parameters(l::Dense, ns_in::Vector{Int})
        return [
            [reshape(glorot_uniform(n_in, l.n), (n_in * l.n, 1));zeros(l.n)]
            for n_in in ns_in
        ] |> (x -> reduce(vcat, x))
    end

    function output_dimension(l::Dense)
        return l.n
    end

    function n_weights(l::Dense, ns_in::Vector{Int}) 
        return [l.n + n_in * l.n for n_in in ns_in] |> sum
    end

    function extract_parameters(l::Dense, params::AbstractVector{Float64}, 
            pos::Int, in_ns::Vector{Int})
        deep_foldl(
            ((curPos, extractedParams), i) ->
            (
                pos + (in_ns[i] + 1) * l.n,
                [
                    extractedParams[1:2(i - 1)];
                    [reshape(
                        params[pos:pos + in_ns[i] * l.n - 1],
                        (l.n, in_ns[i])
                    )];
                    [params[
                        pos + in_ns[i] * l.n:pos + (out_n[i] + 1) * l.n - 1
                    ]];
                    extractedParams[2i + 1:end]
                ]
            ),
            eachindex(in_ns),
            (pos, fill(zeros(0,0), 2 * length(in_ns)))
        )
    end

    function init_state(_::Dense)::Vector{Float64}
        return []
    end

    function get_layer_state(l::Dense, _::AbstractArray)
        return zeros(l.n)
    end

    function get_n_params(_::Dense, in_ns::Vector{Int})
        return 2 * length(in_ns)
    end
end