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

    # struct for specifying a residual neural network layer.
    struct Residual <: NNLayer
        n1::Int
        n2::Int
        activation::Function
    end

    function extract_parameters(l::Residual, params::AbstractVector, pos::Int, 
            last_outputs_n::Vector{Int})
        extractedParams = Vector(undef, 2 * (length(last_outputs_n) + 1))
        nextPos = 0

        for i in eachindex(last_outputs_n)
            nextPos = pos + last_outputs_n[i] * l.n1
            extractedParams[2(i - 1) + 1] = reshape(
                params[pos:nextPos - 1],
                (l.n1, last_outputs_n[i])
            )

            pos = nextPos
            nextPos += l.n1
            extract_parameters[2i] = params[pos: nextPos - 1]
            pos = nextPos
        end

        pos = nextPos
        nextPos += l.n1 * l.n2
        extract_parameters[end - 1] = reshape(params[pos:nextPos - 1], (l.n2, l.n1))

        pos = nextPos
        nextPos += l.n2
        extract_parameters[end] = params[pos:nextPos - 1]
    
        return (nextPos, extractedParams)
    end

    function forward(l::Residual, layer_params::AbstractVector, 
            inputs::AbstractArray, state::AbstractMatrix{Float64})
        output = [
            l.activation.(layer_params[2(i - 1) + 1] * input .+ layer_params[2i])
            for i in eachindex(inputs)
        ] |> sum
        return (sum(inputs) + (layer_params[end - 1] * output .+ layer_params[end]), [])
    end

    function initialise_parameters(l::Residual, ns_in::Vector{Int})
        return [([
            [reshape(glorot_uniform(n_in, l.n1), (n_in * l.n1, 1));
            zeros(l.n1)]
            for n_in in ns_in
        ] |> (x -> reduce(vcat, x)));
            reshape(glorot_uniform(l.n1, l.n2), (l.n1 * l.n2, 1));
            zeros(l.n2)
        ]
    end

    function output_dimension(l::Residual)
        return l.n2
    end

    function n_weights(l::Residual, ns_in::Vector{Int})
        return l.n2 + (l.n1 * l.n2) + [l.n1 + (n_in * l.n1) for n_in in ns_in] |> sum
    end

    function init_state(l::Residual)
        return []
    end

    # state will be [] here
    function get_layer_state(l::Residual, state::AbstractVector)
        return zeros(l.n2)
    end
end