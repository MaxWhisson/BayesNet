module Layer
    export  NNLayer,
            forward,
            initialise_parameters,
            output_dimension,
            n_weights,
            extract_parameters,
            init_state,
            get_layer_state,
            get_n_params

    abstract type NNLayer end

    function forward(l::NNLayer, layer_params::AbstractVector,
            inputs::AbstractArray, state::AbstractMatrix{Float64})
        throw("unimplemented 'forward' method")
    end

    # gives vector of parameter structure from vector of raw parameters 
    function extract_parameters(l::NNLayer, params::AbstractVector, i::Int, 
            last_outputs_n::AbstractVector{Int})
        throw("unimplemented 'extract_parameters' method")
    end

    function initialise_parameters(l::NNLayer, ns_in::Vector{Int})
        throw("unimplemented 'initialise_parameters' method")
    end

    function output_dimension(l::NNLayer)
        throw("unimplemented 'output_dimension' method")
    end

    function n_weights(l::NNLayer, ns_in::Vector{Int})
        throw("unimplemented 'n_weights' method")
    end

    # init state for single sample
    function init_state(l::NNLayer)::Vector{Float64}
        throw("unimplemented 'extract_state' method")
    end

    # get state used as output in recursive structures
    function get_layer_state(l::NNLayer, state::AbstractArray)
        throw("unimplemented 'get_layer_state' method")
    end

    function get_n_params(l::NNLayer, in_ns::Vector{Int})
        throw("unimplemented 'get_n_params' method")
    end
end