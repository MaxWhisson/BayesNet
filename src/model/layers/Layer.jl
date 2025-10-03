module Layer
    export  NNLayer,
            forward,
            initialise_parameters,
            output_dimension,
            n_weights,
            extract_parameters,
            init_state

    abstract type NNLayer end

    function forward(l::NNLayer, layer_params::AbstractVector,
            input::AbstractArray{Float64}, state::AbstractVector{Float64})
        throw("unimplemented 'forward' method")
    end

    function extract_parameters(l::NNLayer, params::AbstractVector, i::Int, 
            last_output_n::Int)
        throw("unimplemented 'extract_parameters' method")
    end

    function initialise_parameters(l::NNLayer, n_in::Int)
        throw("unimplemented 'initialise_parameters' method")
    end

    function output_dimension(l::NNLayer)
        throw("unimplemented 'output_dimension' method")
    end

    function n_weights(l::NNLayer, n_in::Int)
        throw("unimplemented 'n_weights' method")
    end

    function init_state(l::NNLayer)
        throw("unimplemented 'extract_state' method")
    end
end