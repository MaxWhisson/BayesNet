module ConvolutionalLayer
    export  Convolutional,
            forward,
            initialise_parameters,
            output_dimension,
            n_weights,
            extract_parameters,
            init_state

    import ..Layer:NNLayer, 
        forward, 
        initialise_parameters, 
        output_dimension, n_weights, 
        extract_parameters,
        init_state
    
    import Flux.glorot_uniform
    
    # struct for specifying a dense neural network layer.
    struct Convolutional <: NNLayer
        n::Int                  # number of nodes
        kernel_dims::Tuple
        channels::Int
        stride::Int
        dilation::Int
        padding::n
    end

    function zero_pad_matrix(x::AbstractArray, n)
        return zeros(size(x) .+ 2n)[n:end-n, n:end-n] = x
    end

    function forward(l::Convolutional, layer_params::AbstractVector, 
            input::AbstractArray{Float64}, state::AbstractVector{Float64})
        # TODO
    end

    function initialise_parameters(l::Convolutional, n_in::Int)
        # TODO
    end

    function output_dimension(l::Convolutional)
        # TODO
    end

    function n_weights(l::Convolutional, n_in::Int) 
        # TODO
    end

    function extract_parameters(l::Convolutional, params::AbstractVector{Float64}, i::Int, 
            last_output_n::Int)
        # TODO
    end

    function init_state(l::Convolutional)
        return []
    end
end