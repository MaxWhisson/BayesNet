module LSTMLayer
    export  LSTM,
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
    using LogExpFunctions
    import ..HelperFunctions: trace, deep_foldl
    
    # struct for specifying a dense neural network layer.
    struct LSTM <: NNLayer
        n::Int                  # number of neurons in layer
        activation::Function    # activation function
    end

    function forward(l::LSTM, layer_params::AbstractVector,  
            inputs::AbstractVector, state::AbstractVector{Float64})
        input = sum(inputs)
        reshapedState = reshape(state, (2l.n, l.n))
        lstm(
            reshapedState[1:l.n,:],
            reshapedState[l.n + 1: end,:],
            input,
            reshape(layer_params[1], (l.n, 11))
        )
    end

    function lstm(short::AbstractMatrix{Float64}, long::AbstractMatrix{Float64}, 
            input::AbstractVector{Float64}, params::AbstractMatrix{Float64})
        t1 = logistic.(short .* params[:,1] + input .* params[:,2])
        t2 = logistic.(short .* params[:,3] + input .* params[:,4] + params[:,9])
        t3 = tanh.(short * params[:,5] + input .* params[:,6] + params[:,10])
        t4 = logistic.(short .* params[:,7] + input .* params[:,8] + params[:,11])

        longout = tanh(long .* t1 + (t2 .* t3))
        shortout = longout .* t4

        return (shortout, [shortout;longout])
    end

    function initialise_parameters(l::LSTM, _::Vector{Int})
        return [zeros(11l.n)]
    end

    function output_dimension(l::LSTM)
        return l.n
    end

    function n_weights(l::LSTM, _::Vector{Int}) 
        return 11l.n
    end

    function extract_parameters(l::LSTM, params::AbstractVector{Float64}, i::Int, 
            _::AbstractVector{Int})
        j = i + 11l.n
        return (j, [params[i:j - 1]])
    end

    function init_state(l::LSTM)
        return zeros(2l.n)
    end

    function get_layer_state(l::LSTM, state::AbstractArray)
        return state[1:l.n,:]
    end

    function get_n_params(_::LSTM, in_ns::Vector{Int})
        return 1
    end
end