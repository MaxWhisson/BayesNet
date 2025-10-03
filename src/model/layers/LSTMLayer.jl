module LSTMLayer
    export  LSTM,
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
    using LogExpFunctions.logistic
    
    # struct for specifying a dense neural network layer.
    struct LSTM <: NNLayer
        n::Int                  # number of neurons in layer
        activation::Function    # activation function
    end

    function forward(l::LSTM, layer_params::AbstractVector,  
            input::AbstractArray{Float64}, state::AbstractVector{Float64})
        return lstm(state[1:l.n], state[l.n + 1:2l.n], input, layer_params[1])
    end

    function lstm(short::AbstractVector{Float64}, long::AbstractVector{Float64}, 
            input::AbstractVector{Float64}, params::AbstractMatrix{Float64})
        t1 = logistic.(short .* params[:,1] + input .* params[:,2])
        t2 = logistic.(short .* params[:,3] + input .* params[:,4] + params[:,9])
        t3 = tanh.(short * params[:,5] + input .* params[:,6] + params[:,10])
        t4 = logistic.(short .* params[:,7] + input .* params[:,8] + params[:,11])

        longout = tanh(long .* t1 + (t2 .* t3))
        shortout = longout .* t4

        return (shortout, [shortout;longout])
    end

    function initialise_parameters(l::LSTM, n_in::Int)
        return [zeros(11l.n)]
    end

    function output_dimension(l::LSTM)
        return l.n
    end

    function n_weights(l::LSTM, n_in::Int) 
        return 11l.n
    end

    function extract_parameters(l::LSTM, params::AbstractVector{Float64}, i::Int, 
            last_output_n::Int)
        j = i + 11l.n
        return (j, [params[i:j - 1]])
    end

    function init_state(l::LSTM)
        return zeros(i + 2n, state[i:i + 2n - 1])
    end
end