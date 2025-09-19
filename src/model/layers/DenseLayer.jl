module DenseLayer
    export  Dense,
            forward,
            initialise_parameters,
            output_dimension,
            n_weights,
            extract_parameters

    import ..Layer:NNLayer, forward, initialise_parameters, output_dimension, n_weights, extract_parameters
    import Flux.glorot_uniform
    
    # struct for specifying a dense neural network layer.
    struct Dense <: NNLayer
        n::Int                  # number of neurons in layer
        activation::Function    # activation function
    end

    function forward(l::Dense, layer_params::AbstractVector, 
            input::AbstractArray{Float64})
        return l.activation.(layer_params[1] * input .+ layer_params[2])
    end

    function initialise_parameters(l::Dense, n_in::Int)
        return [reshape(glorot_uniform(n_in, l.n), (n_in * l.n, 1));zeros(l.n)]
    end

    function output_dimension(l::Dense)
        return l.n
    end

    function n_weights(l::Dense, n_in::Int) 
        return l.n + n_in * l.n
    end

    function extract_parameters(l::Dense, params::AbstractVector{Float64}, i::Int, 
            last_output_n::Int)
        j = i + last_output_n * l.n
        weights = reshape(params[i:j - 1], (l.n, last_output_n))
        biases = params[j: j + l.n - 1]
        return (j + l.n, [weights, biases])
    end
end