module ResidualLayer
    export  Residual,
            forward,
            initialise_parameters,
            output_dimension,
            n_weights,
            extract_parameters

    import ..Layer:NNLayer, forward, initialise_parameters, output_dimension, n_weights, extract_parameters
    import Flux.glorot_uniform

    # struct for specifying a residual neural network layer.
    struct Residual <: NNLayer
        n1::Int
        n2::Int
        activation::Function
    end

    function extract_parameters(l::Residual, params::AbstractVector, i1::Int, 
            last_output_n::Int)
        j1 = i1 + last_output_n * l.n1
        weights_1 = reshape(params[i1:j1 - 1], (l.n1, last_output_n))
        biases_1 = params[j1:j1 + l.n1 - 1]

        i2 = j + l.n1
        j2 = i2 + l.n1 * l.n2
        weights_2 = reshape(params[i2:j2 - 1], (l.n2, l.n1))
        biases_2 = params[j2:j2 + l.n2 - 1]
        
        return (j2 + l.n2, [weights_1, biases_1, weights_2, biases_2])
    end

    function forward(l::Residual, layer_params::AbstractVector, 
            input::AbstractArray{Float64})
        output_l1 = s.layers[i].activation.(layer_params[1] * input .+ layer_params[2])
        return input + (layer_params[3] * output_l1 .+ layer_params[4])
    end

    function initialise_parameters(l::Residual, n_in::Int)
        return [
            reshape(glorot_uniform(n_in, l.n1), (n_in * l.n1, 1));
            zeros(l.n1);
            reshape(glorot_uniform(l.n1, l.n2), (l.n1 * l.n2, 1));
            zeros(l.n2)
        ]
    end

    function output_dimension(l::Residual)
        return l.n2
    end

    function n_weights(l::Residual, n_in::Int)
        return (l.n1 + l.n2) + (n_in * l.n1) + (l.n1 * l.n2)
    end
end