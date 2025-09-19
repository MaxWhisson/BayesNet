module Normalising
    export  NormalisingFlowLayer,
            inverse_flow

    # normalising flow is vector of layers
    mutable struct NormalisingFlowLayer
        D::Int
        n_params::Int
        func::Function                      # θ -> x -> Type
        jacobian_determinant::Function      # θ -> x -> Type
        inverse_func::Function              # θ -> x -> Type
    end

    function inverse_flow(zₙ::AbstractArray{Float64}, 
            flow_params::AbstractArray{Float64}, 
            flow::AbstractArray{NormalisingFlowLayer})
        foldl(
            ((m, param_index, i), l) -> (
                set_col_matrix_expr(m, i + 1, l.inverse_func(
                    flow_params[param_index:param_index+l.n_params - 1]
                )(m[:,i])),
                param_index + l.n_params,
                i + 1
            ),
            reverse(flow),
            init=([zₙ zeros(length(zₙ), length(flow))], 1, 1)
        )[1]
    end
end