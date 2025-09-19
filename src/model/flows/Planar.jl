module Planar
    export  PlanarFlowLayer

    using ..Normalising

    # h is smooth non-linearity
    function PlanarFlowLayer(D::Int)
        h = tanh
        function ψ(w::AbstractArray{Float64}, b::Float64)
            function f(z)
                x = w'z + b
                gradient(y -> h(y), x)[1] * w
            end
        end

        return NormalisingFlowLayer(
            D,
            2D + 1,
            θ -> (z -> z + (θ[1:D] .* h(θ[D+1:2D]' * z + θ[2D+1]))),
            θ -> (z -> abs(1 + θ[1:D]'ψ(θ[D+1:2D], θ[2D+1])(z))),
            x -> throw("No analytical inverse for Planar Transform")
        )
    end
end