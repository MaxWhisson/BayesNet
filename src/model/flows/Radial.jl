module Radial
    export  RadialFlowLayer

    using ..Normalising

    function RadialFlowLayer(D::Int)
        h = (α, r) -> 1/(α + r)
        h′ = (a, r) -> -1/((a + r) ^ 2)

        function abs_det(θ::AbstractArray{Float64})
            α = abs(θ[1])
            β = -α + log(1 + exp(θ[2]))
            z₀ = θ[3:D+2]
            function f(z::AbstractArray{Float64})
                r = norm(z - z₀)
                t1 = ((1 + β * h(α, r)) ^ (D - 1))
                t2 = 1 + β * h(α, r) + β * h′(α, r) * r
                return t1 * t2
            end
        end

        function func(θ::AbstractArray{Float64})
            α = abs(θ[1])
            β = -α + log(1 + exp(θ[2]))
            z₀ = θ[3:D+2]
            function f(z::AbstractArray{Float64})
                r = norm(z - z₀)
                return z + β * h(α, r) * (z - z₀)
            end
        end

        function inverse_func(θ::AbstractArray{Float64})
            α = abs(θ[1])
            β = -α + log(1 + exp(θ[2]))
            z₀ = θ[3:D+2]
            function f(z′::AbstractArray{Float64})
                r′ = norm(z′ - z₀)
                r = ((r′ - α - β) + sqrt(((α + β - r′) ^ 2) + 4r′ * α)) / 2
                return z₀ + (z′ - z₀) / (1 + β * h(α, r))
            end
        end

        return NormalisingFlowLayer(
            D,
            D + 2,
            func,
            abs_det,
            inverse_func
        )
    end
end