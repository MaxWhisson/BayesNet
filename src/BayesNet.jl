module BayesNet

include("HelperFunctions.jl")

include("model/types/ModelTypes.jl")

include("model/flows/Normalising.jl")
include("model/flows/Planar.jl")
include("model/flows/Radial.jl")

include("model/layers/Layer.jl")
include("model/layers/DenseLayer.jl")
include("model/layers/ResidualLayer.jl")
include("model/layers/LSTMLayer.jl")

include("model/Models.jl") 
include("Training.jl")

include("model/Samplers.jl") 
include("model/LaplaceBNN.jl") 
include("model/VI_BNN.jl") 
include("model/MCMC_BNN.jl") 

include("Analysis.jl")

using .ModelTypes
using .ModelFunctions

using .Normalising
using .Planar
using .Radial

using .Layer
using .DenseLayer
using .ResidualLayer
using .LSTMLayer

using .HelperFunctions
using .Models
using .Samplers
using .Training
using .LaplaceBNN
using .VI_BNN
using .MCMC_BNN
using .Analysis

end
