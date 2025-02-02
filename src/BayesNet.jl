module BayesNet

include("HelperFunctions.jl")
include("model/Models.jl") 
include("model/Samplers.jl") 
include("Training.jl")
include("model/LaplaceBNN.jl") 
include("model/VI_BNN.jl") 
include("model/MCMC_BNN.jl") 

using .HelperFunctions
using .Models
using .Samplers
using .Training
using .LaplaceBNN
using .VI_BNN
using .MCMC_BNN

end
