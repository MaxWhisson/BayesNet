module BayesNet

include("Training.jl")
include("Visualisation.jl")
include("model/LaplaceBNN.jl") 
include("model/VI_BNN.jl") 
include("model/MCMC_BNN.jl") 
include("model/Models.jl") 
include("model/Samplers.jl") 

end
