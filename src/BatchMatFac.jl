
module BatchMatFac

using Flux, Functors, Zygote, ChainRules, ChainRulesCore, CUDA, BSON,
      Distributions

include("util.jl")
include("batch_iter.jl")
include("batch_array.jl")
include("model_core.jl")
include("noise_models.jl")
include("model.jl")
include("update.jl")
include("fit.jl")

end


