
module BatchMatFac

using Flux, Functors, ChainRules, ChainRulesCore

import Flux.trainable

include("util.jl")
include("batch_array.jl")
include("col_map.jl")
include("model_core.jl")
include("noise_models.jl")
include("model.jl")

end


