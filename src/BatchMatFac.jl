module BatchMatFac

using ChainRules, ChainRulesCore, Zygote, CUDA, HDF5, 
      LinearAlgebra, SparseArrays, Statistics

include("typedefs.jl")
include("util.jl")
include("arithmetic.jl")
include("batch_matrix.jl")
include("col_block_map.jl")
include("batch_iter.jl")

include("noise_models.jl")
include("model_def.jl")
include("model_core.jl")
include("model_params.jl")
include("adagrad_updater.jl")
include("io.jl")
include("fit.jl")

include("simulate.jl")

end # module
