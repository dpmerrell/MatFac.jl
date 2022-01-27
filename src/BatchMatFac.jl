module BatchMatFac

using ChainRules, Zygote, CUDA, HDF5, LinearAlgebra, SparseArrays

include("typedefs.jl")
include("util.jl")
include("block_matrix.jl")
include("col_block_map.jl")
include("batch_iter.jl")

include("noise_models.jl")
include("model_def.jl")
include("model_core.jl")
include("io.jl")
include("fit.jl")


end # module
