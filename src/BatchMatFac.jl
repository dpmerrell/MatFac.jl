module BatchMatFac

using ChainRules, Zygote, CUDA, HDF5, LinearAlgebra


include("util.jl")
include("custom_ops.jl")
include("link_functions.jl")
include("model.jl")
include("losses.jl")

end # module
