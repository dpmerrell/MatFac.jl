module BatchMatFac

using ChainRules, Zygote, CUDA
include("batch_qty.jl")
include("link_functions.jl")

end # module
