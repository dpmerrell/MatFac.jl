
using ChainRules, Zygote, CUDA



####################################
# LINK FUNCTIONS
####################################

function quad_link(Z)
    return Z
end

function logistic_link(Z)
    return 1.0 ./ (1.0 .+ exp.(-Z))
end

function poisson_link(Z)
    return exp.(Z)
end


NOISE_MODELS = ["normal","logistic","poisson"]


LINK_FUNCTION_MAP = Dict("normal"=>quad_link,
                         "logistic"=>logistic_link,
                         "poisson"=>poisson_link
                         )


#Z = CUDA.ones(10,10)
#println("Z")
#println(Z)
#plus(x) = x .+ 1
#double(x) = x .* 2
#funcs = [plus, double]
#ranges = [1:5,6:10]
#
#my_crm = ColRangeMap(funcs, ranges)
#println("INITIALIZED MAP FUNCTOR")
#println(my_crm)
#
#grad = gradient(x -> sum(my_crm(x)), Z)
#
#println("COMPUTED GRADIENT")
#println(grad)
