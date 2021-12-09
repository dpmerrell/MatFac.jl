
using ChainRules, Zygote, CUDA


struct ColRangeMap
    funcs::AbstractVector
    col_ranges::AbstractVector
end


function (crm::ColRangeMap)(Z::AbstractMatrix)

    result = zero(Z) 
    for (ln_fn, rng) in zip(crm.funcs, crm.col_ranges)
        result[:,rng] = ln_fn(Z[:,rng])
    end
    return result
end


function ChainRules.rrule(crm::ColRangeMap, Z::AbstractMatrix)

    A = zero(Z)
    func_pullbacks = []
    for (fn, rng) in zip(crm.funcs, crm.col_ranges)
        (A_chunk, new_fn) = Zygote.pullback(fn, Z[:,rng])
        A[:,rng] .= A_chunk
        push!(func_pullbacks, new_fn)
    end

    function ColRangeMap_pullback(A_bar)
        Z_bar = zero(Z)
        for (pb, rng) in zip(func_pullbacks, crm.col_ranges)
            Z_bar[:,rng] .= pb(A_bar[:,rng])[1]
        end
        return ChainRules.NoTangent(), Z_bar
    end

    return A, ColRangeMap_pullback

end


function quad_link(Z)
    return Z
end

function logistic_link(Z)
    return 1.0 ./ (1.0 .+ exp.(-Z))
end

function poisson_link(Z)
    return exp.(Z)
end


LINK_FUNCTIONS = ["quad","logistic","poisson"]
LINK_FUNCTION_MAP = Dict("quad"=>quad_link,
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
