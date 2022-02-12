

import Base: getindex


####################################
# COLUMN BLOCK MAP
####################################

struct ColBlockMap
    funcs::AbstractVector{Function}
    col_blocks::AbstractVector{AbstractRange}
end

function ColBlockMap(funcs, col_block_ids::AbstractVector{String})
    col_blocks = ids_to_ranges(col_block_ids)
    return ColBlockMap(funcs, col_blocks)
end


function getindex(cbm::ColBlockMap, rng::UnitRange)
    
    new_ranges, r_min_idx, r_max_idx = subset_ranges(cbm.col_blocks, rng)

    new_ranges = [(r.start - rng.start + 1):(r.stop - rng.start + 1) for r in new_ranges]

    return ColBlockMap(cbm.funcs[r_min_idx:r_max_idx], new_ranges)

end

function (cbm::ColBlockMap)(Z::AbstractMatrix)

    result = zero(Z) 
    for (ln_fn, rng) in zip(cbm.funcs, cbm.col_blocks)
        result[:,rng] = ln_fn(Z[:,rng])
    end
    return result
end



function ChainRules.rrule(cbm::ColBlockMap, Z)

    A = zero(Z)
    func_pullbacks = []
    for (fn, rng) in zip(cbm.funcs, cbm.col_blocks)
        (A_chunk, new_fn) = Zygote.pullback(fn, Z[:,rng])
        A[:,rng] .= A_chunk
        push!(func_pullbacks, new_fn)
    end

    function ColBlockMap_pullback(A_bar)
        Z_bar = zero(Z)
        for (pb, rng) in zip(func_pullbacks, cbm.col_blocks)
            Z_bar[:,rng] .= pb(A_bar[:,rng])[1]
        end
        return ChainRules.NoTangent(), Z_bar
    end

    return A, ColBlockMap_pullback

end


####################################################
# COLUMN BLOCK AGGREGATOR
####################################################

struct ColBlockAgg
    funcs::AbstractVector{Function}
    col_blocks::AbstractVector{AbstractRange}
end

function ColBlockAgg(funcs, col_block_ids::AbstractVector{String})
    col_blocks = ids_to_ranges(col_block_ids)
    return ColBlockAgg(funcs, col_blocks)
end

function (cba::ColBlockAgg)(Z::AbstractMatrix, A::AbstractMatrix, missing_data::AbstractMatrix)

    result = zero(Z) 
    for (i, (fn, rng)) in enumerate(zip(cba.funcs, cba.col_blocks))
        result[:, rng] .= fn(view(Z,:,rng), view(A,:,rng), view(missing_data,:,rng))
    end
    return result
end


function getindex(cba::ColBlockAgg, rng::UnitRange)
    
    new_ranges, r_min_idx, r_max_idx = subset_ranges(cba.col_blocks, rng)

    new_ranges = [(r.start - rng.start + 1):(r.stop - rng.start + 1) for r in new_ranges]

    return ColBlockAgg(cba.funcs[r_min_idx:r_max_idx], new_ranges)

end


function ChainRules.rrule(cba::ColBlockAgg, Z, A, missing_data)

    result = zero(Z) 
    func_pullbacks = []
    for (i, (fn, rng)) in enumerate(zip(cba.funcs, cba.col_blocks))
        (res, new_fn) = Zygote.pullback(fn, view(Z,:,rng), view(A,:,rng), view(missing_data,:,rng))
        result[:,rng] .= res
        push!(func_pullbacks, new_fn)
    end

    function ColBlockAgg_pullback(result_bar)
        Z_bar = similar(Z)
        A_bar = ChainRulesCore.ZeroTangent()
        missing_data_bar = ChainRulesCore.ZeroTangent()
        for (i,(pb, rng)) in enumerate(zip(func_pullbacks, cba.col_blocks))
            Z_bar[:,rng] .= pb(view(result_bar,:,rng))[1]
        end
        return ChainRules.NoTangent(), Z_bar, A_bar, missing_data_bar
    end

    return result, ColBlockAgg_pullback

end


