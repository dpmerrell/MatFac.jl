

import Base: getindex, setindex!


##########################################
# "BATCH QUANTITY" DEFINITION 
##########################################
mutable struct BatchQuantity 
    values::AbstractMatrix
    row_ranges::AbstractVector{UnitRange}
    col_ranges::AbstractVector{UnitRange}
end


function BatchQuantity(values, row_batches::Vector{Int}, col_batches::Vector{Int})
   
    row_ranges = ids_to_ranges(row_batches)
    col_ranges = ids_to_ranges(col_batches)
    
    return BatchQuantity(values, row_ranges, col_ranges)

end


# For now, we only subset by row.
# All columns are included.
# Note an important difference with AbstractMatrix
# semantics: the result of getindex retains the original
# r
function getindex(A::BatchQuantity, row_range::UnitRange, col_range)

    r_min = row_range.start
    r_max = row_range.stop

    if r_min > r_max
        return row_range
    end

    @assert r_min >= A.row_ranges[1].start
    @assert r_max <= A.row_ranges[end].stop

    row_starts = [rr.start for rr in A.row_ranges]
    r_min_idx = searchsorted(row_starts, r_min).stop
    
    row_stops = [rr.stop for rr in A.row_ranges]
    r_max_idx = searchsorted(row_stops, r_max).start

    new_row_ranges = row_ranges[r_min_idx:r_max_idx]
    new_row_ranges[1] = r_min:new_row_ranges[1].stop
    new_row_ranges[end] = new_row_ranges[end].start:r_max

    return BatchQuantity(A.values[r_min_idx:r_max_idx,:],
                         new_row_ranges,
                         A.col_ranges)
end


##########################################
# "BATCH QUANTITY" ADDITION 
##########################################

function bq_add(A::AbstractMatrix, B::BatchQuantity)

    result = zero(A)
    for (i, row_range) in enumerate(B.row_ranges) 
        for (j, col_range) in enumerate(B.col_ranges)
            result[row_range,col_range] .= A[row_range,col_range] .+ B.values[i,j]
        end
    end
    return result
end


function ChainRules.rrule(::typeof(bq_add), A::AbstractMult, B::BatchQuantity)

    Z = bq_add(A,B)

    function bq_add_pullback(Z_bar)
        A_bar = Z_bar
        B_bar = zero(B.values)
        for (i,row_range) in enumerate(B.row_ranges)
            for (j,col_range) in enumerate(B.col_ranges)
                B_bar[i,j] = sum(Z_bar[row_range,col_range])
            end
        end

        return ChainRules.NoTangent(), A_bar, ChainRules.Tangent{BatchQuantity}(;values=B_bar)
    end
    return Z, bq_add_pullback
end


function bq_add!(A::BatchQuantity, B::BatchQuantity)

    # Locate the index of the first A.value touched by B
    A_starts = Int[rng.start for rng in A.row_ranges]
    A_idx_min = searchsorted(A_starts, B.row_ranges[1].start).stop
    
    # Locate the index of the last A.value touched by B
    A_stops = Int[rng.stop for rng in A.row_ranges]
    A_idx_max = searchsorted(A_stops, B.row_ranges[end].stop).start

    A_idx = A_idx_min
    B_idx = 1
    B_l = A.row_ranges[B_idx].start
    B_u = A.row_ranges[B_idx].stop
    while A_idx <= A_idx_max

        A_l = A.row_ranges[A_idx].start
        A_u = A.row_ranges[A_idx].stop
        
    end


end


##########################################
# "BATCH QUANTITY" MULTIPLICATION 
##########################################

function bq_mult(A::AbstractMatrix, B::BatchQuantity)

    result = zero(A)
    for (i, row_range) in enumerate(B.row_ranges) 
        for (j, col_range) in enumerate(B.col_ranges)
            result[row_range,col_range] .= A[row_range,col_range] .* B.values[i,j]
        end
    end
    return result
end


function ChainRules.rrule(::typeof(bq_mult), A::AbstractMatrix, B::BatchQuantity)

    Z = bq_mult(A,B)

    function bq_add_pullback(Z_bar)
        A_bar = copy(Z_bar)
        B_bar = zero(B.values)
        for (i,row_range) in enumerate(B.row_ranges)
            for (j,col_range) in enumerate(B.col_ranges)
                A_bar[row_range,col_range] .*= B.values[i,j]
                B_bar[i,j] = sum(Z_bar[row_range,col_range] .* A[row_range,col_range])
            end
        end

        return ChainRules.NoTangent(), A_bar, ChainRules.Tangent{BatchQuantity}(;values=B_bar)
    end
    return Z, bq_add_pullback
end


####################################
# COLUMN RANGE MAP
####################################

struct ColRangeMap
    funcs::AbstractVector{Function}
    col_ranges::AbstractVector{AbstractRange}
end

function ColRangeMap(funcs, col_batch_ids::AbstractVector{String})
    col_ranges = ids_to_ranges(col_batch_ids)
    return ColRangeMap(funcs, col_ranges)
end


function (crm::ColRangeMap)(Z::AbstractMatrix)

    result = zero(Z) 
    for (ln_fn, rng) in zip(crm.funcs, crm.col_ranges)
        result[:,rng] = ln_fn(Z[:,rng])
    end
    return result
end



function ChainRules.rrule(crm::ColRangeMap, Z)

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


####################################################
# COLUMN RANGE AGGREGATOR
####################################################

struct ColRangeAgg
    funcs::AbstractVector{Function}
    col_ranges::AbstractVector{AbstractRange}
end

function ColRangeAgg(funcs, col_batch_ids::AbstractVector{String})
    col_ranges = ids_to_ranges(col_batch_ids)
    return ColRangeAgg(funcs, col_ranges)
end

function (cra::ColRangeAgg)(Z::AbstractMatrix, A::AbstractMatrix)

    result = zeros(Float32, length(cra.col_ranges)) 
    for (i, (fn, rng)) in enumerate(zip(cra.funcs, cra.col_ranges))
        result[i] = fn(Z[:,rng], A[:,rng])
    end
    return result
end


function ChainRules.rrule(cra::ColRangeAgg, Z, A)

    result = zeros(Float32, length(cra.col_ranges)) 
    func_pullbacks = []
    for (i, (fn, rng)) in enumerate(zip(cra.funcs, cra.col_ranges))
        (res, new_fn) = Zygote.pullback(fn, Z[:,rng], A[:,rng])
        result[i] = res
        push!(func_pullbacks, new_fn)
    end

    function ColRangeAgg_pullback(result_bar)
        Z_bar = zero(Z)
        A_bar = ChainRules.@thunk(zero(A))
        for (i,(pb, rng)) in enumerate(zip(func_pullbacks, cra.col_ranges))
            Z_bar[:,rng] .= pb(result_bar[i])[1]
        end
        return ChainRules.NoTangent(), Z_bar, A_bar
    end

    return result, ColRangeAgg_pullback

end

