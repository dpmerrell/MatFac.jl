



##########################################
# "BATCH QUANTITY" DEFINITION 
##########################################
mutable struct BatchQuantity
    values::AbstractMatrix
    row_ranges::AbstractVector{AbstractRange}
    col_ranges::AbstractVector{AbstractRange}
end


function BatchQuantity(values, row_batches::Vector{Int}, col_batches::Vector{Int})
   
    row_ranges = ids_to_ranges(row_batches)
    col_ranges = ids_to_ranges(col_batches)
    
    return BatchQuantity(values, row_ranges, col_ranges)

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


function ChainRules.rrule(::typeof(bq_add), A, B)

    Z = bq_add(A,B)

    function bq_add_pullback(Z_bar)
        A_bar = Z_bar
        B_bar = zero(B.values)
        for (i,row_range) in enumerate(B.row_ranges)
            for (j,col_range) in enumerate(B.col_ranges)
                B_bar[i,j] = sum(Z_bar[row_range,col_range])
            end
        end

        return ChainRules.NoTangent(), A_bar, B_bar
    end
    return Z, bq_add_pullback
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


function ChainRules.rrule(::typeof(bq_mult), A, B)

    Z = bq_add(A,B)

    function bq_add_pullback(Z_bar)
        A_bar = copy(Z_bar)
        B_bar = zero(B.values)
        for (i,row_range) in enumerate(B.row_ranges)
            for (j,col_range) in enumerate(B.col_ranges)
                A_bar[row_range,col_range] .*= B.values[i,j]
                B_bar[i,j] = sum(Z_bar[row_range,col_range] .* A[row_range,col_range])
            end
        end

        return ChainRules.NoTangent(), A_bar, B_bar
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




#row_batches = repeat(1:5, inner=(2,))
#col_batches = repeat(1:2, inner=(5,))
#values = reshape(collect(1:10), (5,2))
#
#my_bq = BatchQuantity(values,row_batches,col_batches)
#println("INITIALIZED BQ")
