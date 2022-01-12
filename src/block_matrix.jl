

import Base: size, getindex, setindex!, view


##########################################
# "BLOCK MATRIX" DEFINITION 
##########################################
mutable struct BlockMatrix
    values::AbstractMatrix
    row_ranges::Vector{UnitRange}
    col_ranges::Vector{UnitRange}
end


function block_matrix(values::AbstractMatrix, row_block_ids::Vector, col_block_ids::Vector)
   
    row_ranges = ids_to_ranges(row_block_ids)
    col_ranges = ids_to_ranges(col_block_ids)
    
    return BlockMatrix(copy(values), row_ranges, col_ranges)

end


######################################
# "size" operator
function size(A::BlockMatrix)
    return (A.row_ranges[end].stop - A.row_ranges[1].start + 1,
            A.col_ranges[end].stop - A.col_ranges[1].start + 1)
end


######################################
# "getindex" operator
# For now, we only select rows.
# All columns are included.
function getindex(A::BlockMatrix, row_range::UnitRange)

    r_min = row_range.start
    r_max = row_range.stop
    @assert r_min <= r_max

    @assert r_min >= A.row_ranges[1].start
    @assert r_max <= A.row_ranges[end].stop

    row_starts = [rr.start for rr in A.row_ranges]
    r_min_idx = searchsorted(row_starts, r_min).stop
    
    row_stops = [rr.stop for rr in A.row_ranges]
    r_max_idx = searchsorted(row_stops, r_max).start

    new_row_ranges = A.row_ranges[r_min_idx:r_max_idx]
    new_row_ranges[1] = r_min:new_row_ranges[1].stop
    new_row_ranges[end] = new_row_ranges[end].start:r_max

    new_row_ranges = [(rng.start - r_min + 1):(rng.stop - r_min + 1) for rng in new_row_ranges]

    return BlockMatrix(A.values[r_min_idx:r_max_idx,:],
                         new_row_ranges,
                         A.col_ranges)
end


###############################
## "setindex" operator
#function setindex!(A::BlockMatrix, B::BlockMatrix, I::UnitRange)
#    
#    @assert size(B) == (I.stop - I.start + 1)
#    
#    # Locate the index of the first A.value touched by B
#    A_starts = Int[rng.start for rng in A.row_ranges]
#    A_stops = Int[rng.stop for rng in A.row_ranges]
#    B_starts = Int[rng.start for rng in B.row_ranges]
#    B_stops = Int[rng.stop for rng in B.row_ranges]
#  
#    # Initialize the A_idx and B_idx 
#    if A_starts[1] == B_starts[1]
#        A_idx = 1
#        B_idx = 1
#    elseif A_starts[1] < B_starts[1]
#        B_idx = 1
#        A_idx = searchsorted(A_starts, B_starts[1]).stop
#    else
#        A_idx = 1
#        B_idx = searchsorted(B_starts, A_starts[1]).stop
#    end
#
#    # Iterate through the subsequent A_idxs, B_idxs
#    while (A_idx <= length(A_starts)) & (B_idx <= length(B_starts)) 
#
#        # Compute the fraction of current A covered by
#        # current B
#        overlap = (min(A_stops[A_idx], B_stops[B_idx]) - max(A_starts[A_idx], B_starts[B_idx]) + 1)/(A_stops[A_idx] - A_starts[A_idx] + 1)
#
#        # Update the current A value by the current B value
#        A.values[A_idx,:] .+= (overlap .* B.values[B_idx,:])
#
#        # Update the A_idx or B_idx
#        if A_stops[A_idx] == B_stops[B_idx]
#            A_idx += 1
#            B_idx += 1
#        elseif A_stops[A_idx] < B_stops[B_idx]
#            A_idx += 1
#        else
#            B_idx += 1
#        end
#    end
#
#    
#end

#########################
# Equality operator
function Base.:(==)(A::BlockMatrix, B::BlockMatrix)
    return (A.values == B.values) & (A.row_ranges == B.row_ranges) & (A.col_ranges == B.col_ranges)
end


##########################################
# "BATCH QUANTITY" ADDITION 
##########################################

# Used during the model's "forward" mode
# to add block shift to the data
function Base.:(+)(A::AbstractMatrix, B::BlockMatrix)

    @assert size(A) == size(B)

    result = zero(A)
    for (i, row_range) in enumerate(B.row_ranges) 
        for (j, col_range) in enumerate(B.col_ranges)
            result[row_range,col_range] .= A[row_range,col_range] .+ B.values[i,j]
        end
    end
    return result
end


function ChainRules.rrule(::typeof(+), A::AbstractMatrix, B::BlockMatrix)

    Z = A + B 

    function bm_add_pullback(Z_bar)
        A_bar = Z_bar
        B_bar = zero(B.values)
        for (i,row_range) in enumerate(B.row_ranges)
            for (j,col_range) in enumerate(B.col_ranges)
                B_bar[i,j] = sum(Z_bar[row_range,col_range])
            end
        end

        return ChainRules.NoTangent(), A_bar, ChainRules.Tangent{BlockMatrix}(;values=B_bar, 
                                                                               row_ranges=copy(B.row_ranges), 
                                                                               col_ranges=copy(B.col_ranges))
    end
    return Z, bm_add_pullback
end


# Update A by adding B to the given rows.
# Each block of A is updated by the overlapping blocks of B,
# weighted by the fraction covered by each of them.
# Used during "fit" to update the model's block parameters. 
function add!(A::BlockMatrix, A_rows::UnitRange, B::BlockMatrix)

    @assert size(B)[1] == (A_rows.stop - A_rows.start + 1)

    # Locate the index of the first A.value touched by B
    A_starts = Int[rng.start for rng in A.row_ranges]
    A_stops =  Int[rng.stop for rng in A.row_ranges]
    B_starts = Int[rng.start + A_rows.start - 1 for rng in B.row_ranges]
    B_stops =  Int[rng.stop + A_rows.start - 1 for rng in B.row_ranges]
  
    # Initialize the A_idx and B_idx 
    if A_starts[1] == B_starts[1]
        A_idx = 1
        B_idx = 1
    elseif A_starts[1] < B_starts[1]
        B_idx = 1
        A_idx = searchsorted(A_starts, B_starts[1]).stop
    else
        A_idx = 1
        B_idx = searchsorted(B_starts, A_starts[1]).stop
    end

    # Iterate through the subsequent A_idxs, B_idxs
    while (A_idx <= length(A_starts)) & (B_idx <= length(B_starts)) 

        # Compute the fraction of current A covered by
        # current B
        overlap = ( (min(A_stops[A_idx], B_stops[B_idx]) 
                    - max(A_starts[A_idx], B_starts[B_idx]) + 1)
                    / (A_stops[A_idx] - A_starts[A_idx] + 1)
                   )

        # Update the current A value by the current B value
        A.values[A_idx,:] .+= (overlap .* B.values[B_idx,:])

        # Update the A_idx or B_idx
        if A_stops[A_idx] == B_stops[B_idx]
            A_idx += 1
            B_idx += 1
        elseif A_stops[A_idx] < B_stops[B_idx]
            A_idx += 1
        else
            B_idx += 1
        end
    end

end


##########################################
# "BATCH QUANTITY" MULTIPLICATION 
##########################################

# Used during the model's forward mode
# to add block scaling to the data
function Base.:(*)(A::AbstractMatrix, B::BlockMatrix)

    result = zero(A)
    for (i, row_range) in enumerate(B.row_ranges) 
        for (j, col_range) in enumerate(B.col_ranges)
            result[row_range,col_range] .= A[row_range,col_range] .* B.values[i,j]
        end
    end
    return result
end


function ChainRules.rrule(::typeof(*), A::AbstractMatrix, B::BlockMatrix)

    Z = A * B

    function bm_mult_pullback(Z_bar)
        A_bar = zero(A)
        B_bar = zero(B.values)
        for (i,row_range) in enumerate(B.row_ranges)
            for (j,col_range) in enumerate(B.col_ranges)
                A_bar[row_range,col_range] .= (Z_bar[row_range,col_range] .* B.values[i,j])
                B_bar[i,j] = sum(Z_bar[row_range,col_range] .* A[row_range,col_range])
            end
        end

        return ChainRules.NoTangent(), A_bar, ChainRules.Tangent{BlockMatrix}(;values=B_bar, 
                                                                               row_ranges=copy(B.row_ranges), 
                                                                               col_ranges=copy(B.col_ranges))
    end
    return Z, bm_mult_pullback
end

