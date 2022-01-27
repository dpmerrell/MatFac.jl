

import Base: size, getindex, view, transpose, map!


##########################################
# "BLOCK MATRIX" DEFINITION 
##########################################
mutable struct BlockMatrix
    values::AbstractMatrix
    row_ranges::Vector{UnitRange}
    col_ranges::Vector{UnitRange}
end


function ChainRules.rrule(::typeof(BlockMatrix), values::AbstractMatrix,
                                                 row_ranges::Vector{UnitRange},
                                                 col_ranges::Vector{UnitRange})
    function BlockMatrix_pullback(bm_bar)
        return ChainRules.NoTangent(), bm_bar, ChainRules.NoTangent(), ChainRules.NoTangent()
    end

    return BlockMatrix(values, row_ranges, col_ranges), BlockMatrix_pullback

end


function block_matrix(values::AbstractMatrix, row_block_ids::Vector, col_block_ids::Vector)
   
    row_ranges = ids_to_ranges(row_block_ids)
    col_ranges = ids_to_ranges(col_block_ids)
    
    return BlockMatrix(copy(values), row_ranges, col_ranges)

end


######################################
# zero function
function Base.zero(A::BlockMatrix)
    return BlockMatrix(zero(A.values), copy(A.row_ranges),
                                       copy(A.col_ranges))
end


######################################
# "size" operator
function size(A::BlockMatrix)
    return (A.row_ranges[end].stop - A.row_ranges[1].start + 1,
            A.col_ranges[end].stop - A.col_ranges[1].start + 1)
end


######################################
# "getindex" operator
function getindex(A::BlockMatrix, row_range::UnitRange, col_range::UnitRange)

    r_min = row_range.start
    new_row_ranges, r_min_idx, r_max_idx = subset_ranges(A.row_ranges, row_range)
    new_row_ranges = [(rng.start - r_min + 1):(rng.stop - r_min + 1) for rng in new_row_ranges]
    
    c_min = col_range.start
    new_col_ranges, c_min_idx, c_max_idx = subset_ranges(A.col_ranges, col_range)
    new_col_ranges = [(rng.start - c_min + 1):(rng.stop - c_min + 1) for rng in new_col_ranges]

    return BlockMatrix(A.values[r_min_idx:r_max_idx, c_min_idx:c_max_idx],
                       new_row_ranges, new_col_ranges)
end

######################################
# view operator
function view(A::BlockMatrix, row_range::UnitRange, col_range::UnitRange)

    r_min = row_range.start
    new_row_ranges, r_min_idx, r_max_idx = subset_ranges(A.row_ranges, row_range)
    new_row_ranges = [(rng.start - r_min + 1):(rng.stop - r_min + 1) for rng in new_row_ranges]
    
    c_min = col_range.start
    new_col_ranges, c_min_idx, c_max_idx = subset_ranges(A.col_ranges, col_range)
    new_col_ranges = [(rng.start - c_min + 1):(rng.stop - c_min + 1) for rng in new_col_ranges]

    return BlockMatrix(view(A.values, r_min_idx:r_max_idx, c_min_idx:c_max_idx),
                       new_row_ranges, new_col_ranges)
end


#######################################
# reindex operation
function reindex!(A::BlockMatrix, new_row_start::Integer, new_col_start::Integer)

    r_delta = new_row_start - A.row_ranges[1].start
    c_delta = new_col_start - A.col_ranges[1].start
    A.row_ranges = [(r.start + r_delta):(r.stop + r_delta) for r in A.row_ranges]
    A.col_ranges = [(r.start + r_delta):(r.stop + r_delta) for r in A.col_ranges]

end


####################################
# transpose operation
function transpose(A::BlockMatrix)
    return BlockMatrix(transpose(A.values), A.col_ranges, A.row_ranges)
end


####################################
# Equality operator
function Base.:(==)(A::BlockMatrix, B::BlockMatrix)
    return ((A.values == B.values) & 
            (A.row_ranges == B.row_ranges) & 
            (A.col_ranges == B.col_ranges))
end


####################################
# Map mutator operation
function Base.map!(f::Function, destination::BlockMatrix, 
                                collection::BlockMatrix)
    map!(f, destination.values, collection.values) 
end


##########################################
# ADDITION 
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

        return ChainRules.NoTangent(), A_bar, B_bar
    end
    return Z, bm_add_pullback
end


function add!(A::BlockMatrix, B::BlockMatrix)

    # Locate the row index of the first A.value touched by B
    A_starts = Int[rng.start for rng in A.row_ranges]
    A_stops =  Int[rng.stop for rng in A.row_ranges]
    B_starts = Int[rng.start for rng in B.row_ranges]
    B_stops =  Int[rng.stop for rng in B.row_ranges]
  
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


# Update A by adding B to the given rows.
# Each block of A is updated by the overlapping blocks of B,
# weighted by the fraction covered by each of them.
# Used during "fit" to update the model's block parameters. 
function row_add!(A::BlockMatrix, A_rows::UnitRange, B::BlockMatrix)

    @assert size(B)[1] == (A_rows.stop - A_rows.start + 1)

    reindex!(B, A_rows.start, 1)
    add!(A, B)
    reindex!(B, 1, 1)
end


function col_add!(A::BlockMatrix, A_cols::UnitRange, B::BlockMatrix)
    row_add!(transpose(A), A_cols, transpose(B))
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

        return ChainRules.NoTangent(), A_bar, B_bar 
    end

    return Z, bm_mult_pullback
end

