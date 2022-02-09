

import Base: size, getindex, map!


##########################################
# "BLOCK MATRIX" DEFINITION 
##########################################
mutable struct BlockMatrix
    values::Vector{Vector}
    row_ranges_vec::Vector{Vector{UnitRange}}
    col_ranges::Vector{UnitRange}
end


function block_matrix(values::Vector{Vector{T}} where T, 
                      row_block_dict::Dict{T, Vector{U}} where T where U, 
                      col_block_ids::Vector)
  
    unq_col_blocks = unique(col_block_ids)

    # There must be a one-to-one matching between column blocks
    # and the keys of row_block_dict.
    @assert Set(unq_col_blocks) == Set(keys(row_block_dict))
  
    # All of the column blocks must have 
    # the same number of rows.
    N = length(row_block_dict[unq_col_blocks[1]])
    #@assert all([length(v) for v in values(row_block_dict)] .== N) 

    col_ranges = ids_to_ranges(col_block_ids)
    row_ranges_vec = Vector{UnitRange}[ids_to_ranges(row_block_dict[cblock]) for cblock in unq_col_blocks] 
   
    return BlockMatrix(deepcopy(values), row_ranges_vec, col_ranges)

end


######################################
# zero function
function Base.zero(A::BlockMatrix)

    copy_values = Vector[zero(v) for v in A.values]
    return BlockMatrix(copy_values, deepcopy(A.row_ranges_vec),
                                    copy(A.col_ranges))
end


######################################
# "size" operator
function size(A::BlockMatrix)
        return (A.row_ranges_vec[1][end].stop - A.row_ranges_vec[1][1].start + 1,
                A.col_ranges[end].stop - A.col_ranges[1].start + 1)
end


######################################
# "getindex" operator
function getindex(A::BlockMatrix, row_range::UnitRange, col_range::UnitRange)
    
    c_min = col_range.start
    new_col_ranges, c_min_idx, c_max_idx = subset_ranges(A.col_ranges, col_range)
    new_col_ranges = [(rng.start - c_min + 1):(rng.stop - c_min + 1) for rng in new_col_ranges]
    new_values = A.values[c_min_idx:c_max_idx]

    new_row_ranges_vec = Vector[]
    for (i, old_row_ranges) in enumerate(A.row_ranges_vec)
        r_min = row_range.start
        new_row_ranges, r_min_idx, r_max_idx = subset_ranges(old_row_ranges, row_range)
        new_row_ranges = [(rng.start - r_min + 1):(rng.stop - r_min + 1) for rng in new_row_ranges]

        new_values[i] = new_values[i][r_min_idx:r_max_idx]
        push!(new_row_ranges_vec, new_row_ranges)
    end

    return BlockMatrix(new_values, new_row_ranges_vec, new_col_ranges)
end


#######################################
# reindex operation
function reindex!(A::BlockMatrix, new_row_start::Integer, new_col_start::Integer)

    c_delta = new_col_start - A.col_ranges[1].start
    A.col_ranges = [(rng.start + c_delta):(rng.stop + c_delta) for rng in A.col_ranges]
   
    for (i, row_ranges) in enumerate(A.row_ranges_vec)
        r_delta = new_row_start - row_ranges[1].start
        A.row_ranges_vec[i] = [(rng.start + r_delta):(rng.stop + r_delta) for rng in row_ranges]
    end

end



####################################
# Equality operator
function Base.:(==)(A::BlockMatrix, B::BlockMatrix)
    return ((A.values == B.values) & 
            (A.row_ranges_vec == B.row_ranges_vec) & 
            (A.col_ranges == B.col_ranges))
end


####################################
# Map operations
function Base.map(f::Function, collection::BlockMatrix)
        
    new_values = Vector[map(f, v) for v in collection.values]
    return BlockMatrix(new_values,
                       collection.row_ranges_vec,
                       collection.col_ranges)
end


function Base.map!(f::Function, destination::BlockMatrix, 
                                collection::BlockMatrix)
    for (d_v, c_v) in zip(destination.values, collection.values)
        map!(f, d_v, c_v) 
    end
end


##########################################
# EXPONENTIATION
##########################################
function Base.exp(A::BlockMatrix)
    return map(exp, A)
end

function ChainRules.rrule(::typeof(exp), A::BlockMatrix)

    Z = exp(A)

    function bm_exp_pullback(Z_bar::BlockMatrix)
        A_bar_values = Vector[Z_bar_v .* Z_v for (Z_bar_v, Z_v) in zip(Z_bar.values, Z.values)]
        return ChainRules.NoTangent(), BlockMatrix(A_bar_values,
                                                   Z_bar.row_ranges_vec,
                                                   Z_bar.col_ranges)
    end

    return Z, bm_exp_pullback
end


##########################################
# ADDITION 
##########################################

# Used during the model's "forward" mode
# to add block shift to the data
function Base.:(+)(A::AbstractMatrix, B::BlockMatrix)

    @assert size(A) == size(B)

    result = zero(A)
    for (j, col_range) in enumerate(B.col_ranges)
        for (i, row_range) in enumerate(B.row_ranges_vec[j]) 
            result[row_range,col_range] .= A[row_range,col_range] .+ B.values[j][i]
        end
    end
    return result
end


function ChainRules.rrule(::typeof(+), A::AbstractMatrix, B::BlockMatrix)

    Z = A + B 

    function bm_add_pullback(Z_bar)
        A_bar = Z_bar
        B_bar = zero(B)
        for (j,col_range) in enumerate(B.col_ranges)
            for (i,row_range) in enumerate(B.row_ranges_vec[j])
                B_bar.values[j][i] = sum(Z_bar[row_range,col_range])
            end
        end

        return ChainRules.NoTangent(), A_bar, B_bar 
    end
    return Z, bm_add_pullback
end


function add!(A::BlockMatrix, B::BlockMatrix)

    for (j, (A_row_ranges, B_row_ranges)) in enumerate(zip(A.row_ranges_vec,
                                                           B.row_ranges_vec))

        # Locate the row index of the first A.value touched by B
        A_starts = Int[rng.start for rng in A_row_ranges]
        A_stops =  Int[rng.stop for rng in A_row_ranges]
        B_starts = Int[rng.start for rng in B_row_ranges]
        B_stops =  Int[rng.stop for rng in B_row_ranges]
  
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
            A.values[j][A_idx] += (overlap * B.values[j][B_idx])

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


##########################################
# "BATCH QUANTITY" MULTIPLICATION 
##########################################

# Used during the model's forward mode
# to add block scaling to the data
function Base.:(*)(A::AbstractMatrix, B::BlockMatrix)

    result = zero(A)
    for (j, col_range) in enumerate(B.col_ranges)
        for (i, row_range) in enumerate(B.row_ranges_vec[j]) 
            result[row_range,col_range] .= A[row_range,col_range] .* B.values[j][i]
        end
    end
    return result
end


function ChainRules.rrule(::typeof(*), A::AbstractMatrix, B::BlockMatrix)

    Z = A * B

    function bm_mult_pullback(Z_bar)
        A_bar = zero(A)
        B_bar = zero(B)
        for (j,col_range) in enumerate(B.col_ranges)
            for (i,row_range) in enumerate(B.row_ranges_vec[j])
                A_bar[row_range,col_range] .= (Z_bar[row_range,col_range] .* B.values[j][i])
                B_bar.values[j][i] = sum(Z_bar[row_range,col_range] .* A[row_range,col_range])
            end
        end

        return ChainRules.NoTangent(), A_bar, B_bar 
    end

    return Z, bm_mult_pullback
end

