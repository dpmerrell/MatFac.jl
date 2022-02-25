

import Base: size, getindex, view, map!


##########################################
# "BATCH MATRIX" DEFINITION 
##########################################
mutable struct BatchMatrix{T<:Number}
    col_batches::Vector{UnitRange}
    unq_row_batches::Vector{Vector{Int}}
    row_batch_idx::Vector{Vector{Vector{Int}}}
    values::Vector{Vector{T}}
end


function batch_matrix(values::Vector{Dict{T,U}}, 
                      row_batch_ids::Vector{Vector{T}}, 
                      col_batch_ids::Vector{V}) where T<:KeyType where U<:Number where V<:KeyType
# T : Row key type
# U : matrix "dtype"
# V : column key type

    unq_col_batches = unique(col_batch_ids)
    
    # There must be a one-to-one matching between column batches
    # and the entries of row_batch_ids. (and the entries of `values`)
    @assert length(unq_col_batches) == length(row_batch_ids)
    @assert length(unq_col_batches) == length(values)

    # All of the column blocks must have 
    # the same number of rows.
    N = length(row_batch_ids[1])
    @assert all([length(v) == N for v in row_batch_ids])

    # Translate column batch ids to ranges of indices
    col_batches = ids_to_ranges(col_batch_ids)

    # Encode the row batch ids as integers
    unq_row_batches = Vector{T}[unique(v) for v in row_batch_ids]
    row_batch_encoders = [Dict(name=>i for (i,name) in enumerate(u)) 
                                       for u in unq_row_batches]
    n_row_batches = [length(v) for v in unq_row_batches]
    enc_row_batch_ids = Vector{Int}[Int[rbe[b] for b in rbvec] 
                                    for (rbe,rbvec) in zip(row_batch_encoders, 
                                                           row_batch_ids)]
    enc_unq_row_batches = Vector{Int}[collect(1:n) for n in n_row_batches]

    # Translate the (integer) row batch ids to index vectors 
    row_batch_idx = Vector{Vector{Int}}[Vector{Int}[Int[] for _=1:n] 
                                                          for n in n_row_batches]
    for (j, rbvec) in enumerate(enc_row_batch_ids)
        for (i, rb) in enumerate(rbvec)
            push!(row_batch_idx[j][rb], i)
        end
    end

    # Populate the values 
    values_vecs = Vector{U}[U[vdict[rb] for rb in urb_vec] 
                            for (urb_vec, vdict) in zip(unq_row_batches, values)]

    return BatchMatrix(col_batches, enc_unq_row_batches, 
                       row_batch_idx, values_vecs)

end


######################################
# zero function
function Base.zero(A::BatchMatrix{T}) where T<:Number 

    copy_values = Vector{T}[zero(v) for v in A.values]
    return BatchMatrix(copy(A.col_batches), 
                       deepcopy(A.unq_row_batches),
                       deepcopy(A.row_batch_idx), 
                       copy_values)
end


######################################
# "size" operator
# WARNING: non-constant complexity!!
function size(A::BatchMatrix)
    min_row = minimum(v[1] for v in A.row_batch_idx[1])
    max_row = maximum(v[end] for v in A.row_batch_idx[1])
    return (max_row - min_row + 1,
            A.col_batches[end].stop - A.col_batches[1].start + 1)
end


######################################
# `getindex` operation
function getindex(A::BatchMatrix{T}, row_range::UnitRange, col_range::UnitRange) where T<:Number 
    
    c_min = col_range.start
    new_col_batches, c_min_idx, c_max_idx = subset_ranges(A.col_batches, col_range)
    new_col_batches = UnitRange[(rng.start - c_min + 1):(rng.stop - c_min + 1) 
                                for rng in new_col_batches]
    
    new_values = Vector{T}[] 
    new_unq_row_batches = Vector{Int}[]
    new_row_batch_idx = Vector{Vector{Int}}[]

    for cbatch=c_min_idx:c_max_idx
        old_row_batches = A.unq_row_batches[cbatch]
        old_row_batch_idx = A.row_batch_idx[cbatch]
        r_min = row_range.start - 1
        
        new_rbi, new_unq_batch_idx = subset_idx_vecs(old_row_batch_idx, row_range)

        new_rbi = [v .- r_min for v in new_rbi]
        push!(new_unq_row_batches, old_row_batches[new_unq_batch_idx])
        push!(new_row_batch_idx, new_rbi)
        push!(new_values, A.values[cbatch][new_unq_batch_idx])
    end

    return BatchMatrix(new_col_batches,
                       new_unq_row_batches,
                       new_row_batch_idx,
                       new_values)
end


#######################################
# `reindex` operation
function reindex!(A::BatchMatrix{T}, new_row_start::Integer, 
                  new_col_start::Integer) where T<:Number where U<:KeyType

    c_delta = new_col_start - A.col_batches[1].start
    A.col_batches = [(rng.start + c_delta):(rng.stop + c_delta) for rng in A.col_batches]
   
    for (i, row_dict) in enumerate(A.row_batch_dicts)
        r_min = minimum(v[1] for v in values(row_dict))
        r_delta = new_row_start - r_min
        A.row_batch_dicts[i] = Dict{U,Vector{Int}}(k => v .+ r_delta for (k,v) in row_dict)
    end

end


####################################
# Equality operator
function Base.:(==)(A::BatchMatrix, B::BatchMatrix)
    for fn in fieldnames(BatchMatrix)
        if getproperty(A, fn) != getproperty(B, fn)
            return false
        end
    end
    return true
end


####################################
# Map operations

function Base.map(f::Function, d::Dict{T,U}) where T where U<:Number
    return Dict{T,U}(k => f(v) for (k,v) in d)
end

function Base.map(f::Function, collection::BatchMatrix{T}) where T<:Number where U<:KeyType
        
    new_values = Dict{U,T}[map(f, v) for v in collection.values]
    return BatchMatrix(new_values,
                       collection.row_batch_dicts,
                       collection.col_batches)
end

function Base.map!(f::Function, dest::Dict{T,U},
                   collection::Dict{T,U}) where T where U<:Number
    for k in keys(dest)
        dest[k] = f(collection[k])
    end
end

function Base.map!(f::Function, destination::BatchMatrix, 
                                collection::BatchMatrix)
    for (d_v, c_v) in zip(destination.values, collection.values)
        map!(f, d_v, c_v) 
    end
end


##########################################
# EXPONENTIATION
##########################################
function Base.exp(A::BatchMatrix)
    return map(exp, A)
end


function ChainRules.rrule(::typeof(exp), A::BatchMatrix{U}) where U<:Number where T<:KeyType

    Z = exp(A)

    function bm_exp_pullback(Z_bar::BatchMatrix{U})
        A_bar_values = Dict{T,U}[binop(*, Z_bar_d, Z_d) for (Z_bar_d, Z_d) in zip(Z_bar.values, Z.values)]
        return ChainRules.NoTangent(), BatchMatrix(A_bar_values,
                                                   Z_bar.row_batch_dicts,
                                                   Z_bar.col_batches)
    end

    return Z, bm_exp_pullback
end


##########################################
# ADDITION 
##########################################

# Used during the model's "forward" mode
# to add batch shift to the data
function Base.:(+)(A::AbstractMatrix, B::BatchMatrix)

    result = zero(A)
    for j=1:length(B.col_batches)
        col_range = B.col_batches[j]
        for (k, row_batch_idx) in B.row_batch_dicts[j]
            result[row_batch_idx,col_range] .= A[row_batch_idx,col_range] .+ B.values[j][k]
        end
    end
    return result
end


function ChainRules.rrule(::typeof(+), A::AbstractMatrix, B::BatchMatrix)

    Z = A + B 

    function bm_add_pullback(Z_bar)
        A_bar = copy(Z_bar)
        B_bar = zero(B)

        for j=1:length(B.col_batches)
            col_range = B.col_batches[j]
            for (k,row_idx) in B.row_batch_dicts[j]
                B_bar.values[j][k] = sum(Z_bar[row_idx,col_range])
            end
        end

        return ChainRules.NoTangent(), A_bar, B_bar 
    end
    return Z, bm_add_pullback
end


"""
    add!(A::BatchMatrix, B::BatchMatrix)

    Assume B's row batch structure agrees with that of A,
    and contains only a subset of the batches contained
    in A (most commonly, B = A[rows,:] results from 
    a `getindex` operation).
"""
function add!(A::BatchMatrix, B::BatchMatrix)

    for (j, (A_rb_dict, B_rb_dict)) in enumerate(zip(A.row_batch_dicts,
                                                     B.row_batch_dicts))
        for (k, b_batch_idx) in B_rb_dict

            overlap = length(b_batch_idx)/length(A_rb_dict[k])
            A.values[j][k] += (overlap * B.values[j][k])
        end
    end
end

#function add!(A::BatchMatrix, B::BatchMatrix)
#
#    for (j, (A_row_ranges, B_row_ranges)) in enumerate(zip(A.row_ranges_vec,
#                                                           B.row_ranges_vec))
#
#        # Locate the row index of the first A.value touched by B
#        A_starts = Int[rng.start for rng in A_row_ranges]
#        A_stops =  Int[rng.stop for rng in A_row_ranges]
#        B_starts = Int[rng.start for rng in B_row_ranges]
#        B_stops =  Int[rng.stop for rng in B_row_ranges]
#  
#        # Initialize the A_idx and B_idx 
#        if A_starts[1] == B_starts[1]
#            A_idx = 1
#            B_idx = 1
#        elseif A_starts[1] < B_starts[1]
#            B_idx = 1
#            A_idx = searchsorted(A_starts, B_starts[1]).stop
#        else
#            A_idx = 1
#            B_idx = searchsorted(B_starts, A_starts[1]).stop
#        end
#
#        # Iterate through the subsequent A_idxs, B_idxs
#        while (A_idx <= length(A_starts)) & (B_idx <= length(B_starts)) 
#
#            # Compute the fraction of current A covered by
#            # current B
#            overlap = ( (min(A_stops[A_idx], B_stops[B_idx]) 
#                        - max(A_starts[A_idx], B_starts[B_idx]) + 1)
#                        / (A_stops[A_idx] - A_starts[A_idx] + 1)
#                       )
#
#            # Update the current A value by the current B value
#            A.values[j][A_idx] += (overlap * B.values[j][B_idx])
#
#            # Update the A_idx or B_idx
#            if A_stops[A_idx] == B_stops[B_idx]
#                A_idx += 1
#                B_idx += 1
#            elseif A_stops[A_idx] < B_stops[B_idx]
#                A_idx += 1
#            else
#                B_idx += 1
#            end
#        end
#
#    end
#end


# Update A by adding B to the given rows.
# Each batch of A is updated by the overlapping batches of B,
# weighted by the fraction covered by each of them.
# Used during "fit" to update the model's batch parameters. 
function row_add!(A::BatchMatrix, A_rows::UnitRange, B::BatchMatrix)

    #@assert size(B)[1] == (A_rows.stop - A_rows.start + 1)

    reindex!(B, A_rows.start, 1)
    add!(A, B)
    reindex!(B, 1, 1)
end


##########################################
# BatchMatrix Multiplication 
##########################################

# Used during the model's forward mode
# to add batch scaling to the data
function Base.:(*)(A::AbstractMatrix, B::BatchMatrix)

    result = zero(A)
    for j=1:length(B.col_batches)
        col_range = B.col_batches[j]
        for (k, row_batch_idx) in B.row_batch_dicts[j] 
            result[row_batch_idx,col_range] .= A[row_batch_idx,col_range] .* B.values[j][k]
        end
    end
    return result
end


function ChainRules.rrule(::typeof(*), A::AbstractMatrix, B::BatchMatrix)

    Z = A * B

    function bm_mult_pullback(Z_bar)
        A_bar = zero(A)
        B_bar = zero(B)
        for j=1:length(B.col_batches)
            col_range = B.col_batches[j]
            for (k, row_batch_idx) in B.row_batch_dicts[j]
                A_bar[row_batch_idx,col_range] .= (Z_bar[row_batch_idx,col_range] .* B.values[j][k])
                B_bar.values[j][k] = sum(Z_bar[row_batch_idx,col_range] .* A[row_batch_idx,col_range])
            end
        end

        return ChainRules.NoTangent(), A_bar, B_bar 
    end

    return Z, bm_mult_pullback
end

