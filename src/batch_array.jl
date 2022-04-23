
import Base: view, zero, exp

mutable struct BatchArray
    col_ranges::Tuple # UnitRanges
    row_idx::UnitRange # Stores "view-like" information
    row_batches::Tuple # boolean matrices;
                       # each column is an indicator
                       # vector for a batch. Should
                       # balance space-efficiency and performance
    values::Tuple # vectors of numbers
end

# The values should not be moved to GPU
# (they need to be scalar-indexable)
@functor BatchArray (col_ranges, row_idx, row_batches)


function BatchArray(col_batch_ids::Vector, row_batch_ids, 
                    value_dicts::Vector{<:AbstractDict})

    n_rows = length(row_batch_ids[1])
    
    col_ranges = ids_to_ranges(col_batch_ids)
    row_batches = [ids_to_ind_mat(rbv) for rbv in row_batch_ids]
    values = [[vd[ub] for ub in unique(rbi)] for (vd,rbi) in zip(value_dicts,row_batch_ids)]

    return BatchArray(Tuple(col_ranges), 
                      1:n_rows,
                      Tuple(row_batches), 
                      Tuple(values))
end


function view(ba::BatchArray, idx1, idx2::UnitRange)

    new_col_ranges, r_min, r_max = subset_ranges(ba.col_ranges, idx2)
    new_col_ranges = shift_range.(new_col_ranges, 1 - new_col_ranges[1].start)

    new_row_idx = ba.row_idx[idx1]

    new_row_batches = ba.row_batches[r_min:r_max]
    new_values = ba.values[r_min:r_max]

    return BatchArray(new_col_ranges, new_row_idx, Tuple(new_row_batches), new_values)
end


function zero(ba::BatchArray)
    cvalues = [zero(v) for v in ba.values]
    return BatchArray(deepcopy(ba.col_ranges),
                      deepcopy(ba.row_idx),
                      deepcopy(ba.row_batches), 
                      Tuple(cvalues)) 
end


#########################################
# Addition
function Base.:(+)(A::AbstractMatrix, B::BatchArray)

    result = copy(A)
    for (j,cbr) in enumerate(B.col_ranges)
        for i=1:length(B.values[j])
            result[:,cbr] .+= (B.row_batches[j][B.row_idx,i].*B.values[j][i])
        end
    end
    return result
end


function ChainRules.rrule(::typeof(+), A::AbstractMatrix, B::BatchArray)
    
    result = A + B
    
    function ba_plus_pullback(result_bar)
        A_bar = copy(result_bar) # Just a copy of the result tangent 
        
        B_bar = zero(B) # Just sum the result tangents corresponding
                        # to each value of B
        for (j, cbr) in enumerate(B_bar.col_ranges)
            for i=1:length(B_bar.values[j])
                B_bar.values[j][i] = sum(result_bar[B_bar.row_batches[j][B.row_idx,i], cbr])
            end
        end
        return ChainRulesCore.NoTangent(), A_bar, B_bar 
    end

    return result, ba_plus_pullback
end


#########################################
# Multiplication
function Base.:(*)(A::AbstractMatrix, B::BatchArray)

    result = copy(A)
    for (j,cbr) in enumerate(B.col_ranges)
        for i=1:length(B.values[j])
            result[B.row_batches[j][B.row_idx,i], cbr] .*= B.values[j][i]
        end
    end
    return result
end


function ChainRules.rrule(::typeof(*), A::AbstractMatrix, B::BatchArray)
    
    result = A * B
    
    function ba_mult_pullback(result_bar)
        A_bar = (result_bar .+ zero(A)) * B # result tangent multiplied by B
        B_bar = zero(B) # Just sum the (result_tangents.*A_entries) corresponding
                        # to each value of B

        for (j, cbr) in enumerate(B_bar.col_ranges)
            for i=1:length(B_bar.values[j])
                rbatch = B_bar.row_batches[j][B.row_idx,i]
                B_bar.values[j][i] = sum(result_bar[rbatch,cbr] .* A[rbatch,cbr])
            end
        end

        return ChainRulesCore.NoTangent(), A_bar, B_bar 
    end

    return result, ba_mult_pullback
end


#########################################
# Exponentiation
function exp(ba::BatchArray)
    return BatchArray(deepcopy(ba.col_ranges),
                      deepcopy(ba.row_idx),
                      deepcopy(ba.row_batches),
                      map(v->exp.(v), ba.values))
end


function ChainRules.rrule(::typeof(exp), ba::BatchArray)
    
    Z = exp(ba)

    function ba_exp_pullback(Z_bar)
        ba_bar = BatchArray(deepcopy(ba.col_ranges),
                            deepcopy(ba.row_idx),
                            deepcopy(ba.row_batches),
                            deepcopy(Z.values))
        ba_bar.values = map(.*, ba_bar.values, Z_bar.values)

        return ChainRulesCore.NoTangent(), ba_bar
    end

    return Z, ba_exp_pullback
end


