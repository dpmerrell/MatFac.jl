

import Base: iterate, length

mutable struct BatchIter
    N::Int
    batch_size::Int
end


function iterate(bi::BatchIter)

    l_indices = collect(1:bi.batch_size:bi.N)
    r_indices = vcat(l_indices[2:end] .- 1, bi.N)
    ranges = [l:r for (l, r) in zip(l_indices, r_indices)]
    rng = popfirst!(ranges)
    
    return rng, ranges
end


function iterate(bi::BatchIter, rem_ranges)

    if length(rem_ranges) == 0
        return nothing
    end
    rng = popfirst!(rem_ranges)
    
    return rng, rem_ranges
end


function length(bi::BatchIter)
    n_batch = div(bi.N, bi.batch_size)
    if bi.batch_size*n_batch < bi.N
        n_batch += 1
    end
    return n_batch
end


function batch_iterations(N, batch_size)
    l_indices = collect(1:batch_size:N)
    r_indices = vcat(l_indices[2:end] .- 1, N)
    ranges = [l:r for (l, r) in zip(l_indices, r_indices)]
    return ranges 
end
