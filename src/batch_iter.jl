

import Base: iterate

mutable struct BatchIter
    N::Int
    batch_size::Int
end


function iterate(bi::BatchIter)

    l_indices = collect(1:bi.batch_size:bi.N)
    r_indices = vcat(l_indices[2:end] .- 1, bi.N)
    ranges = [l:r for (l, r) in zip(l_indices, r_indices)]
    shuffle!(ranges)
    
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


