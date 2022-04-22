

import Base: iterate

mutable struct BatchIter
    N::Int
    batch_size::Int
end


function iterate(bi::BatchIter)

    l_idx = 1
    r_idx = min(bi.N, l_idx + bi.batch_size - 1)

    if r_idx < l_idx
        return nothing
    end

    return l_idx:r_idx, r_idx
end


function iterate(bi::BatchIter, prev_r_idx)

    l_idx = prev_r_idx + 1
    
    r_idx = min(bi.N, l_idx + bi.batch_size - 1)

    if r_idx < l_idx
        return nothing
    end

    return l_idx:r_idx, r_idx

end


