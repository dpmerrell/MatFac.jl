

import Base: iterate

mutable struct DataLoader
    data::Vector{AbstractMatrix}
    batch_rows::Int
end


function DataLoader(data::Vector{AbstractMatrix}; batch_rows=1000)

    n_rows = size(data[1],1)
    for array in data
        @assert size(array,1) == n_rows
    end

    return DataLoader(data, batch_rows)
end


function iterate(dl::DataLoader)

    l_idx = 1
    n_rows = size(dl.data[1],1)
    r_idx = min(n_rows, l_idx + dl.batch_rows - 1)

    if r_idx < l_idx
        return nothing
    end

    return [view(arr, l_idx:r_idx, :) for arr in dl.data], r_idx
end


function iterate(dl::DataLoader, prev_r_idx)

    l_idx = prev_r_idx + 1
    
    n_rows = size(dl.data[1],1)
    r_idx = min(n_rows, l_idx + dl.batch_rows - 1)

    if r_idx < l_idx
        return nothing
    end

    return [view(arr, l_idx:r_idx, :) for arr in dl.data], r_idx

end


