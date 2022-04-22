

"""
Check that the values in `vec` occur in contiguous blocks.
I.e., the unique values are grouped together, with no intermingling.
I.e., for each unique value the set of indices mapping to that value
occur consecutively.
"""
function is_contiguous(vec::AbstractVector{T}) where T

    past_values = Set{T}()
    
    for i=1:(length(vec)-1)
        next_value = vec[i+1]
        if in(vec[i+1], past_values)
            return false
        end
        if vec[i+1] != vec[i]
            push!(past_values, vec[i])
        end
    end

    return true
end


function ids_to_ranges(id_vec)

    @assert is_contiguous(id_vec) "IDs in id_vec need to appear in contiguous chunks."

    unique_ids = unique(id_vec)
    start_idx = indexin(unique_ids, id_vec)
    end_idx = length(id_vec) .- indexin(unique_ids, reverse(id_vec)) .+ 1
    ranges = UnitRange[start:finish for (start,finish) in zip(start_idx, end_idx)]

    return ranges
end


function ids_to_ind_mat(id_vec)

    unq_ids = unique(id_vec)
    ind_mat = zeros(Bool, length(id_vec), length(unq_ids))

    for (i,name) in enumerate(unq_ids)
        ind_mat[:,i] .= (id_vec .== name)
    end

    return ind_mat
end


function subset_ranges(ranges::Vector, rng::UnitRange) 
    
    r_min = rng.start
    r_max = rng.stop
    @assert r_min <= r_max

    @assert r_min >= ranges[1].start
    @assert r_max <= ranges[end].stop

    starts = [rr.start for rr in ranges]
    r_min_idx = searchsorted(starts, r_min).stop
    
    stops = [rr.stop for rr in ranges]
    r_max_idx = searchsorted(stops, r_max).start

    new_ranges = ranges[r_min_idx:r_max_idx]
    new_ranges[1] = r_min:new_ranges[1].stop
    new_ranges[end] = new_ranges[end].start:r_max

    return new_ranges, r_min_idx, r_max_idx
end

function shift_range(rng, delta)
    return (rng.start + delta):(rng.stop + delta) 
end




