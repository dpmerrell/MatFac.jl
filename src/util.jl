

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

    @assert is_contiguous(id_vec)

    unique_ids = unique(id_vec)
    start_idx = indexin(unique_ids, id_vec)
    end_idx = length(id_vec) .- indexin(unique_ids, reverse(id_vec)) .+ 1
    ranges = BMFRange[start:finish for (start,finish) in zip(start_idx, end_idx)]

    return ranges
end


function ids_to_idx_vecs(id_vec)

    unq_ids = unique(id_vec)
    name_to_number = Dict(name=>number for (number,name) in enumerate(unq_ids)) 
    idx_vec = Vector{Int}[Int[] for unq_id in unq_ids]

    for (i,name) in enumerate(id_vec)
        push!(idx_vec[name_to_number[name]], i)
    end

    return idx_vec
end


#function values_dict_to_vec(values_dict, row_batch_vec)
#
#    unq_ids = unique(row_batch_vec)
#    id_to_idx = Dict(id => idx for (idx, id) in enumerate(unq_ids))
#    values_vec = zeros(length(unq_ids))
#    for (k,v) in values_dict
#        values_vec[id_to_idx[k]] = v
#    end
#    return values_vec
#end



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


function subset_idx_vecs(idx_vecs::Vector{Vector{Int}}, rng::UnitRange)

    r_min = rng.start
    r_max = rng.stop

    @assert r_min < r_max

    new_vec = Vector{Int}[]
    kept_idx = Int[]
    # Loop the index vectors
    for (i, idx_vec) in enumerate(idx_vecs)
        # If the given range intersects with this 
        # index vector, then we keep a subset of it
        if (r_min <= idx_vec[end]) & (r_max >= idx_vec[1])
            start_idx = searchsorted(idx_vec, r_min).start
            stop_idx = searchsorted(idx_vec, r_max).stop

            if start_idx <= stop_idx
                push!(new_vec, idx_vec[start_idx:stop_idx])
                push!(kept_idx, i)
            end
        end
    end

    return new_vec, kept_idx
end


