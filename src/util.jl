

function ids_to_ranges(id_vec)

    @assert issorted(id_vec)

    unique_ids = unique(id_vec)
    start_idx = indexin(unique_ids, id_vec)
    end_idx = length(id_vec) .- indexin(unique_ids, reverse(id_vec)) .+ 1
    ranges = [start:finish for (start,finish) in zip(start_idx, end_idx)]

    return ranges
end


#a = [1,1,3,1,2,2,2,1,2,2,3,3,3,2,3,3,3,1,3,3,3]
#
#ranges = ids_to_ranges(a)
#
#println("IDS: ")
#println(a)
#
#
#println("RANGES:")
#println(ranges)
