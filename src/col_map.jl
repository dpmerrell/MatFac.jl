
import Base: view

mutable struct ColMap
    col_ranges::Vector{UnitRange}
    funcs::Vector # Each object in "funcs"
                  # must (1) be callable and
                  # (2) have a `view` method.
                  # In general, they may contain trainable parameters.
end


Flux.trainable(cm::ColMap) = (cm.funcs,)


function view(cm::ColMap, idx)
    new_ranges, r_min, r_max = subset_ranges(cm.col_ranges, idx)
    shifted_new_ranges = shift_range.(new_ranges, (1 - new_ranges[1].start))

    # Create views of the column functions
    newfunc_views = []
    for (n_rng, o_rng, f) in zip(new_ranges, cm.col_ranges[r_min:r_max], cm.funcs[r_min:r_max])
        sh_rng = shift_range(n_rng, (1 - o_rng.start))
        push!(newfunc_views, view(f, sh_rng))
    end

    return ColMap(shifted_new_ranges, newfunc_views)
end


function (cm::ColMap)(A::AbstractMatrix)
    
    result = zero(A)
    for (rng, f) in zip(cm.col_ranges, cm.funcs)
        result[:,rng] .= f(A[:,rng])
    end
    return result
end


function ChainRules.rrule(cm::typeof(ColMap), A)

    result = zero(A)

    function colmap_pullback(result_bar)
        return cm_bar, A_bar
    end

    return result, colmap_pullback
end



