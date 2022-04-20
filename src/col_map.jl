
import Base: view

mutable struct ColMap
    col_ranges::Tuple
    funcs::Tuple # Each object in "funcs"
                  # must (1) be callable and
                  # (2) have a `view` method.
                  # In general, they may contain trainable parameters.
end


function view(cm::ColMap, idx)
    new_ranges, r_min, r_max = subset_ranges(cm.col_ranges, idx)
    shifted_new_ranges = shift_range.(new_ranges, (1 - new_ranges[1].start))

    # Create views of the column functions
    newfunc_views = []
    for (n_rng, o_rng, f) in zip(new_ranges, cm.col_ranges[r_min:r_max], cm.funcs[r_min:r_max])
        sh_rng = shift_range(n_rng, (1 - o_rng.start))
        push!(newfunc_views, view(f, sh_rng))
    end

    return ColMap(Tuple(shifted_new_ranges), 
                  Tuple(newfunc_views))
end


#############################################
# One argument

function (cm::ColMap)(A::AbstractMatrix)
    return hcat(map((f,rng)->f(A[:,rng]), cm.funcs, cm.col_ranges)...)
end

function ChainRules.rrule(cm::ColMap, A)

    pbs = []
    result = similar(A)
    for (rng, f) in zip(cm.col_ranges, cm.funcs)
        res, pb = ChainRules.rrule(f, A[:,rng])
        result[:,rng] .= res
        push!(pbs, pb)
    end

    function colmap_pullback(result_bar)
        A_bar = similar(A)
        funcs_bar = []
        for (rng, pb) in zip(cm.col_ranges, pbs)
            fb, Ab = pb(result_bar[:,rng])
            A_bar[:,rng] .= Ab
            push!(funcs_bar, fb)
        end

        cm_bar = Tangent{ColMap}(funcs=Tuple(funcs_bar))
        return cm_bar, A_bar
    end

    return result, colmap_pullback
end


#####################################################
# Two arguments
function (cm::ColMap)(A::AbstractMatrix, D::AbstractMatrix)
    return hcat(map((f,rng)->f(A[:,rng], D[:,rng]), cm.funcs, cm.col_ranges)...)
end


function ChainRules.rrule(cm::ColMap, A, D)

    pbs = []
    result = similar(A)
    for (rng, f) in zip(cm.col_ranges, cm.funcs)
        res, pb = ChainRules.rrule(f, A[:,rng], D[:,rng])
        result[:,rng] .= res
        push!(pbs, pb)
    end

    function colmap_pullback(result_bar)
        A_bar = similar(A)
        funcs_bar = []
        for (rng, pb) in zip(cm.col_ranges, pbs)
            fb, Ab, _ = pb(result_bar[:,rng])
            A_bar[:,rng] .= Ab
            push!(funcs_bar, fb)
        end

        cm_bar = Tangent{ColMap}(funcs=Tuple(funcs_bar))
        return cm_bar, A_bar, ChainRulesCore.NoTangent()
    end

    return result, colmap_pullback
end


