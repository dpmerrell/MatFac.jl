

###################################
# AdaGradUpdate
###################################
"""
This struct maintains the state of an Adagrad optimizer.
When supplied with a gradient, it updates the state
and returns the appropriate additive updates
"""
mutable struct AdaGradUpdate
    sum_sq
end


function adagrad_updater(arrays...; epsilon=1e-8)
    sum_sq = [zero(arr) .+ epsilon for arr in arrays]
    return AdaGradUpdate(sum_sq)
end


function (agu::AdaGradUpdate)(lr::Real, gradient)

    for (arr, grad, ssq) in zip(arrays, gradients, agu.sum_sq)
        ssq .+= grad.^2
        arr .-= (lr .* grad ./ sqrt.(sum_sq))
    end

end



