

###################################
# AdaGradUpdater
###################################
"""
This struct maintains the state of an Adagrad optimizer.
When supplied with a gradient, it updates the state
and returns the appropriate additive updates
"""
mutable struct AdaGradUpdater
    sum_sq::ModelParams
end


# Convenience constructor
function adagrad_updater(mp::ModelParams; epsilon=1e-8) 
                         

    my_init = x -> x .+ epsilon

    sum_sq = zero(mp)
    map!(my_init, sum_sq, sum_sq)

    return AdaGradUpdater(sum_sq)
end


function load_property_arr(mp::ModelParams, p::Symbol)
    a = getproperty(mp, p)
    if typeof(a) <: BatchMatrix
        return a.values
    end
    return a
end


#################################################
# Apply the update. Mutates the params inplace
#################################################
function (agu::AdaGradUpdater)(params::ModelParams, gradients::ModelParams; lr=0.01,
                               fields::Union{Nothing,Vector{Symbol}}=nothing)

    if fields == nothing
        fields = propertynames(gradients)
    end

    for pn in fields
        sum_sq = load_property_arr(agu.sum_sq, pn)
        grad = load_property_arr(gradients, pn)
    
        grad_sq = my_sq(grad)
        binop!(.+, sum_sq, grad_sq)

        value = load_property_arr(params, pn)

        # compute the following: value .-= lr .* grad ./ sqrt.(sum_sq)
        update = binop(.*, grad, lr)
        binop!(./, update, my_sqrt(sum_sq))
        binop!(.-, value, update)
    end

end



