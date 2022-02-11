

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
    if typeof(a) == BatchMatrix
        return a.values
    end
    return a
end


########################################
# UNIFIED INTERFACE FOR ARITHMETIC OPS
########################################

function binop(op::Function, a::Vector{AbstractArray},
                             b::Vector{AbstractArray})
    return [op(u,v) for (u,v) in zip(a,b)]
end

function binop(op::Function, a::AbstractArray{<:Number},
                             b::AbstractArray{<:Number})
    return op(a,b)
end

function binop!(op::Function, a::Vector{<:AbstractArray},
                              b::Vector{<:AbstractArray})
    for (u,v) in zip(a,b)
        u .= op(u,v)
    end
end

function binop!(op::Function, a::AbstractArray{<:Number},
                              b::AbstractArray{<:Number})
    a .= op(a,b)
end

function binop(op::Function, a::Vector{<:AbstractArray},
                             k::Number)
    return [op(u,k) for u in a]
end

function binop(op::Function, a::AbstractArray{<:Number},
                             k::Number)
    return op(a,k)
end

function binop!(op::Function, a::Vector{<:AbstractArray},
                              k::Number)
    for u in a
        u .= op(u,k)
    end
end

function binop!(op::Function, a::AbstractArray{<:Number},
                              k::Number)
    a .= op(a,k)
end

my_sq = x -> x.*x
my_sqrt = x -> sqrt.(x)


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
    
        grad_sq = map(my_sq, grad)
        binop!(.+, sum_sq, grad_sq)

        value = load_property_arr(params, pn)

        # value .-= lr .* grad ./ sqrt.(sum_sq)
        update = binop(.*, grad, lr)
        binop!(./, update, map(my_sqrt, sum_sq))
        binop!(.-, value, update)
    end

end



