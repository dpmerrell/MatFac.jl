

mutable struct ModelParams 

    X::BMFMat
    Y::BMFMat

    mu::BMFVec
    log_sigma::BMFVec

    theta::BlockMatrix
    log_delta::BlockMatrix

end


# Create a YParams object from a Model object
function ModelParams(model::BMFModel)

    row_ranges = ids_to_ranges(model.sample_group_ids)
    col_ranges = ids_to_ranges(model.feature_group_ids)

    return ModelParams(model.X, model.Y,
                       model.mu, model.log_sigma,
                       BlockMatrix(model.theta, 
                                   row_ranges, 
                                   col_ranges),
                       BlockMatrix(model.log_delta, 
                                   row_ranges, 
                                   col_ranges)
                      )
end


function Base.map(f::Function, collection::ModelParams; 
                  fields::Union{Nothing,Vector{Symbol}}=nothing)
    values = []
    if fields == nothing
        fields = propertynames(collection)
    end
    for pn in fields 
        push!(values, map(f, getproperty(collection, pn)))
    end
    return typeof(collection)(values...)
end


# Create a copy of mp with all 
# values set to zero
function Base.zero(mp::ModelParams)
    return map(zero, mp)
end


function Base.map!(f::Function, destination::ModelParams, 
                   collection::ModelParams;
                   fields::Union{Nothing,Vector{Symbol}}=nothing)

    if fields == nothing
        fields = propertynames(collection)
    end

    for pn in fields 
        map!(f, getproperty(destination, pn), 
                getproperty(collection, pn))
    end
end


######################################
# DEFINE ARITHMETIC OPERATIONS
function binop!(op::Function, a::AbstractArray, b::AbstractArray)
    a .= op(a, b)
end


function binop!(op::Function, a::AbstractArray, b::Number)
    a .= op(a, b)
end

function binop!(op::Function, a::ModelParams, b::ModelParams;
                fields::Union{Nothing,Vector{Symbol}}=nothing)
    
    if fields == nothing
        fields = propertynames(a)
    end
    for pn in fields
        binop!(op, getproperty(a, pn), getproperty(b,pn))
    end
end

function binop!(op::Function, a::ModelParams, b::Number;
                fields::Union{Nothing,Vector{Symbol}}=nothing)
    
    if fields == nothing
        fields = propertynames(a)
    end
    for pn in fields 
        binop!(op, getproperty(a, pn), b)
    end
end



