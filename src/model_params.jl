

mutable struct ModelParams

    X::BMFMat
    Y::BMFMat

    mu::BMFVec
    log_sigma::BMFVec

    theta::BlockMatrix
    log_delta::BlockMatrix

end


# Create a ModelParams object from a Model object
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

# Create a copy of mp with all 
# values set to zero
function Base.zero(mp::ModelParams)
    values = []
    for pn in propertynames(mp)
        push!(values, zero(getproperty(mp, pn)))
    end
    return ModelParams(values...)
end


function Base.map!(f::Function, destination::ModelParams, 
                   collection::ModelParams)

    for pn in propertynames(destination)
        map!(f, getproperty(destination, pn), 
                getproperty(collection, pn))
    end

end


function add!(a::AbstractArray, b::AbstractArray)
    a .+= b
end


# Add b to a, mutating a
function add!(a::ModelParams, b::ModelParams)
    for pn in propertynames(a)
        add!(getproperty(a, pn), getproperty(b,pn))
    end
end



