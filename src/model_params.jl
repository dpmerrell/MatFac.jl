

mutable struct ModelParams 

    X::BMFMat
    Y::BMFMat

    mu::BMFVec
    log_sigma::BMFVec

    theta::BatchMatrix
    log_delta::BatchMatrix

end


# Create a YParams object from a Model object
function ModelParams(model::BMFModel)

    return ModelParams(model.X, model.Y,
                       model.mu, model.log_sigma,
                       batch_matrix(model.theta_values, 
                                    model.row_batch_dicts, 
                                    model.col_batches),
                       batch_matrix(model.log_delta_values, 
                                    model.row_batch_dicts, 
                                    model.col_batches)
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



