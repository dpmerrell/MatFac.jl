
import Base: map!

mutable struct BatchMatFacModelParams

        X::AbstractMatrix
        Y::AbstractMatrix

        mu::AbstractVector
        sigma::AbstractVector

        theta::AbstractVector
        delta::AbstractVector

end

Params = BatchMatFacModelParams

function BatchMatFacModelParams(model::BatchMatFacModel; rule=copy)
    return Params(rule(model.X),
                  rule(model.Y),
                  rule(model.mu),
                  rule(model.sigma),
                  rule(model.theta.values),
                  rule(model.delta.values)
                  )
end


function add!(a::Params, b::Params)
    for field_name in fieldnames(Params)
        setfield!(a, field_name, getfield(a, field_name) .+ getfield(b, field_name)) 
    end
end

function mult!(a::Params, b::Params)
    for field_name in fieldnames(Params)
        setfield!(a, field_name, getfield(a, field_name) .* getfield(b, field_name)) 
    end
end

function map!(func, destination::Params, collection::Params)
    for field_name in fieldnames(Params)
        setfield!(destination, field_name, func(getfield(collection, field_name)))
    end
end

function add!(model::BatchMatFacModel, params::Params)
    model.X .+= params.X
    model.Y .+= params.Y
    model.mu .+= params.mu
    model.sigma .+= params.sigma
    model.theta.values .+= params.theta
    model.delta.values .+= params.delta
end

