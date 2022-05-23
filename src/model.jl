
import Base: view

export BatchMatFacModel, save_model, load_model

mutable struct BatchMatFacModel
    X::AbstractMatrix
    Y::AbstractMatrix
    row_transform
    col_transform
    noise_model::CompositeNoise

    # Regularizers. May be pure functions
    # or callable structs (with their own 
    # trainable parameters!)
    X_reg
    Y_reg
    row_transform_reg
    col_transform_reg
    noise_model_reg
end

@functor BatchMatFacModel


function BatchMatFacModel(X::AbstractMatrix, Y::AbstractMatrix, 
                          cscale::ColScale, cshift::ColShift, 
                          bscale::BatchScale, bshift::BatchShift, 
                          noise_model::CompositeNoise;
                          X_reg=x->0.0, Y_reg=x->0.0,
                          row_transform_reg=x->0.0,
                          col_transform_reg=x->0.0,
                          noise_model_reg=x->0.0)

    
    row_transform = x -> x
    col_transform = BatchMatFacLayers(cscale, cshift,
                                      bscale, bshift)


    return BatchMatFacModel(X, Y,
                            row_transform,
                            col_transform, 
                            noise_model,
                            X_reg, Y_reg,
                            row_transform_reg, 
                            col_transform_reg, 
                            noise_model_reg)

end


function BatchMatFacModel(M::Integer, N::Integer, K::Integer,
                          col_batch_ids::Vector, row_batch_ids::Vector,
                          col_losses::Vector{String}; kwargs...)

    X = randn(K,M)
    Y = randn(K,N)
    cscale = ColScale(N)
    cshift = ColShift(N)
    bscale = BatchScale(col_batch_ids, row_batch_ids)
    bshift = BatchShift(col_batch_ids, row_batch_ids)

    noise_model = CompositeNoise(col_losses)

    return BatchMatFacModel(X, Y, cscale, cshift,
                                  bscale, bshift,
                                  noise_model; kwargs...)
end


function (bm::BatchMatFacModel)()
    return invlink(bm.noise_model,
            bm.col_transform(
             bm.row_transform(
               transpose(bm.X)*bm.Y
             )
            )
           )
end


function view(bm::BatchMatFacModel, idx1, idx2)
    return BatchMatFacModel(view(bm.X, :, idx1),
                            view(bm.Y, :, idx2),
                            view(bm.row_transform, idx1, idx2),
                            view(bm.col_transform, idx1, idx2),
                            view(bm.noise_model, idx2),
                            nothing, nothing, nothing,
                            nothing, nothing,
                           )

end


function Base.size(bm::BatchMatFacModel)
    return (size(bm.X, 2), size(bm.Y, 2))
end

##############################################
# Equality operation
EqTypes = Union{ColScale,ColShift,BatchScale,BatchShift,CompositeNoise,
                BatchArray,NormalNoise,PoissonNoise,BernoulliNoise,OrdinalNoise,
                BatchMatFacModel,BatchMatFacLayers}

NoEqTypes = Function

function Base.:(==)(a::T, b::T) where T <: EqTypes
    for fn in fieldnames(T)
        af = getfield(a, fn)
        bf = getfield(b, fn)
        if !(af == bf)
            if !((typeof(af) <: NoEqTypes) & (typeof(bf) <: NoEqTypes))
                println(string("NOT EQUAL: ", fn))
                return false
            end
        end
    end
    return true
end


################################################
# Model file I/O

function save_model(filename, model)
    BSON.@save filename model
end

function load_model(filename)
    d = BSON.load(filename, @__MODULE__)
    return d[:model]
end

