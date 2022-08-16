
import Base: view

export MatFacModel, save_model, load_model

mutable struct MatFacModel
    X::AbstractMatrix
    Y::AbstractMatrix
    row_transform::Any  # These transforms may be functions
    col_transform::Any  # or callable structs (with trainable parameters)
    noise_model::CompositeNoise

    X_reg::Any                   # These regularizers may be functions
    Y_reg::Any                   # or callable structs (with trainable
    row_transform_reg::Any       # parameters)
    col_transform_reg::Any
    noise_model_reg::Any

    lambda_X::Number             # Regularizer weights
    lambda_Y::Number
    lambda_row::Number
    lambda_col::Number
    lambda_noise::Number
end

@functor MatFacModel


function MatFacModel(X::AbstractMatrix, Y::AbstractMatrix, 
                     noise_model::CompositeNoise;
                     row_transform=x->x,
                     col_transform=x->x,
                     X_reg=x->0.0, Y_reg=x->0.0,
                     row_transform_reg=x->0.0,
                     col_transform_reg=x->0.0,
                     noise_model_reg=x->0.0,
                     lambda_X=1.0, lambda_Y=1.0,
                     lambda_row=1.0, lambda_col=1.0,
                     lambda_noise=1.0)

    row_transform = make_viewable(row_transform)
    col_transform = make_viewable(col_transform)
    
    return MatFacModel(X, Y,
                       row_transform,
                       col_transform, 
                       noise_model,
                       X_reg, Y_reg,
                       row_transform_reg, 
                       col_transform_reg, 
                       noise_model_reg,
                       lambda_X, lambda_Y,
                       lambda_row, lambda_col,
                       lambda_noise)

end


"""
    MatFacModel(M::Int, N::Int, K::Int, col_losses::Vector{String};
                row_transform=identity,
                col_transform=identity,
                X_reg=x->0, Y_reg=y->0, 
                row_transform_reg=x->0,
                col_transform_reg=x->0,
                noise_model_reg=x->0)

Construct a matrix factorization model for an
M x N dataset. Assume K latent factors.

Specify the loss for each column via `col_losses`,
a length-N vector of strings. Permissible string values are:

* "normal"
* "bernoulli"
* "poisson"
* "ordinalN", where "N" is replaced by some integer.
  E.g., "ordinal3", "ordinal4".

All of the keyword arguments are either (a) functions or
(b) callable structs (i.e., functors). We allow for the possibility
for each of these to contain trainable parameters.

* `row_transform`: a (trainable) transformation on the _rows_ of the model.
* `col_transform`: a (trainable) transformation on the _columns_ of the model.
* `*_reg`: a (trainable) regularizer for the specified object.

"""
function MatFacModel(M::Integer, N::Integer, K::Integer,
                     col_losses::Vector{String}; kwargs...)

    X = randn(K,M) ./ (sqrt(K)*100) 
    Y = randn(K,N) ./ 100

    noise_model = CompositeNoise(col_losses)

    return MatFacModel(X, Y, noise_model; kwargs...)
end


"""
    MatFacModel(M, N, K, loss::String; kwargs...)

Identical to the other constructor, but applies the same
`loss` function to all columns of data.
"""
function MatFacModel(M::Integer, N::Integer, K::Integer,
                     loss::String; kwargs...)
    losses = fill(loss, N)
    return MatFacModel(M, N, K, losses; kwargs)
end



function (bm::MatFacModel)()
    return invlink(bm.noise_model,
            bm.col_transform(
             bm.row_transform(
               transpose(bm.X)*bm.Y
             )
            )
           )
end



function view(bm::MatFacModel, idx1, idx2)
    return MatFacModel(view(bm.X, :, idx1),
                            view(bm.Y, :, idx2),
                            view(bm.row_transform, idx1, idx2),
                            view(bm.col_transform, idx1, idx2),
                            view(bm.noise_model, idx2),
                            nothing, nothing, nothing,
                            nothing, nothing,
                            0.0, 0.0, 0.0, 0.0, 0.0
                           )

end


function Base.size(bm::MatFacModel)
    return (size(bm.X, 2), size(bm.Y, 2))
end


##############################################
# Equality operation
EqTypes = Union{CompositeNoise,NormalNoise,PoissonNoise,BernoulliNoise,OrdinalNoise,
                MatFacModel,ViewableFunction}

NoEqTypes = Function

function Base.:(==)(a::T, b::T) where T <: EqTypes
    for fn in fieldnames(T)
        af = getfield(a, fn)
        bf = getfield(b, fn)
        if !(af == bf)
            if !((typeof(af) <: NoEqTypes) & (typeof(bf) <: NoEqTypes))
                return false
            end
        end
    end
    return true
end


################################################
# Model file I/O

"""
    save_model(filename, model)

Save `model` to a BSON file located at `filename`.
"""
function save_model(filename, model)
    BSON.@save filename model
end

"""
    load_model(filename)

load a model from the BSON located at `filename`.
"""
function load_model(filename)
    d = BSON.load(filename, @__MODULE__)
    return d[:model]
end

