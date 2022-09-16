

#################################################
# Extensions to Flux's update! function
#################################################


import Flux.Optimise: update!


Optimiser = Flux.Optimise.AbstractOptimiser


TupleTypes = Union{Tuple,NamedTuple}

function update!(opt::Optimiser, params::Any, grads::Nothing)
    return
end

function update!(opt::Optimiser, params::Any, grads::TupleTypes)
    for pname in propertynames(grads)
        g = getproperty(grads,pname)
        if g != nothing
            update!(opt, getproperty(params, pname), g) 
        end
    end
end


#################################################
# Extensions to Functors' fmap functions
#################################################
import Functors: fmap, fmapstructure

# I don't see why Functors doesn't handle
# this case by default...
fmap(f, t::Tuple{}) = ()
fmapstructure(f, t::Tuple{}) = ()

###################################################
# Other arithmetic operations for gradient updates
###################################################

function binop!(op, a::Any, b::Nothing)
    return
end

function binop!(op, a::AbstractArray, b::AbstractArray)
    a .= op(a,b)
end

function binop!(op, a::Any, b::TupleTypes)
    for pname in propertynames(b)
        v = getproperty(b, pname)
        if v != nothing
            u = getproperty(a, pname)
            binop!(op, u, v)
        end
    end
end

####################################################
# Extract a whole tree of trainable parameters
####################################################

function rec_trainable(obj::AbstractArray)
    return obj
end

function rec_trainable(obj::Tuple)
    return map(rec_trainable, obj) 
end

function rec_trainable(obj)
    trn = Flux.trainable(obj)
    return map(rec_trainable, trn)
end

###################################################
# Define a custom `zero` function
###################################################

tozero(x::Tuple{}) = ()
tozero(x::Nothing) = nothing
tozero(x) = zero(x)


# Define some other useful array operations.
# CUDA.jl will compile these into efficient GPU kernels.
function replace_if(a::T, v::T, b::Bool) where T
    if b
        return v
    else
        return a
    end
end

function tozero!(A::AbstractArray{T,K}, idx::AbstractArray{Bool,K}) where T where K
    map!((a, b) -> replace_if(a, T(0), b), A, A, idx) 
end

function tonan!(A::AbstractArray{T,K}, idx::AbstractArray{Bool,K}) where T where K
    map!((a, b) -> replace_if(a, T(NaN), b), A, A, idx) 
end

function toone!(A::AbstractArray{T,K}, idx::AbstractArray{Bool,K}) where T where K
    map!((a, b) -> replace_if(a, T(1), b), A, A, idx) 
end


###################################################
# Compute means and variances of data columns
###################################################

function column_means(D::AbstractMatrix, row_batch_size::Number)

    nan_idx = isnan.(D)
    tozero!(D, nan_idx)
    nonnan_idx = (!).(nan_idx)
    M_vec = vec(sum(nonnan_idx, dims=1))

    # Compute column means
    sum_vec = vec(sum(D, dims=1))
    mean_vec = sum_vec ./ M_vec
    mean_nan_idx = isnan.(mean_vec)
    tozero!(mean_vec, mean_nan_idx)

    return mean_vec
end


function column_meanvar(D::AbstractMatrix, row_batch_size::Number)

    nan_idx = isnan.(D)
    tozero!(D, nan_idx)
    nonnan_idx = (!).(nan_idx)
    M_vec = vec(sum(nonnan_idx, dims=1))

    # Compute column means
    sum_vec = vec(sum(D, dims=1))
    mean_vec = sum_vec ./ M_vec
    mean_nan_idx = isnan.(mean_vec)
    tozero!(mean_vec, mean_nan_idx)

    # Compute column variances
    M, N = size(D)
    sumsq_vec = similar(D, N)
    sumsq_vec .= 0
    for row_batch in BatchIter(M, row_batch_size)
        diff = view(D, row_batch, :) .- transpose(mean_vec)
        batch_nonnan_idx = view(nonnan_idx, row_batch, :)
        diff .*= batch_nonnan_idx
        diff .*= diff
        sumsq_vec .+= vec(sum(diff, dims=1))
    end
    var_vec = sumsq_vec ./ (M_vec .+ 1) # Unbiased estimate
    
    # Restore NaN values
    tonan!(D, nan_idx)

    return mean_vec, var_vec
end


function column_avg_loss(noise_model, D::AbstractMatrix, row_batch_size::Number)

    M, N = size(D)
    col_mean_vec = column_means(D, row_batch_size)

    col_errors = zeros(N)
    col_nnz = zeros(N)
    for row_batch in BatchIter(M, row_batch_size)
        batch_D = view(D, row_batch, :)
        batch_error = loss(noise_model, transpose(col_mean_vec), batch_D)

        batch_nan = isnan.(batch_D)
        tozero!(batch_error, batch_nan)

        col_errors += reshape(sum(batch_error, dims=1),:)
        col_nnz += reshape(sum( (!).(batch_nan), dims=1),:)
    end
    col_errors ./= col_nnz

    return col_errors
end


