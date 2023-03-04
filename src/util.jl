

#################################################
# Extensions to Flux's update! function
#################################################


import Flux.Optimise: update!


Optimiser = Flux.Optimise.AbstractOptimiser


TupleTypes = Union{Tuple,NamedTuple}

function update!(opt::Optimiser, params::Any, grads::Nothing)
    return
end

function update!(opt::Optimiser, params::AbstractArray, grads::Nothing)
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

# We do not update pure functions!
function update!(opt::Optimiser, params::Function, grads::TupleTypes)
    return
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
# Other arithmetic operations for parameters 
# and gradients
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


tozero(x::Tuple{}) = ()
tozero(x::Nothing) = nothing
tozero(x) = zero(x)

function zero_out!(x::TupleTypes)
    for pname in propertynames(x)
        v = getproperty(x, pname)
        zero_out!(v)
    end
end

function zero_out!(x::AbstractArray)
    x .= 0
end


####################################################
# Extract a whole tree of trainable parameters
####################################################

function rec_trainable(obj::AbstractArray{<:Number})
    return obj
end

function rec_trainable(obj::Number)
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
# Functions for setting parameters to zero
###################################################

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
# "Batched" operations for large arrays
###################################################

function batched_mapreduce(mp, red, D_list...; capacity=Int(25e6), start=0.0)

    M, N = size(D_list[1])
    row_batch_size = div(capacity, N)
    result = start

    for row_batch in BatchIter(M, row_batch_size)
        view_tuple = map(D -> view(D, row_batch, :), D_list)
        result = red(result, mp(view_tuple...))
    end

    return result
end

function batched_reduce(red, D::AbstractMatrix; kwargs...)
    return batched_mapreduce(x -> x, red, D; kwargs...)
end 


###################################################
# Compute means and variances of data columns
###################################################

function column_nonnan(D::AbstractMatrix)
    nonnan_idx = (!isnan).(D)
    M_vec = vec(sum(nonnan_idx, dims=1))
    return M_vec
end


function column_means(D::AbstractMatrix)

    nan_idx = isnan.(D)
    tozero!(D, nan_idx)
    nonnan_idx = (!).(nan_idx)
    M_vec = vec(sum(nonnan_idx, dims=1))

    # Compute column means
    sum_vec = vec(sum(D, dims=1))
    mean_vec = sum_vec ./ M_vec
    mean_nan_idx = isnan.(mean_vec)
    tozero!(mean_vec, mean_nan_idx)
    tonan!(D, nan_idx)

    return mean_vec
end


function batched_column_meanvar(D::AbstractMatrix; capacity=Int(25e6))

    M, N = size(D)
    
    # Replace NaNs with zeros
    nan_idx = isnan.(D)
    nonnan_idx = (!).(nan_idx)
    tozero!(D, nan_idx)
    M_vec = vec(sum(nonnan_idx, dims=1))

    # Compute column means
    sum_vec = vec(sum(D, dims=1))
    mean_vec = sum_vec ./ M_vec

    # Compute column variances via 
    # V[x] = E[x^2] - E[x]^2
    reduce_start = similar(D, 1, N)
    reduce_start .= 0
    sumsq_vec = vec(batched_reduce((v, D2) -> v .+ sum(D2.*D2; dims=1), D;
                                 start=reduce_start, capacity=capacity)
                   )
    meansq_vec = sumsq_vec ./ M_vec
    var_vec = meansq_vec .- (mean_vec.*mean_vec)

    # Restore NaN values
    tonan!(D, nan_idx)

    return mean_vec, var_vec
end


function batched_link_scale(model, D; capacity=10^8, latent_map_fn=(m,Z)->Z)

    M, N = size(D)
    
    # Replace NaNs with zeros
    nan_idx = isnan.(D)
    nonnan_idx = (!).(nan_idx)
    tozero!(D, nan_idx)
    M_vec = vec(sum(nonnan_idx, dims=1))

    nm = model.noise_model

    # Compute column means -- after link function
    reduce_start = similar(D, 1, N)
    reduce_start .= 0
    mean_vec = vec(batched_mapreduce((m, d) -> latent_map_fn(m, link(nm, d)), 
                                     (s, Z) -> s .+ sum(Z, dims=1), 
                                     model, D; start=reduce_start, capacity=capacity)
                  ) ./ M_vec

    # Compute column variances via 
    # V[x] = E[x^2] - E[x]^2
    # (after link function)
    reduce_start = similar(D, 1, N)
    reduce_start .= 0
    sumsq_vec = vec(batched_mapreduce((m, d) -> latent_map_fn(m, link(nm, d)).^2, 
                                      (s, Z) -> s .+ sum(Z, dims=1), 
                                      model, D; start=reduce_start, capacity=capacity)
                   ) 
    meansq_vec = sumsq_vec ./ M_vec
    var_vec = meansq_vec .- (mean_vec.*mean_vec)

    # Restore NaN values
    tonan!(D, nan_idx)

    return sqrt.(var_vec)
end


function column_total_loss(noise_model, D, mean_vec)
    col_mean = repeat(transpose(mean_vec), size(D,1), 1)
    ls = loss(noise_model, col_mean, D; calibrate=true)
    nan_idx = isnan.(ls)
    tozero!(ls, nan_idx)
    return vec(sum(ls, dims=1))
end


function batched_data_loss(model, D::AbstractMatrix; capacity=Int(25e6))

    M, N = size(D)
    total_loss = batched_mapreduce((model, D) -> data_loss(model, D; calibrate=true),
                                   (x,y) -> x+y, 
                                   model, D; start=0.0, capacity=capacity)
    return total_loss 
end




