

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

# Accumulate the sum of `collection` in its first entry. 
function accumulate_sum!(collection)
    for i=2:length(collection)
        binop!(.+, collection[1], collection[i])
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

# GPU implementations
function batched_map_gpu(mp, D_list...; capacity=10^8)
    M, N = size(D_list[1])
    row_batch_size = div(capacity, N)
    n_batches = div(M, row_batch_size)
    result = Vector{Any}(undef, n_batches)

    for (i, row_batch) in enumerate(BatchIter(M, row_batch_size))
        result[i] = mp(map(D -> view(D, row_batch, :), D_list)...) 
    end

    return result
end

function batched_mapreduce_gpu(mp, red, D_list...; capacity=10^8, start=Float32(0.0))
    
    M, N = size(D_list[1])
    row_batch_size = div(capacity, N)
    result = start

    for row_batch in BatchIter(M, row_batch_size)
        view_tuple = map(D -> view(D, row_batch, :), D_list)
        result = red(result, mp(view_tuple...))
    end

    return result
end


# Multiprocess CPU implementations
function batched_map_cpu(mp, D_list...; capacity=10^8)

    M, N = size(D_list[1])
    nthread = Threads.nthreads()
    capacity = min(capacity, M*N)
    row_batch_size = max(div(capacity,(N*nthread)),1)
    row_batches = batch_iterations(M, row_batch_size)
    n_batches = length(row_batches)
    result = Vector{Any}(undef, n_batches)

    Threads.@threads for i=1:n_batches
        row_batch=row_batches[i]
        result[i] = mp(map(D -> view(D, row_batch, :), D_list)...) 
    end

    return result
end

function batched_mapreduce_cpu(mp, red, D_list...; capacity=10^8, start=0.0)

    # Apply the map function in a parallelized fashion
    mapped_vals = batched_map_cpu(mp, D_list...; capacity=capacity)
    
    # Accumulate the reduced result
    result = start
    for v in mapped_vals
        result = red(result, v)
    end

    return result
end

test_cuda(D_list...) = any(map(d->isa(d, CuArray), D_list))

function batched_map(mp, D_list...; capacity=10^8)
    if test_cuda(D_list...)
        return batched_map_gpu(mp, D_list...; capacity=capacity)
    else
        return batched_map_cpu(mp, D_list...; capacity=capacity)
    end
end


function batched_mapreduce(mp, red, D_list...; capacity=10^8, start=Float32(0.0))
    if test_cuda(D_list...)
        return batched_mapreduce_gpu(mp, red, D_list...; capacity=capacity, start=start)
    else
        return batched_mapreduce_cpu(mp, red, D_list...; capacity=capacity, start=start)
    end
end


###################################################
# Compute means and variances of data columns
###################################################

function column_nonnan(D::AbstractMatrix)
    nonnan_idx = (!isnan).(D)
    N = size(D,2)
    result = similar(D,N)
    result .= vec(sum(nonnan_idx, dims=1))
    return result
end

function column_nansum(D::AbstractMatrix)
    nan_idx = (!isfinite).(D)
    tozero!(D, nan_idx)

    # Compute column means
    sum_vec = sum(D, dims=1)
    tonan!(D, nan_idx)

    return sum_vec
end

function column_nanmeans(D::AbstractMatrix)

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

    nan_mean_idx = isnan.(mean_vec)
    mean_vec[nan_mean_idx] .= 0

    return mean_vec
end


function batched_column_nanvar(D; capacity=10^8)
    M, N = size(D)
    nonnan_idx = isfinite.(D)
    nan_idx = (!).(nonnan_idx)
    M_vec = vec(sum(nonnan_idx, dims=1))
    D[nan_idx] .= 0

    s = similar(D,1, N)
    s .= 0
    s = vec(batched_mapreduce(d->sum(d, dims=1),
                              (s,Z) -> s .+ Z,
                              D; start=s, capacity=capacity))
    s .= s ./ M_vec
    #s .= s ./ M

    ssq = similar(D, 1, N)
    ssq .= 0
    ssq = vec(batched_mapreduce(d->sum(d.*d, dims=1),
                                (ssq,Z) -> ssq .+ Z,
                                D; start=ssq, capacity=capacity))
    ssq .= ssq ./ M_vec
    #ssq .= ssq ./ M

    D[nan_idx] .= NaN
    
    vars = ssq .- (s.*s)
    return vars 
end


function batched_link_mean(noise_model, D; capacity=10^8, latent_map_fn=l->l)

    M, N = size(D)

    nonnan_idx = isfinite.(D)
    nan_idx = (!).(nonnan_idx) 
    M_vec = vec(sum(nonnan_idx, dims=1))

    # Apply the link function to the data;
    # then, optionally, apply another function
    # to the latent representation. 
    # Set the nonfinite entries to zero.
    function map_func(d)
        l = latent_map_fn(link(noise_model, d))
        l[(!isfinite).(l)] .= 0
        return sum(l, dims=1)
    end

    reduce_start = similar(D, 1, N)
    reduce_start .= 0
    mean_vec = vec(batched_mapreduce(map_func,
                                     (s,Z) -> s .+ Z,
                                     D; start=reduce_start, capacity=capacity)
                  ) ./M_vec
    return mean_vec
end


function sqerr_func(model, D)
    diff = (link(model.noise_model, D) .- model())
    diff .*= diff
    diff[(!isfinite).(diff)] .= 0
    return diff
end


function batched_link_col_sqerr(model, D::AbstractMatrix; capacity=10^8)

    N = size(D, 2)
    reduce_start = similar(D, 1, N)
    reduce_start .= Float32(0)
    result = vec(batched_mapreduce((m,d)->sum(sqerr_func(m,d), dims=1),
                                   (st, ssq) -> st .+ ssq,
                                   model, D; start=reduce_start, capacity=capacity)
                )
    return result
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


#######################################################
# Training history utils
#######################################################

# "hist" is a dictionary; its entries are lists.
# With each iteration, we append values to them.

function history!(nothing; kwargs...)
    return
end

function history!(hist::AbstractDict; kwargs...)
    for (k,v) in kwargs
        l = get!(hist, string(k), MutableLinkedList())
        push!(l, v)
    end
end

function finalize_history!(hist::AbstractDict; kwargs...)
    for k in keys(hist)
        hist[k] = collect(hist[k])
    end
    for (k,v) in kwargs
        hist[string(k)] = v
    end
end

function finalize_history!(hist::Nothing; kwargs...)
    return
end

####################################################
# Debugging tool
####################################################

function print_nan(X, name)
    nan_idx = (!isfinite).(X)
    if any(nan_idx)
        println(string("Non-finite values in ", name, ": ", sum(nan_idx)))
    end
end
