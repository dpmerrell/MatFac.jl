

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


###################################################
# Compute means and variances of data columns
###################################################

function column_meanvar(D::AbstractMatrix, M::Number, row_batch_size::Number)

    nan_idx = isnan.(D)
    D[nan_idx] .= 0
    nonnan_idx = (!).(nan_idx)
    M_vec = vec(sum(nonnan_idx, dims=1))

    # Compute column means
    sum_vec = vec(sum(D, dims=1))
    mean_vec = sum_vec ./ M_vec

    # Compute column variances
    sumsq_vec = similar(D, size(D,2))
    for row_batch in BatchIter(M, row_batch_size)
        diff = D[row_batch,:] .- transpose(mean_vec)
        batch_nan_idx = view(nan_idx, row_batch, :)
        diff[batch_nan_idx] .= 0
        diff .*= diff
        sumsq_vec .+= vec(sum(diff, dims=1))
    end
    var_vec = sumsq_vec ./ (M_vec .+ 1) # Unbiased estimate

    # Restore NaN values
    D[nan_idx] .= NaN

    return mean_vec, var_vec
end


