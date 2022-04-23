
import Base: view

##################################
# Matrix Product
mutable struct MatProd
    X::AbstractMatrix
    Y::AbstractMatrix
end

@functor MatProd

function MatProd(M::Int, N::Int, K::Int)
    X = randn(K,M) .* .01 / sqrt(K)
    Y = randn(K,N) .* .01
    return MatProd(X,Y)
end


function (m::MatProd)()
    return transpose(m.X)*m.Y
end


function view(m::MatProd, idx1, idx2)
    return MatProd(view(m.X,:,idx1),
                   view(m.Y,:,idx2))
end


##################################
# Column Scale

mutable struct ColScale
    logsigma::AbstractVector
end

@functor ColScale

function ColScale(N::Int)
    return ColScale(zeros(N))
end


function (cs::ColScale)(Z::AbstractMatrix)
    return Z .* transpose(exp.(cs.logsigma))
end


function view(cs::ColScale, idx)
    return ColScale(view(cs.logsigma, idx))
end


##################################
# Column Shift

mutable struct ColShift
    mu::AbstractVector
end

@functor ColShift

function ColShift(N::Int)
    return ColShift(randn(N) .* 1e-4)
end


function (cs::ColShift)(Z::AbstractMatrix)
    return Z .+ transpose(cs.mu)
end


function view(cs::ColShift, idx)
    return ColShift(view(cs.mu, idx))
end


#################################
# Batch Scale

mutable struct BatchScale
    logdelta::BatchArray
end

@functor BatchScale

function BatchScale(col_batches, row_batches)

    values = [Dict(urb => 0.0 for urb in unique(rbv)) for rbv in row_batches]
    logdelta = BatchArray(col_batches, row_batches, values)

    return BatchScale(logdelta)
end


function (bs::BatchScale)(Z::AbstractMatrix)
    return Z * exp(bs.logdelta)
end


function view(bs::BatchScale, idx1, idx2)
    if typeof(idx2) == Colon
        idx2 = 1:bs.logdelta.col_ranges[end].stop
    end
    return BatchScale(view(bs.logdelta, idx1, idx2)) 
end


##################################
# Batch Shift

mutable struct BatchShift
    theta::BatchArray
end

@functor BatchShift

function BatchShift(col_batches, row_batches)
    
    values = [Dict(urb => 0.0 for urb in unique(rbv)) for rbv in row_batches]
    theta = BatchArray(col_batches, row_batches, values)

    return BatchShift(theta)
end


function (bs::BatchShift)(Z::AbstractMatrix)
    return Z + bs.theta
end


function view(bs::BatchShift, idx1, idx2)
    if typeof(idx2) == Colon
        idx2 = 1:bs.theta.col_ranges[end].stop
    end
    return BatchShift(view(bs.theta, idx1, idx2))
end


