
import Base: view


mutable struct BatchMatFacModel
    mp::MatProd
    cscale::ColScale
    cshift::ColShift
    bscale::BatchScale
    bshift::BatchShift
    noise_model::CompositeNoise

    # Regularizers. May be pure functions
    # or callable structs (with their own parameters!)
    X_reg
    Y_reg
    logsigma_reg
    mu_reg
    logdelta_reg
    theta_reg
end

@functor BatchMatFacModel


function BatchMatFacModel(mp::MatProd, cscale::ColScale,
                          cshift::ColShift, bscale::BatchScale,
                          bshift::BatchShift, noise_model::CompositeNoise;
                          X_reg=nothing, Y_reg=nothing,
                          logsigma_reg=nothing, mu_reg=nothing,
                          logdelta_reg=nothing, theta_reg=nothing)

    kwargs = Dict{Symbol,Any}(:X_reg=>X_reg, :Y_reg=>Y_reg,
                              :logsigma_reg=>logsigma_reg,
                              :mu_reg=>mu_reg,
                              :logdelta_reg=>logdelta_reg,
                              :theta_reg=>theta_reg)
                  
    for (k,v) in kwargs
        if v == nothing
            kwargs[k] = x -> 0.0
        end
    end

    return BatchMatFacModel(mp, cscale, cshift, bscale,
                            bshift, noise_model,
                            kwargs[:X_reg], kwargs[:Y_reg],
                            kwargs[:logsigma_reg], kwargs[:mu_reg],
                            kwargs[:logdelta_reg], kwargs[:theta_reg])

end


function BatchMatFacModel(M::Integer, N::Integer, K::Integer,
                          col_batch_ids::Vector, row_batch_ids::Vector,
                          col_losses::Vector{String}; kwargs...)

    mp = MatProd(M,N,K)
    cscale = ColScale(N)
    cshift = ColShift(N)
    bscale = BatchScale(col_batch_ids, row_batch_ids)
    bshift = BatchShift(col_batch_ids, row_batch_ids)

    noise_model = CompositeNoise(col_losses)

    return BatchMatFacModel(mp, cscale, cshift,
                                bscale, bshift,
                                noise_model; kwargs...)
end


function (bm::BatchMatFacModel)()
    return invlink(bm.noise_model, 
                    bm.bshift(
                          bm.bscale(
                                 bm.cshift(
                                       bm.cscale(
                                             bm.mp()
                                                )
                                          )
                                    )
                             )
                  )
end


function view(bm::BatchMatFacModel, idx1, idx2)
    return BatchMatFacModel(view(bm.mp, idx1, idx2),
                            view(bm.cscale, idx2),
                            view(bm.cshift, idx2),
                            view(bm.bscale, idx1, idx2),
                            view(bm.bshift, idx1, idx2),
                            view(bm.noise_model, idx2),
                            nothing, nothing, nothing,
                            nothing, nothing, nothing
                           )

end


function Base.size(bm::BatchMatFacModel)
    return (size(bm.mp.X,2),size(bm.mp.Y,2))
end


function save_model(filename, model)
    BSON.@save filename model
end

function load_model(filename)
    BSON.@load filename model
    return model
end

