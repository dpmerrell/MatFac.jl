
export BatchMatFacModel, forward


mutable struct BatchMatFacModel

        X::AbstractMatrix
        Y::AbstractMatrix

        X_reg::AbstractVector{AbstractMatrix}
        Y_reg::AbstractVector{AbstractMatrix}

        mu::AbstractVector
        sigma::AbstractVector

        mu_reg::AbstractMatrix
        sigma_reg::AbstractMatrix

        theta::BatchQuantity
        delta::BatchQuantity

        sample_batches::AbstractVector{Int}
        feature_batches::AbstractVector{Int}

        feature_link_map::ColRangeMap
        feature_loss_map::ColRangeAgg

end


function BatchMatFacModel(X_reg, Y_reg, mu_reg, sigma_reg,
                          sample_batch_ids, feature_batch_ids,
                          feature_loss_names; on_gpu=true)

    M = size(X_reg[1], 1)
    N = size(Y_reg[1], 1)
    K = length(X_reg)

    @assert K == length(Y_reg)

    if on_gpu
        my_randn = CUDA.randn
        my_ones = CUDA.ones
    else
        my_randn = randn
        my_ones = ones
    end

    # These may be on the GPU or CPU
    X = my_randn(Float32, K,M) ./ Float32(10.0*sqrt(K))
    Y = my_randn(Float32, K,N) ./ Float32(10.0*sqrt(K))
    mu = my_randn(Float32, N) ./ Float32(100.0)
    sigma = my_ones(Float32, N)

    n_sample_batches = length(unique(sample_batch_ids))
    n_feature_batches = length(unique(feature_batch_ids))

    # These will always be on the CPU -- they're relatively small
    theta_value = randn(Float32, n_sample_batches, n_feature_batches) ./ Float32(100.0)
    theta = BatchQuantity(theta_value, sample_batch_ids, feature_batch_ids) 
    delta_value = ones(Float32, n_sample_batches, n_feature_batches)
    delta = BatchQuantity(delta_value, sample_batch_ids, feature_batch_ids) 

    link_function_map = ColRangeMap([LINK_FUNCTION_MAP[ln] for ln in unique(feature_loss_names)], feature_loss_names)
    loss_function_map = ColRangeAgg([LOSS_FUNCTION_MAP[ln] for ln in unique(feature_loss_names)], feature_loss_names)

    return BatchMatFacModel(X, Y, X_reg, Y_reg, 
                            mu, sigma, mu_reg, sigma_reg,
                            theta, delta, 
                            sample_batch_ids, feature_batch_ids,
                            link_function_map, loss_function_map)

end



function forward(model::BatchMatFacModel)

    A = (transpose(model.X)*model.Y).* transpose(model.sigma) .+ transpose(model.mu)

    A = bq_add( bq_mult(A, model.delta), model.theta)

    A = model.feature_link_map(A)

    return A
end


#function ChainRules.rrule(forward, model::BatchMatFacModel)
#
#    A = forward(model)
#    function forward_rrule(A_bar)
#        model_bar = ChainRules.Tangent{BatchMatFacModel}()
#
#        return ChainRules.NoTangent(), model_bar
#    end
#    return A, forward_rrule
#end


