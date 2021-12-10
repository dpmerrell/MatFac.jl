
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
        feature_loss_map::ColRangeMap

end


function BatchMatFacModel(X_reg, Y_reg, mu_reg, sigma_reg,
                          sample_batch_ids, feature_batch_ids,
                          feature_loss_names)

    M = size(X_reg[1], 1)
    N = size(Y_reg[1], 1)
    K = length(X_reg)

    @assert K == length(Y_reg)

    X = randn(K,M) ./ (10.0*sqrt(K))
    Y = randn(K,N) ./ (10.0*sqrt(K))
    mu = randn(N) ./ 100.0 
    sigma = ones(N)

    n_sample_batches = length(unique(sample_batch_ids))
    n_feature_batches = length(unique(feature_batch_ids))

    theta_value = randn(n_sample_batches, n_feature_batches) ./ 100.0
    theta = BatchQuantity(theta_value, sample_batch_ids, feature_batch_ids) 
    delta_value = ones(n_sample_batches, n_feature_batches)
    delta = BatchQuantity(delta_value, sample_batch_ids, feature_batch_ids) 

    link_function_map = ColRangeMap([LINK_FUNCTION_MAP[ln] for ln in unique(feature_loss_names)], feature_loss_names)
    loss_function_map = ColRangeMap([LOSS_FUNCTION_MAP[ln] for ln in unique(feature_loss_names)], feature_loss_names)

    return BatchMatFacModel(X, Y, X_reg, Y_reg, 
                            mu, sigma, mu_reg, sigma_reg,
                            theta, delta, 
                            sample_batch_ids, feature_batch_ids,
                            link_function_map, loss_function_map)

end



function forward(model::BatchMatFacModel)

    A = (transpose(model.X)*model.Y).* transpose(model.sigma) .+ transpose(model.mu)

    A = bq_add( bq_mult(A, model.delta), model.theta )

    A = model.feature_link_map(A)

    return A
end



