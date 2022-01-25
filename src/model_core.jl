


function forward(X::AbstractMatrix, Y::AbstractMatrix, mu::AbstractVector, log_sigma::AbstractVector, 
                 theta::AbstractMatrix, log_delta::AbstractMatrix,
                 row_batches::Vector{UnitRange},
                 col_batches::Vector{UnitRange},
                 feature_link_map::ColBlockMap)

    sigma = exp.(log_sigma)

    delta = exp.(log_delta)
    delta_bmat = BlockMatrix(delta, row_batches, col_batches)
    
    theta_bmat = BlockMatrix(theta, row_batches, col_batches)

    Z = transpose(X)*Y
    Z = Z .* transpose(sigma)
    Z = Z .+ transpose(mu)
    Z = Z * delta_bmat
    Z = Z + theta_bmat
    A = feature_link_map(Z)
    return A
end


function neg_log_likelihood(X::AbstractMatrix, Y::AbstractMatrix, 
                            mu::AbstractVector, log_sigma::AbstractVector, 
                            theta::AbstractMatrix, log_delta::AbstractMatrix,
                            row_batches::Vector{UnitRange},
                            col_batches::Vector{UnitRange},
                            feature_link_map::ColBlockMap, 
                            feature_loss_map::ColBlockAgg, D::AbstractMatrix)

    A = forward(X, Y, mu, log_sigma, theta, log_delta, 
                row_batches, col_batches, feature_link_map)
    
    return sum(feature_loss_map(A, D))
end


function matrix_nlprior(X::AbstractMatrix, X_reg::Vector{T}) where T <: AbstractMatrix

    loss = BMFFloat(0.0f0)
    k = length(X_reg)
    for i=1:k
        loss += BMFFloat(0.5f0)*dot(X[i,:], X_reg[i] * X[i,:])
    end
    return loss
end


function ChainRules.rrule(::typeof(matrix_nlprior), X::AbstractMatrix, X_reg::Vector{T}) where T <: AbstractMatrix
    loss = BMFFloat(0.0f0)
    X_grad = zero(X)
    k = length(X_reg)
    for i=1:k
        X_grad[i,:] .= X_reg[i]*X[i,:]
        loss += BMFFloat(0.5f0)*dot(X[i,:], X_grad[i,:])
    end

    function matrix_nlprior_pullback(loss_bar)
        return ChainRules.NoTangent(), loss_bar.*X_grad, ChainRules.ZeroTangent()
    end

    return loss, matrix_nlprior_pullback
end


function neg_log_prior(X::AbstractMatrix, X_reg::Vector{T}, Y::AbstractMatrix, Y_reg::Vector{T}, 
                       mu::AbstractVector, mu_reg::AbstractMatrix, 
                       log_sigma::AbstractVector, log_sigma_reg::AbstractMatrix) where T <: AbstractMatrix
    
    loss = BMFFloat(0.0f0)

    loss += matrix_nlprior(X, X_reg)
    loss += matrix_nlprior(Y, Y_reg)

    loss += BMFFloat(0.5f0) * dot(mu, mu_reg * mu)
    loss += BMFFloat(0.5f0) * dot(log_sigma, log_sigma_reg * log_sigma)

    return loss

end


function neg_log_prob(X::AbstractMatrix, X_reg::Vector{T}, Y::AbstractMatrix, Y_reg::Vector{T}, 
                      mu::AbstractVector, mu_reg::BMFRegMat, 
                      log_sigma::AbstractVector, log_sigma_reg::BMFRegMat,
                      row_ranges::Vector{UnitRange}, col_ranges::Vector{UnitRange},
                      theta::AbstractMatrix, log_delta::AbstractMatrix, 
                      feature_link_map::ColBlockMap, feature_loss_map::ColBlockMap, 
                      D::AbstractMatrix) where T <: AbstractMatrix

    nlp = neg_log_likelihood(X, Y, mu, log_sigma, theta, log_delta,
                             row_ranges, col_ranges,
                             feature_link_map, feature_loss_map, D)
    nlp += neg_log_prior(X, X_reg, Y, Y_reg, mu, mu_reg, log_sigma, log_sigma_reg)

    return nlp

end


