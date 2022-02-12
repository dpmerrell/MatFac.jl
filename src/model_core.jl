


function forward(X::AbstractMatrix, Y::AbstractMatrix, mu::AbstractVector, log_sigma::AbstractVector, 
                 theta::BatchMatrix, log_delta::BatchMatrix, feature_link_map::ColBlockMap)

    sigma = exp.(log_sigma)
    delta = exp(log_delta)

    Z = transpose(X)*Y
    Z = Z .* transpose(sigma)
    Z = Z .+ transpose(mu)
    Z = Z * delta
    Z = Z + theta
    A = feature_link_map(Z)
    return A
end


function neg_log_likelihood(X::AbstractMatrix, Y::AbstractMatrix, 
                            mu::AbstractVector, log_sigma::AbstractVector, 
                            theta::BatchMatrix, log_delta::BatchMatrix,
                            feature_link_map::ColBlockMap, 
                            feature_loss_map::ColBlockAgg, D::AbstractMatrix,
                            missing_data::AbstractMatrix)

    A = forward(X, Y, mu, log_sigma, theta, log_delta, feature_link_map)

    return sum(feature_loss_map(A, D, missing_data))
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
        return ChainRules.NoTangent(), loss_bar.*X_grad, ChainRulesCore.ZeroTangent()
    end

    return loss, matrix_nlprior_pullback
end


function neg_log_prior(X::AbstractMatrix, X_reg::Vector{<:AbstractMatrix}, 
                       Y::AbstractMatrix, Y_reg::Vector{<:AbstractMatrix}, 
                       mu::AbstractVector, mu_reg::AbstractMatrix, 
                       log_sigma::AbstractVector, log_sigma_reg::AbstractMatrix)
    
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
                      theta::BatchMatrix, log_delta::BatchMatrix, 
                      feature_link_map::ColBlockMap, feature_loss_map::ColBlockMap, 
                      D::AbstractMatrix, missing_data::AbstractMatrix) where T <: AbstractMatrix

    nlp = neg_log_likelihood(X, Y, mu, log_sigma, theta, log_delta,
                             feature_link_map, feature_loss_map, D, missing_data)
    nlp += neg_log_prior(X, X_reg, Y, Y_reg, mu, mu_reg, log_sigma, log_sigma_reg)

    return nlp

end


