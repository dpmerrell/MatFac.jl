


function forward(X::BMFMat, Y::BMFMat, mu::BMFVec, sigma::BMFVec, 
                 theta::BlockMatrix, delta::BlockMatrix, 
                 feature_link_map::ColBlockMap)

    Z = transpose(X)*Y
    Z = Z .* transpose(sigma)
    Z = Z .+ transpose(mu)
    Z = Z * delta
    Z = Z + theta
    A = feature_link_map(Z)
    return A
end


function neg_log_likelihood(X::BMFMat, Y::BMFMat, mu::BMFVec, sigma::BMFVec, 
                            theta::BlockMatrix, delta::BlockMatrix, 
                            feature_link_map::ColBlockMap, 
                            feature_loss_map::ColBlockAgg, D::BMFData)

    A = forward(X, Y, mu, sigma, theta, delta, feature_link_map)
    
    return sum(feature_loss_map(A, D))
end


function matrix_nlprior(X::BMFMat, X_reg::BMFRegMat)

    loss = BMFFloat(0.0f0)
    k = length(X_reg)
    for i=1:k
        loss += BMFFloat(0.5f0)*dot(X[i,:], X_reg[i] * X[i,:])
    end
    return loss
end


function ChainRules.rrule(::typeof(matrix_nlprior), X::BMFMat, X_reg::BMFRegMat)
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


function neg_log_prior(X::BMFMat, X_reg::BMFRegMat, Y::BMFMat, Y_reg::BMFRegMat, 
                       mu::BMFVec, mu_reg::BMFRegMat, 
                       sigma::BMFVec, sigma_reg::BMFRegMat)
    
    nlp = BMFFloat(0.0f0)

    loss += matrix_nlprior(X, X_reg)
    loss += matrix_nlprior(Y, Y_reg)

    loss += BMFFloat(0.5f0) * dot(mu, mu_reg * mu)

    return loss

end


function neg_log_prob(X::BMFMat, X_reg::BMFRegMat, Y::BMFMat, Y_reg::BMFRegMat, 
                      mu::BMFVec, mu_reg::BMFRegMat, 
                      sigma::BMFVec, sigma_reg::BMFRegMat,
                      theta::BlockMatrix, delta::BlockMatrix, 
                      feature_link_map::ColBlockMap, feature_loss_map::ColBlockMap, 
                      D::BMFData)

    nlp = neg_log_likelihood(X, Y, mu, sigma, theta, delta,
                             feature_link_map, feature_loss_map, D)
    nlp += neg_log_prior(X, X_reg, Y, Y_reg, mu, mu_reg, sigma, sigma_reg)

    return nlp

end


