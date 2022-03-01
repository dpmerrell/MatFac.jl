


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
                            missing_mask::AbstractMatrix, nonmissing::AbstractMatrix)

    A = forward(X, Y, mu, log_sigma, theta, log_delta, feature_link_map)

    M, N = size(A)

    return sum(feature_loss_map(A, D, missing_mask, nonmissing)) / (M*N)
end


function matrix_nlprior(X::AbstractMatrix, X_reg::Vector{T}) where T <: AbstractMatrix

    loss = BMFFloat(0.0f0)
    K,M = size(X)
    for i=1:K
        loss += BMFFloat(0.5f0)*dot(X[i,:], X_reg[i] * X[i,:])
    end
    return loss/(K*M)
end


function ChainRules.rrule(::typeof(matrix_nlprior), X::AbstractMatrix, X_reg::Vector{T}) where T <: AbstractMatrix
    loss = BMFFloat(0.0f0)
    X_grad = zero(X)
    K,M = size(X)
    KM_inv = 1/(K*M)
    for i=1:K
        X_grad[i,:] .= X_reg[i]*X[i,:]
        loss += BMFFloat(0.5f0)*dot(X[i,:], X_grad[i,:])
    end

    loss *= KM_inv

    function matrix_nlprior_pullback(loss_bar)
        return ChainRules.NoTangent(), loss_bar.*X_grad.*KM_inv, ChainRulesCore.ZeroTangent()
    end

    return loss, matrix_nlprior_pullback
end


function batch_param_prior(param::BatchMatrix{T}, weight::Number) where T <: Number

    N_vec = Int[length(v) for v in param.values]
    means = map(mean, param.values)
    diffs = Vector{Vector{T}}(undef, length(means))
    for i=1:length(diffs)
        diffs[i] = param.values[i] .- means[i]
    end
    return 0.5 * weight * sum(dot(d,d)/N for (d,N) in zip(diffs,N_vec))
end


function ChainRules.rrule(::typeof(batch_param_prior), 
                          param::BatchMatrix{T}, weight::Number) where T <: Number
    
    means = map(mean, param.values)
    N_vec = Vector{Int}(undef, length(means))
    diffs = Vector{Vector{T}}(undef, length(means))
    for i=1:length(diffs)
        diffs[i] = param.values[i] .- means[i]
        N_vec[i] = length(param.values[i])
    end

    function batch_param_prior_pullback(loss_bar)
        param_bar = zero(param)
        param_bar.values = Vector{T}[d ./ N for (d, N) in zip(diffs, N_vec)]
        return ChainRules.NoTangent(), param_bar, ChainRules.NoTangent()
    end

    loss = 0.5 * weight * sum(dot(d,d)/N for (d,N) in zip(diffs,N_vec))
    return loss, batch_param_prior_pullback
end


function neg_log_prior(X::AbstractMatrix, X_reg::Vector{<:AbstractMatrix}, 
                       Y::AbstractMatrix, Y_reg::Vector{<:AbstractMatrix}, 
                       mu::AbstractVector, mu_reg::AbstractMatrix, 
                       log_sigma::AbstractVector, log_sigma_reg::AbstractMatrix,
                       theta::BatchMatrix, theta_reg::Number, 
                       log_delta::BatchMatrix, log_delta_reg::Number)

    loss = BMFFloat(0.0f0)

    loss += matrix_nlprior(X, X_reg)
    loss += matrix_nlprior(Y, Y_reg)

    _,N = size(Y)
    N_inv = 1/N

    loss += BMFFloat(0.5f0) * N_inv * dot(mu, mu_reg * mu) 
    loss += BMFFloat(0.5f0) * N_inv * dot(log_sigma, log_sigma_reg * log_sigma) 

    loss += batch_param_prior(theta, theta_reg)
    loss += batch_param_prior(log_delta, log_delta_reg)

    return loss

end


function neg_log_prob(X::AbstractMatrix, X_reg::Vector{T}, Y::AbstractMatrix, Y_reg::Vector{T}, 
                      mu::AbstractVector, mu_reg::BMFRegMat, 
                      log_sigma::AbstractVector, log_sigma_reg::BMFRegMat,
                      theta::BatchMatrix, theta_reg::Number, 
                      log_delta::BatchMatrix, log_delta_reg::Number, 
                      feature_link_map::ColBlockMap, feature_loss_map::ColBlockMap, 
                      D::AbstractMatrix, missing_mask::AbstractMatrix,
                      nonmissing::AbstractMatrix) where T <: AbstractMatrix

    nlp = neg_log_likelihood(X, Y, mu, log_sigma, theta, log_delta,
                             feature_link_map, feature_loss_map, D,
                             missing_mask, nonmissing)
    nlp += neg_log_prior(X, X_reg, Y, Y_reg, mu, mu_reg, log_sigma, log_sigma_reg,
                         theta, theta_reg, log_delta, log_delta_reg)

    return nlp

end


