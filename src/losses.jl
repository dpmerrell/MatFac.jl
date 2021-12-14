
export total_loss


function likelihood_loss(model, D)

    A = forward(model)

    losses = model.feature_loss_map(A, D)

    return sum(losses)
end


function prior_loss(model)

    loss = Float32(0.0)
    
    # Priors on latent factors X, Y
    loss += row_reg_loss(model.X, model.X_reg) 
    loss += row_reg_loss(model.Y, model.Y_reg) 

    # Prior on mu
    loss += Float32(0.5)*dot(model.mu, model.mu_reg * model.mu)
    return loss
end


function row_reg_loss(matrix, reg_mats)
    loss = 0.0f0
    k = length(reg_mats)
    for i=1:k
        loss += 0.5f0*dot(matrix[i,:], reg_mats[i] * matrix[i,:])
    end
    return loss
end


function ChainRules.rrule(::typeof(row_reg_loss), matrix, reg_mats)
    loss = 0.0f0
    matrix_grad = zero(matrix)
    k = length(reg_mats)
    for i=1:k
        matrix_grad[i,:] .= reg_mats[i]*matrix[i,:]
        loss += 0.5f0*dot(matrix[i,:], matrix_grad[i,:])
    end

    function row_reg_loss_pullback(loss_bar)
        return ChainRules.NoTangent(), loss_bar.*matrix_grad, ChainRules.ZeroTangent()
    end

    return loss, row_reg_loss_pullback
end


function total_loss(model, D)
    return likelihood_loss(model, D) + prior_loss(model)    
end


####################################
# LOSS FUNCTIONS
####################################

function quad_loss(A::AbstractArray, D::AbstractArray)
    return sum((A .- D).^2)
end

function ChainRules.rrule(::typeof(quad_loss), A, D)
    
    diff = A .- D

    function quad_loss_pullback(loss_bar)
        return ChainRules.NoTangent(), loss_bar.*diff, ChainRules.NoTangent() 
    end

    return sum(diff.^2), quad_loss_pullback 
end


function logistic_loss(A::AbstractArray, D::AbstractArray)
    return -sum( D.*log.(A) .+ (1 .- D).*log.( 1 .- A) )
end


function ChainRules.rrule(::typeof(logistic_loss), A, D)
    loss = logistic_loss(A,D)

    function logistic_loss_pullback(loss_bar)
        A_bar = loss_bar .* (D./A .+ (1 .- D)./(1 .- D))
        return ChainRules.NoTangent(), A_bar, ChainRules.NoTangent()
    end
    return loss, logistic_loss_pullback 
end


function poisson_loss(A::AbstractArray, D::AbstractArray)
    return sum(A - D.*log.(A))
end


function ChainRules.rrule(::typeof(poisson_loss), A, D)
    
    loss = poisson_loss(A, D)

    function poisson_loss_pullback(loss_bar)
        A_bar = loss_bar .* (1 .- D ./ A)
        return ChainRules.NoTangent(), A_bar, ChainRules.NoTangent() 
    end

    return loss, poisson_loss_pullback 
end


LOSS_FUNCTION_MAP = Dict("normal"=>quad_loss,
                         "logistic"=>logistic_loss,
                         "poisson"=>poisson_loss
                         )
