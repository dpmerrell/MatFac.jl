
export total_loss


function likelihood_loss(model, D)

    A = forward(model)

    losses = model.feature_loss_map(A, D)

    return sum(losses)
end


function prior_loss(model)

    loss = 0.0f0
    
    # Priors on latent factors X, Y
    loss += row_reg_loss(model.X, model.X_reg) 
    loss += row_reg_loss(model.Y, model.Y_reg) 

    # Prior on mu
    loss += 0.5f0*dot(model.mu, model.mu_reg * model.mu)
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


function total_loss(model, D)
    return likelihood_loss(model, D) + prior_loss(model)    
end


