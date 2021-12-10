

function likelihood_loss(model, D)

    A = forward(model)

    losses = model.loss_function_map(A, D)

    return sum(losses)
end


function prior_loss(model)

    loss = 0.0
    K = length(model.X_reg)
    
    # Priors on latent factors X, Y
    for i=1:K
        loss += 0.5*transpose(model.X[i,:])*(model.X_reg[i] * model.X[i,:]) 
        loss += 0.5*transpose(model.Y[i,:])*(model.Y_reg[i] * model.Y[i,:])
    end

    # Prior on mu
    loss += transpose(model.mu)*(model.mu_reg * model.mu)

    return loss
end


function total_loss(model, D)
    return likelihood_loss(model, D) + prior_loss(model)    
end


####################################
# LOSS FUNCTIONS
####################################

function quad_loss(A, D)
    return (A .- D).^2
end


function logistic_loss(A, D)
    return -( D.*log.(A) .+ (1.0 .- D).*log.( 1.0 .- A) )
end


function poisson_loss(A, D)
    return A - D.*log.(A)  
end


LOSS_FUNCTION_MAP = Dict("normal"=>quad_loss,
                         "logistic"=>logistic_loss,
                         "poisson"=>poisson_loss
                         )
