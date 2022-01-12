

function fit!(model::BatchMatFacModel, A::AbstractMatrix;
              batch_size::Integer=1000, max_epochs::Integer=1000, 
              lr::Real=0.01f0, abs_tol::Real=1e-9, rel_tol::Real=1e-9)


    # Initialize empty history
    history = Float64[]

    M = size(A,1)

    # Curry away the column block operations
    model_ll = (X,Y,mu,sigma,theta,delta,A) -> neg_log_likelihood(X,Y,mu,sigma,theta,delta,
                                                                  model.feature_link_map,
                                                                  model.feature_loss_map,
                                                                  A)

    # For each epoch
    for epoch=1:max_epochs
    
        epoch_loss = 0.0
        # For each minibatch
        for batch in BatchIter(M)

            # Select the data and model parameters
            # pertaining to this minibatch
            batch_A = view(A, batch, :)
            batch_X = view(model.X, :, batch)
            batch_delta = model.delta[batch]
            batch_theta = model.theta[batch]
   
            # Curry away the data in the log-likelihood 
            curried_ll = (X,Y,mu,sigma,
                          theta,delta) -> model_ll(X,Y,mu,sigma,
                                                   theta,delta, batch_A)
            
            # Compute the batch's likelihood gradients w.r.t.
            # X, Y, sigma, mu, 
            grad_X, grad_Y, grad_mu, 
            grad_sigma, grad_theta, grad_delta = gradient(curried_ll, batch_X, model.Y, 
                                                                      model.mu, model.sigma, 
                                                                      batch_theta, batch_delta) 

            # update the model parameters X, Y, mu, sigma, theta, delta
            batch_X .-= lr.*grad_X
            Y .-= lr.*grad_Y
            model.mu .-= lr.*grad_mu
            model.sigma .-= lr.*grad_sigma
            
            grad_theta.values .*= lr
            add!(model.theta, batch, grad_theta)

            grad_delta.values .*= lr
            add!(model.delta, batch, grad_delta)

        end

        push!(history, epoch_loss)
    end

    return history
end
