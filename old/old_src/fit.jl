

import ScikitLearnBase: fit! 

function vprint(a...; verbose=false)
    if verbose
        print(string(a...))
    end
end

function bm_summary(bm::BatchMatrix)
    min_vals = [minimum(values(d)) for d in bm.values]
    max_vals = [maximum(values(d)) for d in bm.values]

    return string("MINS: ", min_vals, "\nMAXES: ", max_vals)
end

function fit!(model::BatchMatFacModel, A::AbstractMatrix;
              capacity::Integer=Integer(1e8), max_epochs::Number=1000, 
              lr::Real=0.01f0, abs_tol::Real=1e-9, rel_tol::Real=1e-6,
              verbose::Bool=false)

    # Move data to GPU
    A_gpu = CuArray(A)
    
    # Handle missing values.
    missing_value = BMFFloat(0.5)
    missing_data = isnan.(A_gpu)
    missing_mask = missing_data .* missing_value
    nonmissing = (!).(missing_data)
    A_gpu .*= nonmissing
    A_gpu .+= missing_mask 

    M = size(A,1)
    N = size(A,2)

    # Figure out the sizes of batches
    row_batch_size = div(capacity,N)
    col_batch_size = div(capacity,M)

    # Objects representing parameters and gradients
    params = ModelParams(model)
    gradients = zero(params)

    # AdaGrad update rule
    adagrad = adagrad_updater(params) 

    # Construct the feature noise models.
    # Column-specific link and loss functions
    unq_noises = unique(model.feature_noise_models) 
    feature_link_map = ColBlockMap([LINK_FUNCTION_MAP[ln] for ln in unq_noises],
                                   model.feature_noise_models)
    feature_loss_map = ColBlockAgg([LOSS_FUNCTION_MAP[ln] for ln in unq_noises], 
                                   model.feature_noise_models)

    # Some useful curries of the prior function
    curried_prior = (X, Y, mu, log_sigma, 
                     theta, log_delta) -> neg_log_prior(X, model.X_reg, 
                                                        Y, model.Y_reg, 
                                                        mu, model.mu_reg, 
                                                        log_sigma, 
                                                        model.log_sigma_reg,
                                                        theta, model.theta_reg,
                                                        log_delta, model.log_delta_reg)
    row_prior = (Y, mu, log_sigma, theta, log_delta) -> curried_prior(params.X, Y, 
                                                                      mu, log_sigma,
                                                                      theta, log_delta)
    col_prior = X -> curried_prior(X, params.Y, params.mu, params.log_sigma,
                                   params.theta, params.log_delta)


    prev_loss = Inf
    loss = Inf

    # For each epoch
    epoch = 1
    while epoch <= max_epochs
  
        # Zero out the gradients
        map!(zero, gradients, gradients)

        vprint("EPOCH ", epoch; verbose=verbose)

        # Updates for column-wise and block-wise parameters:
        # Y, mu, sigma, theta, delta
        # Iterate through minibatches of rows
        for row_batch in BatchIter(M, row_batch_size)
            # Select the corresponding rows of X, theta, delta
            batch_X = view(params.X, :, row_batch)
            batch_theta = params.theta[row_batch, 1:N]
            batch_log_delta = params.log_delta[row_batch, 1:N]

            # Select the corresponding rows of A; move to GPU
            batch_A = view(A_gpu, row_batch, :)
            batch_missing_mask = view(missing_mask, row_batch,:) 
            batch_nonmissing = view(nonmissing,row_batch,:)

            # Curry away the non-updated variables
            row_batch_log_lik = (Y, mu, log_sigma, 
                                 theta, log_delta) -> neg_log_likelihood(batch_X, Y, 
                                                                         mu, log_sigma, 
                                                                         theta, log_delta,
                                                                         feature_link_map, 
                                                                         feature_loss_map,
                                                                         batch_A,
                                                                         batch_missing_mask,
                                                                         batch_nonmissing)
            # Compute the batch's likelihood loss gradients 
            # w.r.t. Y, mu, sigma, theta, delta
            grad_Y, grad_mu, grad_log_sigma,
            grad_theta, grad_log_delta = gradient(row_batch_log_lik, 
                                                  params.Y, 
                                                  params.mu,
                                                  params.log_sigma, 
                                                  batch_theta,
                                                  batch_log_delta)

            # Accumulate these gradients into the full gradients
            gradients.Y .+= grad_Y
            gradients.mu .+= grad_mu
            gradients.log_sigma .+= grad_log_sigma
                        
            reindex!(grad_theta, row_batch.start, 1)
            add!(gradients.theta, grad_theta)
            reindex!(grad_log_delta, row_batch.start, 1)
            add!(gradients.log_delta, grad_log_delta)

        end

        # Compute prior gradients for Y, mu, sigma
        loss, (grad_Y, grad_mu, grad_log_sigma,
               grad_theta, grad_log_delta) = withgradient(row_prior, 
                                                          params.Y,
                                                          params.mu,
                                                          params.log_sigma,
                                                          params.theta,
                                                          params.log_delta)

        # Add prior gradients to full Y, mu, sigma gradients
        gradients.Y .+= grad_Y
        gradients.mu .+= grad_mu
        gradients.log_sigma .+= grad_log_sigma
        binop!(.+, gradients.theta.values, grad_theta.values)
        binop!(.+, gradients.log_delta.values, grad_log_delta.values)

        # Perform AdaGrad updates for Y, mu, sigma, theta, delta
        adagrad(params, gradients; lr=lr, 
                fields=[:Y, :mu, :log_sigma, :theta, :log_delta])


        # Updates for row-wise model parameters (i.e., X)
        # Iterate through minibatches of columns...
        for col_batch in BatchIter(N, col_batch_size)
            
            # Select the corresponding columns of Y, mu, sigma, theta, delta
            batch_Y = view(params.Y, :, col_batch)
            batch_mu = view(params.mu, col_batch)
            batch_log_sigma = view(params.log_sigma, col_batch)
            batch_theta = params.theta[1:M, col_batch]
            batch_log_delta = params.log_delta[1:M, col_batch]

            # Select the corresponding columns of A
            batch_A = view(A_gpu, :, col_batch)
            batch_missing_mask = view(missing_mask, :, col_batch) 
            batch_nonmissing = view(nonmissing, :, col_batch)
           
            batch_link_map = feature_link_map[col_batch]
            batch_loss_map = feature_loss_map[col_batch]

            # Curry away the non-updated variables
            col_batch_log_lik = X -> neg_log_likelihood(X, batch_Y, 
                                                        batch_mu, batch_log_sigma,
                                                        batch_theta, 
                                                        batch_log_delta,
                                                        batch_link_map, 
                                                        batch_loss_map, 
                                                        batch_A,
                                                        batch_missing_mask,
                                                        batch_nonmissing)
            
            # Compute the likelihood loss gradients w.r.t. X
            batch_loss, (grad_X,) = withgradient(col_batch_log_lik, params.X)
            loss += batch_loss

            # Add to the full gradient
            gradients.X .+= grad_X
        end
        
        # Compute prior gradient for X 
        prior_loss, (grad_X,) = withgradient(col_prior, params.X)
        loss += prior_loss

        # Add prior gradient to full X gradient
        gradients.X .+= grad_X

        # Perform AdaGrad update for X
        adagrad(params, gradients; lr=lr, fields=[:X])
        
        vprint("\tLoss: ", loss, "\n"; verbose=verbose)

        # Check termination conditions
        loss_diff = prev_loss - loss 
        if loss_diff < abs_tol
            break
        elseif loss_diff/loss < abs_tol
            break
        else
            prev_loss = loss
            epoch += 1
        end
    end

    set_params!(model, params)

    return #history
end


