

function print_if_nan(arr, name)
    if !all(map(!, isnan.(arr)))
        println(string(name, "\t", arr)) 
    end
end


function fit!(model::BatchMatFacModel, A::AbstractMatrix;
              capacity::Integer=Integer(1e8), max_epochs::Integer=1000, 
              lr::Real=0.01f0, abs_tol::Real=1e-9, rel_tol::Real=1e-9)


    M = size(A,1)
    N = size(A,2)

    # Figure out the sizes of batches
    row_batch_size = div(capacity,N)
    col_batch_size = div(capacity,M)


    # Curry away the regularizers from the 
    # prior function
    curried_prior = (X, Y, mu, log_sigma) -> neg_log_prior(X, model.X_reg, 
                                                           Y, model.Y_reg, 
                                                           mu, model.mu_reg, 
                                                           log_sigma, 
                                                           model.log_sigma_reg)


    sample_group_ranges = ids_to_ranges(model.sample_group_ids)
    feature_group_ranges = ids_to_ranges(model.feature_group_ids)

    theta_block_matrix = BlockMatrix(view(model.theta,:,:), 
                                     sample_group_ranges,
                                     feature_group_ranges)
    log_delta_block_matrix = BlockMatrix(view(model.log_delta,:,:), 
                                         sample_group_ranges,
                                         feature_group_ranges)

    # For each epoch
    for epoch=1:max_epochs
   

        println(string("EPOCH ", epoch))
        #println(model)

        # Updates for column-wise and block-wise parameters:
        # Y, mu, sigma, theta, delta
        # Iterate through minibatches of rows
        for row_batch in BatchIter(M, row_batch_size)

            batch_frac = (row_batch.stop - row_batch.start+1)/M
            
            # figure out the blocks of rows affected by this minibatch
            #r_block_ranges, r_block_min, r_block_max = subset_ranges(sample_group_ranges, row_batch) 
            #_, r_block_min, r_block_max = subset_ranges(sample_group_ranges, row_batch) 

            # Select the corresponding rows of X, theta, delta
            batch_X = view(model.X, :, row_batch)
            #batch_theta = model.theta[r_block_min:r_block_max, :]
            #batch_log_delta = model.log_delta[r_block_min:r_block_max, :]
            batch_theta = theta_block_matrix[row_batch, 1:N]
            batch_log_delta = log_delta_block_matrix[row_batch, 1:N]

            # Select the corresponding rows of A
            batch_A = CuArray(view(A, row_batch, :))

            # Curry away the non-updated variables
            row_batch_log_lik = (Y, mu, log_sigma, 
                                 theta, log_delta) -> neg_log_likelihood(batch_X, Y, 
                                                                         mu, log_sigma, 
                                                                         theta, log_delta,
                                                                         batch_theta.row_ranges,
                                                                         batch_theta.col_ranges,
                                                                         model.feature_link_map, 
                                                                         model.feature_loss_map,
                                                                         batch_A)

            # Compute the likelihood loss gradients w.r.t. Y, mu, sigma, theta, delta
            grad_Y, grad_mu, grad_log_sigma,
            grad_theta, grad_log_delta = gradient(row_batch_log_lik, model.Y, 
                                                                     model.mu,
                                                                     model.log_sigma, 
                                                                     batch_theta.values,
                                                                     batch_log_delta.values)
            print_if_nan(grad_Y, "GRAD_Y") 
            print_if_nan(grad_mu, "GRAD_MU") 
            print_if_nan(grad_log_sigma, "GRAD_LOG_SIGMA") 
            print_if_nan(grad_theta, "GRAD_THETA") 
            print_if_nan(grad_log_delta, "GRAD_LOG_DELTA") 

            # Updates for the batch parameters
            grad_theta_block_matrix = BlockMatrix(grad_theta, batch_theta.row_ranges,
                                                              batch_theta.col_ranges)
            grad_theta_block_matrix.values .*= -lr
            row_add!(theta_block_matrix, row_batch, grad_theta_block_matrix) 

            grad_log_delta_block_matrix = BlockMatrix(grad_log_delta, batch_theta.row_ranges,
                                                                      batch_theta.col_ranges)
            grad_log_delta_block_matrix.values .*= -lr
            row_add!(log_delta_block_matrix, row_batch, grad_log_delta_block_matrix)

            #model.theta[r_block_min:r_block_max, :] .-= lr .* grad_theta
            #model.log_delta[r_block_min:r_block_max, :] .-= lr .* grad_log_delta

            # Likelihood-gradient updates for the other parameters
            model.Y .-= lr .* grad_Y
            model.mu .-= lr .* grad_mu
            model.log_sigma .-= lr .* grad_log_sigma

            print_if_nan(model.Y, "model_Y") 
            print_if_nan(model.mu, "model_MU") 
            print_if_nan(model.log_sigma, "model_LOG_SIGMA") 
            print_if_nan(model.theta, "model_THETA") 
            print_if_nan(model.log_delta, "model_LOG_DELTA")

            # Compute the prior loss gradients w.r.t. Y, mu, and sigma
            # (multiply by fraction of rows in minibatch (row_batch.stop-row_batch.start+1)/M)
            row_batch_prior = (Y, mu, log_sigma) -> curried_prior(model.X, Y, mu, log_sigma)
            grad_Y, grad_mu, grad_log_sigma = gradient(row_batch_prior, 
                                                       model.Y,
                                                       model.mu,
                                                       model.log_sigma)

            # Prior-gradient updates 
            model.Y .-= (batch_frac*lr) .* grad_Y
            model.mu .-= (batch_frac*lr) .* grad_mu
            model.log_sigma .-= (batch_frac*lr) .* grad_log_sigma
        
            println(string("\t", row_batch, " Y, MU, SIGMA, THETA, DELTA UPDATES FINISHED"))
        end


        # Updates for row-wise model parameters (i.e., X)
        # Iterate through minibatches of columns...
        for col_batch in BatchIter(N, col_batch_size)
            
            batch_frac = (col_batch.stop - col_batch.start+1)/N
            
            # figure out the blocks of columns affected by this minibatch
            #c_block_ranges, c_block_min, c_block_max = subset_ranges(feature_group_ranges, col_batch) 

            # Select the corresponding columns of Y, mu, sigma, theta, delta
            batch_Y = view(model.Y, :, col_batch)
            batch_mu = view(model.mu, col_batch)
            batch_log_sigma = view(model.log_sigma, col_batch)
            #batch_theta = model.theta[:,c_block_min:c_block_max]
            #batch_log_delta = model.log_delta[:,c_block_min:c_block_max]
            batch_theta = theta_block_matrix[1:M, col_batch]
            batch_log_delta = log_delta_block_matrix[1:M, col_batch]

            # Select the corresponding columns of A
            batch_A = CuArray(view(A, :, col_batch))
           
            batch_link_map = model.feature_link_map[col_batch]
            batch_loss_map = model.feature_loss_map[col_batch]

            # Curry away the non-updated variables
            col_batch_log_lik = X -> neg_log_likelihood(X, batch_Y, 
                                                        batch_mu, batch_log_sigma,
                                                        batch_theta.values, 
                                                        batch_log_delta.values,
                                                        batch_theta.row_ranges,
                                                        batch_theta.col_ranges,
                                                        batch_link_map, 
                                                        batch_loss_map, 
                                                        batch_A)
            
            # Compute the likelihood loss gradients w.r.t. X
            grad_X = gradient(col_batch_log_lik, model.X)[1]
            print_if_nan(grad_X, "GRAD_X") 

            model.X .-= lr .* grad_X
            print_if_nan(model.X, "model_X") 

            # Compute the prior loss gradients w.r.t. X
            # (multiply by fraction of columns in minibatch, (col_batch.stop-col_batch.start+1)/N)
            col_batch_prior = X -> curried_prior(X, model.Y, model.mu, model.log_sigma)
            grad_X = gradient(col_batch_prior, model.X)[1]

            # Update X according to the update rule
            model.X .-= (batch_frac * lr) .* grad_X
            
            println(string("\t", col_batch, " X UPDATES FINISHED"))

        end

        #push!(history, epoch_loss)
    end

    return #history
end


