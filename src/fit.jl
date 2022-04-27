

import ScikitLearnBase: fit!


export fit!


function fit!(model::BatchMatFacModel, D::AbstractMatrix;
              capacity::Integer=Integer(1e8), 
              max_epochs=1000, lr=0.01, abs_tol=1e-9, rel_tol=1e-6,
              verbose=false)

    # Define a verbose-print
    function vprint(a...)
        if verbose
            print(string(a...))
        end
    end

    D = gpu(D)
    model = gpu(model)

    M_D, N_D = size(D)
    M = size(model.mp.X,2)
    N = size(model.mp.Y,2)

    # Validate input size
    if (M != M_D)|(N != N_D) 
        throw(ArgumentError, string("Incompatible sizes! Data:", 
                                    size(D), 
                                    "; Model: ", (M_model, N_model)
                                   )
             )
    end

    row_batch_size = div(capacity,N)
    col_batch_size = div(capacity,M)

    # Unpack regularizers
    col_reg = (X, X_reg) -> X_reg(X)
    col_reg_params = (model.mp.X, model.X_reg)

    row_reg = (Y, Y_reg,
               logsigma, logsigma_reg,
               mu, mu_reg,
               logdelta, logdelta_reg,
               theta, theta_reg) -> (Y_reg(Y)
                                     +logsigma_reg(logsigma)
                                     +mu_reg(mu)
                                     +logdelta_reg(logdelta)
                                     +theta_reg(theta)
                                    )
    
    row_reg_params = (model.mp.Y, model.Y_reg,
                      model.cscale.logsigma, model.logsigma_reg,
                      model.cshift.mu, model.mu_reg,
                      model.bscale.logdelta, model.logdelta_reg,
                      model.bshift.theta, model.theta_reg)
    

    # Initialize the optimizer
    opt = Flux.Optimise.ADAGrad(lr)

    # Track the loss
    prev_loss = Inf
    loss = Inf

    epoch = 1
    while epoch <= max_epochs

        vprint("Epoch ",epoch,":  ")

        loss = 0.0

        ######################################
        # Iterate through the ROWS of data
        for row_batch in BatchIter(M,row_batch_size)

            #println(string("\tROW BATCH, ", row_batch))

            D_v = view(D, row_batch, :)
            model_v = view(model, row_batch, :)
    
            # Define some sets of parameters for convenience
            row_loss_params = (model.mp.Y, model.cscale, model.cshift,
                               model_v.bscale, model_v.bshift,
                               model.noise_model)
          
            # Curry out the X for this batch;
            # we'll take the gradient for everything else.
            row_likelihood = (Y,
                              csc,
                              csh,
                              bsc,
                              bsh,
                              noise)-> invlinkloss(noise,
                                         bsh(
                                           bsc(
                                             csh(
                                               csc(
                                                 transpose(model_v.mp.X)*Y
                                                   )
                                                 )
                                               )
                                             ),
                                         D_v
                                         )

            # Update these parameters via log-likelihood gradient 
            batchloss, grads = Zygote.withgradient(row_likelihood, row_loss_params...)
            update!(opt, row_loss_params, grads)

            loss += batchloss
        end
        # Apply regularizer gradients to Y, sigma, mu, etc.
        regloss, reg_grads = Zygote.withgradient(row_reg, row_reg_params...)
        update!(opt, row_reg_params, reg_grads)
        loss += regloss


        ######################################
        # Iterate through the COLUMNS of data
        for col_batch in BatchIter(N,col_batch_size)
            
            #println(string("\tCOL BATCH, ", col_batch))

            D_v = view(D, :, col_batch)
            model_v = view(model, :, col_batch)
    
            col_loss_params = (model.mp.X,)

            col_likelihood = (X,) -> invlinkloss(model_v.noise_model,
                                       model_v.bshift(
                                         model_v.bscale(
                                           model_v.cshift(
                                             model_v.cscale(
                                               transpose(transpose(model_v.mp.Y)*X)
                                                            )
                                                          )
                                                        )
                                                      ),
                                                  D_v
                                                  )

            batchloss, grads = Zygote.withgradient(col_likelihood, col_loss_params...)
            update!(opt, col_loss_params, grads)
        
            loss += batchloss
        end

        # Apply regularizer gradients to X 
        regloss, reg_grads = Zygote.withgradient(col_reg, col_reg_params...)
        update!(opt, col_reg_params, reg_grads)
        loss += regloss

        # Report the loss
        vprint("Loss=",loss, "\n")

        # Check termination conditions
        loss_diff = prev_loss - loss 
        if loss_diff < abs_tol
            println(string("Terminated: reached abs_tol<",abs_tol))
            break
        elseif loss_diff/loss < abs_tol
            println(string("Terminated: reached rel_tol<",rel_tol))
            break
        else
            prev_loss = loss
            epoch += 1
        end
    end
    println(string("Terminated: reached max_epochs=",max_epochs))

    model = cpu(model)
end


