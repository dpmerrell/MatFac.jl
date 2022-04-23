

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
    col_reg = (X,) -> model.X_reg(X)
    row_reg = (Y,logsigma,mu,
               logdelta,theta) -> (model.Y_reg(Y)
                                   +model.logsigma_reg(logsigma)
                                   +model.mu_reg(mu)
                                   +model.logdelta_reg(logdelta)
                                   +model.theta_reg(theta)
                                  )
    
    row_reg_params = (model.mp.Y, model.cscale.logsigma,
                                  model.cshift.mu,
                                  model.bscale.logdelta,
                                  model.bshift.theta)
    col_reg_params = (model.mp.X,)
    

    # Initialize the optimizer
    opt = Flux.Optimise.ADAGrad(lr)

    # Track the loss
    prev_loss = Inf
    loss = Inf

    epoch = 1
    while epoch <= max_epochs

        vprint("Epoch ",epoch,":\t")

        loss = 0.0

        ######################################
        # Iterate through the ROWS of data
        for row_batch in BatchIter(M,row_batch_size)
            
            D_v = view(D, row_batch, :)
            model_v = view(model, row_batch, :)
    
            # Define some sets of parameters for convenience
            row_loss_params = (model_v.mp.Y, model_v.cscale, model_v.cshift,
                               model_v.bscale, model_v.bshift,
                               model_v.noise_models)
          
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

            #sync()
        end
        # Also apply regularizer updates
        regloss, reg_grads = Zygote.withgradient(row_reg, row_reg_params...)
        update!(opt, row_reg_params, reg_grads)
        loss += regloss


        ######################################
        # Iterate through the COLUMNS of data
        for col_batch in BatchIter(N,col_batch_size)
            D_v = view(D, :, col_batch)
            model_v = view(model, :, col_batch)
    
            col_loss_params = (model_v.mp.X,)

            col_likelihood = (X,) -> invlinkloss(model_v.noise,
                                       model_v.bshift(
                                         model_v.bscale(
                                           model_v.cshift(
                                             model_v.cscale(
                                               transpose(X)*model_v.mp.Y
                                                            )
                                                          )
                                                        )
                                                      ),
                                                  D_v
                                                  )

            batchloss, grads = Zygote.withgradient(col_likelihood, col_loss_params...)
            update!(opt, col_loss_params, grads)
            loss += batchloss

            #sync()
        end

        # 
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

    cpu(model)
end


