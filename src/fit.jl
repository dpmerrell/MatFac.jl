

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

    model_d = gpu(model)
    D = gpu(D)

    M_D, N_D = size(D)
    K, M = size(model_d.mp.X)
    N = size(model_d.mp.Y,2)

    inv_MN = 1.0/(M*N)
    inv_MK = 1.0/(M*K)
    inv_NK = 1.0/(N*K)

    # Validate input size
    if (M != M_D)|(N != N_D) 
        throw(ArgumentError, string("Incompatible sizes! Data:", 
                                    size(D), 
                                    "; Model: ", (M,N)
                                   )
             )
    end

    row_batch_size = div(capacity,N)
    col_batch_size = div(capacity,M)

    # Unpack regularizers
    col_reg = (X, X_reg) -> inv_MK*X_reg(X)
    col_reg_params = (model_d.mp.X, model_d.X_reg)

    row_reg = (Y, Y_reg,
               logsigma, logsigma_reg,
               mu, mu_reg,
               logdelta, logdelta_reg,
               theta, theta_reg) -> inv_NK*(Y_reg(Y)
                                           +logsigma_reg(logsigma)
                                           +mu_reg(mu)
                                           +logdelta_reg(logdelta)
                                           +theta_reg(theta)
                                          )
    
    row_reg_params = (model_d.mp.Y, model_d.Y_reg,
                      model_d.cscale.logsigma, model_d.logsigma_reg,
                      model_d.cshift.mu, model_d.mu_reg,
                      model_d.bscale.logdelta, model_d.logdelta_reg,
                      model_d.bshift.theta, model_d.theta_reg)
    
    # Initialize the optimizer
    opt = Flux.Optimise.ADAGrad(lr)

    # Track the loss
    prev_loss = Inf
    loss = Inf

    epoch = 1
    t_start = time()
    while epoch <= max_epochs

        vprint("Epoch ",epoch,":  ")

        loss = 0.0

        ######################################
        # Iterate through the ROWS of data
        for row_batch in BatchIter(M,row_batch_size)

            D_v = view(D, row_batch, :)
            model_d_v = view(model_d, row_batch, :)
    
            # Define some sets of parameters for convenience
            row_loss_params = (model_d.mp.Y, model_d.cscale, model_d.cshift,
                               model_d_v.bscale, model_d_v.bshift,
                               model_d.noise_model)
          
            # Curry out the X for this batch;
            # we'll take the gradient for everything else.
            row_likelihood = (Y,
                              csc,
                              csh,
                              bsc,
                              bsh,
                              noise)-> inv_MN*invlinkloss(noise,
                                                           bsh(
                                                             bsc(
                                                               csh(
                                                                 csc(
                                                                   transpose(model_d_v.mp.X)*Y
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
           
            D_v = view(D, :, col_batch)
            model_d_v = view(model_d, :, col_batch)
    
            col_loss_params = (model_d.mp.X,)

            col_likelihood = (X,) -> inv_MN*invlinkloss(model_d_v.noise_model,
                                                         model_d_v.bshift(
                                                           model_d_v.bscale(
                                                             model_d_v.cshift(
                                                               model_d_v.cscale(
                                                                 transpose(transpose(model_d_v.mp.Y)*X)
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
        elapsed = time()-t_start
        vprint("Loss=",loss, " (", round(Int, elapsed), "s elapsed)\n")

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
    if epoch >= max_epochs 
        println(string("Terminated: reached max_epochs=",max_epochs))
    end

    # Make sure the model gets updated with the trained values
    for pname in propertynames(model_d)
        setproperty!(model, pname, cpu(getproperty(model_d, pname)))
    end
end


