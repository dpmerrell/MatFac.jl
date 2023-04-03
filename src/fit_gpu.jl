

##########################################################################
# GPU-COMPATIBLE VERSION OF `fit!`
##########################################################################

function fit!(model::MatFacModel, D::CuMatrix;
              capacity::Integer=10^8, 
              max_epochs=1000, lr::Number=0.01,
              opt::Union{Nothing,AbstractOptimiser}=nothing,
              abs_tol::Number=1e-9, rel_tol::Number=1e-9,
              tol_max_iters::Number=3, 
              scale_column_losses=true,
              calibrate_losses=true,
              verbosity::Integer=1,
              print_prefix="",
              print_iter::Integer=10,
              keep_history=false,
              reg_relative_weighting=true,
              update_X=true,
              update_Y=true,
              update_row_layers=true,
              update_col_layers=true,
              update_noise_models=true,
              update_X_reg=true,
              update_Y_reg=true,
              update_row_layers_reg=true,
              update_col_layers_reg=true,
              t_start=nothing
              )
    
    #############################
    # Preparations
    #############################
    
    # Define a verbose print
    function vprint(a...; level=1, prefix=print_prefix)
        if verbosity >= level
            print(string(prefix, a...))
        end
    end

    # Validate input size
    M_D, N_D = size(D)
    K, M = size(model.X)
    N = size(model.Y,2)
    if (M != M_D)|(N != N_D) 
        throw(ArgumentError, string("Incompatible sizes! Data:", 
                                    size(D), 
                                    "; Model: ", (M,N)
                                   )
             )
    end

    col_batch_size = div(capacity,M)
    row_batch_size = div(capacity,N)
    
    # Reweight the column losses if necessary
    if scale_column_losses
        vprint("Re-weighting column losses...\n")
        rescale_column_losses!(model, D; verbosity=verbosity,
                                         prefix=string(print_prefix, "    "))
    end

    # Prep the row and column transformations 
    row_layers = make_viewable(model.row_transform)
    col_layers = make_viewable(model.col_transform)

    # Define the log-likelihood function
    likelihood = (X,Y,
                  r_layers,
                  c_layers,
                  noise, D)-> data_loss(X,Y,
                                        r_layers,
                                        c_layers,
                                        noise, D; 
                                        calibrate=calibrate_losses)

    # Prep the regularizers
    col_layer_regs = model.col_transform_reg
    row_layer_regs = model.row_transform_reg

    col_layer_regularizer = (layers, reg) -> model.lambda_col*reg(layers)
    row_layer_regularizer = (layers, reg) -> model.lambda_row*reg(layers)

    # "Relative weighting" rescales the regularization by
    # the size of the parameter. This makes it possible to use the 
    # same regularizer on different problem sizes
    # without dramatically changing its effect.
    NdK = N/K
    MdK = M/K
    if reg_relative_weighting
        X_regularizer = (X, reg) -> NdK*model.lambda_X*reg(X)
        Y_regularizer = (Y, reg) -> MdK*model.lambda_Y*reg(Y)
    else
        X_regularizer = (X, reg) -> model.lambda_X*reg(X)
        Y_regularizer = (Y, reg) -> model.lambda_Y*reg(Y)
    end

    # If no optimiser is provided, initialize
    # the default (an AdaGrad optimiser)
    if opt == nothing
        opt = Flux.Optimise.AdaGrad(lr)
    end

    # If required, construct an object to store
    # training history
    hist = nothing
    if keep_history
        hist = Dict() 
    end

    # Track the loss
    best_loss = Inf
    prev_loss = Inf
    loss = Inf

    # Define objects to hold gradients
    X_grad = zero(model.X)
    Y_grad = zero(model.Y)
    row_layer_grad = deepcopy(rec_trainable(model.row_transform))
    col_layer_grad = deepcopy(rec_trainable(model.col_transform))
    noise_model_grad = deepcopy(rec_trainable(model.noise_model))

    #############################
    # Main loop 
    #############################
    epoch = 1
    if t_start == nothing
        t_start = time()
    end
    tol_iters = 0
    #vprint("Fitting model parameters...\n"; prefix=print_prefix)
    while epoch <= max_epochs

        if (epoch % print_iter) == 0
            vprint("Epoch ", epoch,":  ")
        end

        # Set losses to zero
        loss = 0.0
        d_loss = 0.0

        # Set gradients to zero
        zero_out!(X_grad) 
        zero_out!(Y_grad)
        zero_out!(row_layer_grad)
        zero_out!(col_layer_grad)
        zero_out!(noise_model_grad)

        ######################################
        # Iterate through the ROWS of data
        if (update_Y | update_col_layers)
            for row_batch in BatchIter(M, row_batch_size)

                X_view = view(model.X, :, row_batch)
                D_v = view(D, row_batch, :)
                row_layers_view = view(row_layers, row_batch, 1:N)
                col_layers_view = view(col_layers, row_batch, 1:N)

                # Define the likelihood for this batch
                col_likelihood = (Y, cl, noise) -> likelihood(X_view, Y, 
                                                              row_layers_view, cl, 
                                                              noise, D_v)

                # Accumulate the gradient
                batch_loss, grads = Zygote.withgradient(col_likelihood, 
                                                        model.Y,
                                                        col_layers_view,
                                                        model.noise_model)
                
                # Update the data loss *only* if we won't
                # iterate in the other direction.
                if !(update_X | update_row_layers)
                    d_loss += batch_loss
                end

                # Accumulate gradients
                if update_Y
                    binop!(.+, Y_grad, grads[1])
                end
                if update_col_layers
                    binop!(.+, col_layer_grad, grads[2])
                end
                if update_noise_models
                    binop!(.+, noise_model_grad, grads[3])
                end
            end
        end

        Y_reg_loss = 0.0
        if (update_Y | update_Y_reg)
            Y_reg_loss, Y_reg_grad = Zygote.withgradient(Y_regularizer, model.Y, model.Y_reg)
            if update_Y
                # Add Y's regularizer gradient
                binop!(.+, Y_grad, Y_reg_grad[1])
                
                # Update Y with the accumulated gradient
                update!(opt, model.Y, Y_grad)
            end
            if update_Y_reg
                # Update the Y regularizer
                update!(opt, model.Y_reg, Y_reg_grad[2])
            end
        end

        col_layer_reg_loss = 0.0
        if (update_col_layers | update_col_layers_reg)
            col_layer_reg_loss, reg_grads = Zygote.withgradient(col_layer_regularizer, col_layers, col_layer_regs)
            if update_col_layers
                # Accumulate column layers' regularizer gradients.
                # Then update the column layers with the accumulated gradient 
                binop!(.+, col_layer_grad, reg_grads[1])
                update!(opt, model.col_transform, col_layer_grad)
            end
            if update_col_layers_reg
                # Update column layer regularizer's parameters
                update!(opt, col_layer_regs, reg_grads[2])
            end
        end
        if update_noise_models 
            # Update the noise model
            update!(opt, model.noise_model, noise_model_grad)
        end

        ######################################
        # Iterate through the COLUMNS of data
        if (update_X | update_row_layers)
            for col_batch in BatchIter(N, col_batch_size)
               
                # Define batch views of data and model parameters 
                D_v = view(D, :, col_batch)
                noise_view = view(model.noise_model, col_batch)
                Y_view = view(model.Y, :, col_batch)

                row_layers_view = view(row_layers, 1:M, col_batch)
                col_layers_view = view(col_layers, 1:M, col_batch)

                row_likelihood = (X, rl) -> likelihood(X, Y_view, 
                                                       rl, col_layers_view, 
                                                       noise_view, D_v)
                # Compute loss and gradients
                batchloss, g = Zygote.withgradient(row_likelihood, model.X, row_layers_view)
                d_loss += batchloss

                # Accumulate gradients
                if update_X
                    binop!(.+, X_grad, g[1])
                end
                if update_row_layers
                    binop!(.+, row_layer_grad, g[2]) 
                end
            end
        end

        # Update X via regularizer gradients
        X_reg_loss = 0.0
        if (update_X | update_X_reg)
            X_reg_loss, X_reg_grads = Zygote.withgradient(X_regularizer, model.X, model.X_reg)
            if update_X
                binop!(.+, X_grad, X_reg_grads[1])
        
                update!(opt, model.X, X_grad)
            end
            if update_X_reg
                # Update the X regularizer
                update!(opt, model.X_reg, X_reg_grads[2])
            end
        end

        row_layer_reg_loss = 0.0
        if (update_row_layers | update_row_layers_reg)
            row_layer_reg_loss, reg_grads = Zygote.withgradient(row_layer_regularizer, row_layers, row_layer_regs)
            if update_row_layers
                # Update the row layers via regularizer gradients
                binop!(.+, row_layer_grad, reg_grads[1])

                update!(opt, row_layers, row_layer_grad)
            end
            if update_row_layers_reg
                # Update the row layer regularizer's parameters
                update!(opt, row_layer_regs, reg_grads[2])
            end
        end

        # Sum the loss components (halve the data loss
        # to account for row-pass and col-pass)
        loss = (d_loss + row_layer_reg_loss 
                + col_layer_reg_loss + X_reg_loss + Y_reg_loss)
        elapsed = time() - t_start

        history!(hist; data_loss=d_loss, 
                       row_layer_reg_loss=row_layer_reg_loss,
                       col_layer_reg_loss=col_layer_reg_loss,
                       X_reg_loss=X_reg_loss,
                       Y_reg_loss=Y_reg_loss,
                       total_loss=loss,
                       elapsed_time=elapsed)

        # Report the loss
        if (epoch % print_iter) == 0
            vprint("Loss=",loss, " (", round(Int, elapsed), "s elapsed)\n"; prefix="")
        end

        # Check termination conditions
        loss_diff = prev_loss - loss
        epoch += 1 
        if loss <= best_loss # We're at least as good as the best loss!
            best_loss = loss
            if loss_diff < abs_tol
                tol_iters += 1
                vprint("termination counter: ", tol_iters,"/",tol_max_iters ,"; abs_tol<",abs_tol, "\n"; level=1)
            elseif loss_diff/abs(loss) < abs_tol
                tol_iters += 1
                vprint("termination counter: ", tol_iters,"/",tol_max_iters ,"; rel_tol<",rel_tol, "\n"; level=1)
            else
                tol_iters = 0
            end
        else # Loss is higher than the best loss!
            if loss_diff < 0 # Loss is increase! 
                tol_iters += 1
                vprint("termination counter: ", tol_iters,"/",tol_max_iters ,"; loss increase\n"; level=1)
            end
        end
        prev_loss = loss

        if tol_iters >= tol_max_iters
            vprint("Reached max termination counter (", tol_max_iters, "). Terminating\n"; level=1)
            break
        end

    end
    if epoch >= max_epochs 
        vprint("Terminated: reached max_epochs=", max_epochs, "\n"; level=1)
    end

    #############################
    # Termination
    #############################
    finalize_history!(hist)

    return hist
end
