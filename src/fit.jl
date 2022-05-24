

import ScikitLearnBase: fit!

AbstractOptimiser = Flux.Optimise.AbstractOptimiser

export fit!

function fit!(model::MatFacModel, D::AbstractMatrix;
              capacity::Integer=Integer(1e8), 
              max_epochs=1000, lr::Number=0.01,
              opt::Union{Nothing,AbstractOptimiser}=nothing,
              abs_tol::Number=1e-9, rel_tol::Number=1e-6,
              verbosity::Integer=1)
    
    #############################
    # Preparations
    #############################
    
    # Define a verbose print
    function vprint(a...; level=1)
        if verbosity >= level
            print(string(a...))
        end
    end

    M_D, N_D = size(D)
    K, M = size(model.X)
    N = size(model.Y,2)

    # Validate input size
    if (M != M_D)|(N != N_D) 
        throw(ArgumentError, string("Incompatible sizes! Data:", 
                                    size(D), 
                                    "; Model: ", (M,N)
                                   )
             )
    end

    inv_MN = 1.0/(M*N)
    inv_MK = 1.0/(M*K)
    inv_NK = 1.0/(N*K)

    col_batch_size = div(capacity,N)
    row_batch_size = div(capacity,M)

    # Prep the row and column transformations 
    col_layers = make_viewable(model.col_transform)
    row_layers = make_viewable(model.row_transform)

    # Define the likelihood function
    likelihood = (X,Y,
                  r_layers,
                  c_layers,
                  noise, D)-> inv_MN*invlinkloss(noise, 
                                                  c_layers(
                                                   r_layers(
                                                    transpose(X)*Y
                                                   )
                                                  ),
                                               D)

    # Prep the regularizers
    col_layer_regs = make_viewable(model.col_transform_reg)
    row_layer_regs = make_viewable(model.row_transform_reg)
   
    col_regularizer = (layers, reg) -> inv_NK*reg(layers)
    row_regularizer = (layers, reg) -> inv_MK*reg(layers)
    X_regularizer = (X, reg) -> inv_MK*reg(X)
    Y_regularizer = (Y, reg) -> inv_NK*reg(Y)

    # Initialize some objects to store gradients
    Y_grad = zero(model.Y)
    X_grad = zero(model.X)
    col_layer_grads = fmapstructure(tozero, rec_trainable(col_layers))
    row_layer_grads = fmapstructure(tozero, rec_trainable(row_layers))
    noise_model_grads = fmapstructure(tozero, rec_trainable(model.noise_model))

    # If no optimiser is provided, initialize
    # the default (an ADAGrad optimiser)
    if opt == nothing
        opt = Flux.Optimise.ADAGrad(lr)
    end

    # Track the loss
    prev_loss = Inf
    loss = Inf

    #############################
    # Main loop 
    #############################
    epoch = 1
    t_start = time()
    while epoch <= max_epochs

        vprint("Epoch ", epoch,":  ")

        loss = 0.0

        Y_grad .= 0
        col_layer_grads = fmap(tozero, col_layer_grads)
        noise_model_grads = fmap(tozero, noise_model_grads)

        ######################################
        # Iterate through the ROWS of data
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
            batchloss, grads = Zygote.withgradient(col_likelihood, 
                                                   model.Y,
                                                   col_layers_view,
                                                   model.noise_model)
    
            binop!(.+, Y_grad, grads[1])
            binop!(.+, col_layer_grads, grads[2])
            binop!(.+, noise_model_grads, grads[3])

            loss += batchloss

        end

        # Accumulate Y regularizer gradient
        regloss, Y_reg_grad = Zygote.withgradient(Y_regularizer, model.Y, model.Y_reg)
        loss += regloss
        binop!(.+, Y_grad, Y_reg_grad[1])
        
        # Update Y and the Y regularizer
        update!(opt, model.Y, Y_grad)
        update!(opt, model.Y_reg, Y_reg_grad[2])

        # Accumulate layer regularizer gradients
        regloss, reg_grads = Zygote.withgradient(col_regularizer, col_layers, col_layer_regs)
        loss += regloss
        binop!(.+, col_layer_grads, reg_grads[1])
        
        # Update layers and layer regularizers
        update!(opt, col_layers, col_layer_grads)
        update!(opt, col_layer_regs, reg_grads[2])

        # Update the noise model
        update!(opt, model.noise_model, noise_model_grads)


        X_grad .= 0
        row_layer_grads = fmap(tozero, row_layer_grads)

        ######################################
        # Iterate through the COLUMNS of data
        for col_batch in BatchIter(N, col_batch_size)
          
            D_v = view(D, :, col_batch)
            noise_view = view(model.noise_model, col_batch)
            Y_view = view(model.Y, :, col_batch)

            row_layers_view = view(row_layers, 1:M, col_batch)
            col_layers_view = view(col_layers, 1:M, col_batch)

            row_likelihood = (X, rl) -> likelihood(X, Y_view, 
                                                   rl, col_layers_view, 
                                                   noise_view, D_v)

            batchloss, g = Zygote.withgradient(row_likelihood, model.X, row_layers_view)
            binop!(.+, X_grad, g[1])
            binop!(.+, row_layer_grads, g[2])
        
            loss += batchloss
        end

        # Accumulate X regularizer gradients
        regloss, X_reg_grads = Zygote.withgradient(X_regularizer, model.X, model.X_reg)

        loss += regloss
        binop!(.+, X_grad, X_reg_grads[1])

        # Update X and the X regularizer
        update!(opt, model.X, X_grad)
        update!(opt, model.X_reg, X_reg_grads[2])

        # Accumulate the layer regularizer gradients
        regloss, reg_grads = Zygote.withgradient(row_regularizer, row_layers, row_layer_regs)
        loss += regloss
        binop!(.+, row_layer_grads, reg_grads[1])

        # Update the layers and regularizers
        update!(opt, row_layers, row_layer_grads)
        update!(opt, row_layer_regs, reg_grads[2])

        # Report the loss
        elapsed = time()-t_start
        vprint("Loss=",loss, " (", round(Int, elapsed), "s elapsed)\n")

        # Check termination conditions
        loss_diff = prev_loss - loss 
        if loss_diff < abs_tol
            vprint("Terminated: reached abs_tol<",abs_tol, "\n"; level=0)
            break
        elseif loss_diff/loss < abs_tol
            vprint("Terminated: reached rel_tol<",rel_tol, "\n"; level=0)
            break
        else
            prev_loss = loss
            epoch += 1
        end
    end
    if epoch >= max_epochs 
        vprint("Terminated: reached max_epochs=",max_epochs, "\n"; level=0)
    end

    #############################
    # Termination
    #############################

    return model
end


