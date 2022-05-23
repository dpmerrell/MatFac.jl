

import ScikitLearnBase: fit!


export fit!


function fit!(model::BatchMatFacModel, D::AbstractMatrix;
              capacity::Integer=Integer(1e8), 
              max_epochs=1000, lr=0.01, abs_tol=1e-9, rel_tol=1e-6,
              verbose=false)
    
    #############################
    # Preparations
    #############################
    
    # Define a verbose-print
    function vprint(a...)
        if verbose
            print(string(a...))
        end
    end

    model_d = gpu(model)
    D = gpu(D)

    M_D, N_D = size(D)
    K, M = size(model_d.X)
    N = size(model_d.Y,2)

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

    col_batch_size = div(capacity,N)
    row_batch_size = div(capacity,M)

    # Prep the row and column transformations 
    col_layers = make_viewable(model_d.col_transform)
    row_layers = make_viewable(model_d.row_transform)

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
    col_layer_regs = make_viewable(model_d.col_transform_reg)
    row_layer_regs = make_viewable(model_d.row_transform_reg)
   
    col_regularizer = (layers, reg) -> inv_NK*reg(layers)
    row_regularizer = (layers, reg) -> inv_MK*reg(layers)
    X_regularizer = (X, reg) -> inv_MK*reg(X)
    Y_regularizer = (Y, reg) -> inv_NK*reg(Y)

    # Initialize some objects to store gradients
    Y_grad = zero(model_d.Y)
    X_grad = zero(model_d.X)
    col_layer_grads = fmapstructure(zero, rec_trainable(col_layers))
    row_layer_grads = fmapstructure(zero, rec_trainable(row_layers))
    noise_model_grads = fmap(zero, rec_trainable(model_d.noise_model))

    # Initialize the optimizer
    opt = Flux.Optimise.ADAGrad(lr)

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
        col_layer_grads = fmap(zero, col_layer_grads)
        noise_model_grads = fmap(zero, noise_model_grads)

        ######################################
        # Iterate through the ROWS of data
        for row_batch in BatchIter(M, row_batch_size)

            X_view = view(model_d.X, :, row_batch)
            D_v = view(D, row_batch, :)
            row_layers_view = view(row_layers, row_batch, 1:N)

            # Define the likelihood for this batch
            col_likelihood = (Y, cl, noise) -> likelihood(X_view, Y, 
                                                          row_layers_view, cl, 
                                                          noise, D_v)

            # Accumulate the gradient 
            batchloss, grads = Zygote.withgradient(col_likelihood, 
                                                   model_d.Y,
                                                   col_layers,
                                                   model_d.noise_model)
            
            binop!(.+, Y_grad, grads[1])
            binop!(.+, col_layer_grads, grads[2])
            binop!(.+, noise_model_grads, grads[3])

            loss += batchloss

        end

        # Accumulate Y regularizer gradient
        regloss, Y_reg_grad = Zygote.withgradient(Y_regularizer, model_d.Y, model_d.Y_reg)
        loss += regloss
        binop!(.+, Y_grad, Y_reg_grad[1])
        
        # Update Y and the Y regularizer
        update!(opt, model_d.Y, Y_grad)
        update!(opt, model_d.Y_reg, Y_reg_grad[2])

        # Accumulate layer regularizer gradients
        regloss, reg_grads = Zygote.withgradient(col_regularizer, col_layers, col_layer_regs)
        loss += regloss
        binop!(.+, col_layer_grads, reg_grads[1])
        
        # Update layers and layer regularizers
        update!(opt, col_layers, col_layer_grads)
        update!(opt, col_layer_regs, reg_grads[2])

        X_grad .= 0
        row_layer_grads = fmap(zero, row_layer_grads)

        ######################################
        # Iterate through the COLUMNS of data
        for col_batch in BatchIter(N, col_batch_size)
          
            D_v = view(D, :, col_batch)
            noise_view = view(model_d.noise_model, 1:M)
            Y_view = view(model_d.Y, :, col_batch)

            col_layers_view = view(col_layers, 1:M, col_batch)

            row_likelihood = (X, rl) -> likelihood(X, Y_view, 
                                                   rl, col_layers_view, 
                                                   noise_view, D_v)

            batchloss, g = Zygote.withgradient(row_likelihood, model_d.X, row_layers)
            binop!(.+, X_grad, g[1])
            binop!(.+, row_layer_grads, g[2])
        
            loss += batchloss
        end

        # Accumulate X regularizer gradients
        regloss, X_reg_grads = Zygote.withgradient(X_regularizer, model_d.X, model_d.X_reg)
        loss += regloss
        binop!(.+, X_grad, X_reg_grads[1])

        # Update X and the X regularizer
        update!(opt, model_d.X, X_grad)
        update!(opt, model_d.X_reg, X_reg_grads[2])

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

    #############################
    # Termination
    #############################

    # Make sure the model gets updated with the trained values
    for pname in propertynames(model_d)
        setproperty!(model, pname, cpu(getproperty(model_d, pname)))
    end

    return model
end


