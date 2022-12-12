

import StatsBase: fit!

AbstractOptimiser = Flux.Optimise.AbstractOptimiser


"""
    fit!(model::MatFacModel, D::AbstractMatrix;
         scale_columns=true, capacity=10^8, 
         opt=AdaGrad(), lr=0.01, max_epochs=1000,
         abs_tol=1e-9, rel_tol=1e-6, 
         verbosity=1, print_iter=10,
         callback=nothing,
         update_factors=true,
         update_layers=true)

Fit a MatFacModel to dataset D. 

* If `scale_column_losses` is `true`, then we weight each
  column's loss by the inverse of its variance.
* `capacity` refers to memory capacity. It controls
  the size of minibatches during training. I.e., larger
  `capacity` means larger minibatches.
* `opt` is an optional Flux `AbstractOptimiser` object.
  Overrides the `lr` kwarg.
* `lr` is learning rate. Defaults to 0.01.
* `abs_tol` and `rel_tol` are standard convergence criteria.
* `max_epochs`: an upper bound on the number of epochs.
* `verbosity`: larger values make the method print more 
  information to stdout.
* `print_iter`: the number of iterations between printouts to stdout
* `callback`: a function (or callable struct) called at the end of each iteration
* `update_factors`: whether to update the factors (i.e., setting to false holds X,Y fixed)
* `update_layers`: whether to update the layer parameters

We recommend loading `model` and `D` to GPU _before_ fitting:
```
model = gpu(model)
D = gpu(D)
```
"""
function fit!(model::MatFacModel, D::AbstractMatrix;
              capacity::Integer=10^8, 
              max_epochs=1000, lr::Number=0.01,
              opt::Union{Nothing,AbstractOptimiser}=nothing,
              abs_tol::Number=1e-9, rel_tol::Number=1e-6,
              tol_max_iters::Number=3, 
              scale_column_losses=true,
              calibrate_losses=true,
              verbosity::Integer=1,
              print_iter::Integer=10,
              callback=nothing,
              update_factors=true,
              update_layers=true)
    
    #############################
    # Preparations
    #############################
    
    # Define a verbose print
    function vprint(a...; level=1)
        if verbosity >= level
            print(string(a...))
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

    inv_MN = 1.0/(M*N)

    col_batch_size = div(capacity,M)
    row_batch_size = div(capacity,N)
    
    # Reweight the column losses if necessary
    if scale_column_losses
        vprint("Re-weighting column losses\n")
        col_errors = batched_column_mean_loss(model.noise_model, D; 
                                              capacity=capacity)
        weights = abs.(1 ./ col_errors)
        weights = map(x -> max(x, 1e-5), weights)
        weights[ (!isfinite).(weights) ] .= 1
        set_weight!(model.noise_model, weights)
    end

    # Prep the row and column transformations 
    row_layers = make_viewable(model.row_transform)
    col_layers = make_viewable(model.col_transform)

    # Define the log-likelihood function
    likelihood = (X,Y,
                  r_layers,
                  c_layers,
                  noise, D)-> inv_MN*data_loss(X,Y,
                                               r_layers,
                                               c_layers,
                                               noise, D; 
                                               calibrate=calibrate_losses)

    # Prep the regularizers
    col_layer_regs = make_viewable(model.col_transform_reg)
    row_layer_regs = make_viewable(model.row_transform_reg)
   
    col_layer_regularizer = (layers, reg) -> model.lambda_col*reg(layers)
    row_layer_regularizer = (layers, reg) -> model.lambda_row*reg(layers)
    X_regularizer = (X, reg) -> model.lambda_X*reg(X)
    Y_regularizer = (Y, reg) -> model.lambda_Y*reg(Y)

    # If no optimiser is provided, initialize
    # the default (an AdaGrad optimiser)
    if opt == nothing
        opt = Flux.Optimise.AdaGrad(lr)
    end

    # If no callback is provided, construct
    # a do-nothing function
    if callback == nothing
        callback = (args...) -> nothing
    end

    # Track the loss
    prev_loss = Inf
    loss = Inf

    #############################
    # Main loop 
    #############################
    epoch = 1
    t_start = time()
    tol_iters = 0
    while epoch <= max_epochs

        if (epoch % print_iter) == 0
            vprint("Epoch ", epoch,":  ")
        end

        loss = 0.0
        data_loss = 0.0

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
            grads = Zygote.gradient(col_likelihood, 
                                    model.Y,
                                    col_layers_view,
                                    model.noise_model)
            if update_factors
                update!(opt, model.Y, grads[1])
            end
            if update_layers
                update!(opt, col_layers_view, grads[2])
                update!(opt, model.noise_model, grads[3])
            end
        end

        Y_reg_loss = 0.0
        if update_factors
            Y_reg_loss, Y_reg_grad = Zygote.withgradient(Y_regularizer, model.Y, model.Y_reg)
            # Update Y via regularizer gradient
            update!(opt, model.Y, Y_reg_grad[1])
            # Update the Y regularizer
            update!(opt, model.Y_reg, Y_reg_grad[2])
        end

        if update_layers
        end

        col_layer_reg_loss = 0.0
        if update_layers
            col_layer_reg_loss, reg_grads = Zygote.withgradient(col_layer_regularizer, col_layers, col_layer_regs)
            # Update column layers via regularizer gradients
            update!(opt, col_layers, reg_grads[1])
            # Update column layer regularizer's parameters
            update!(opt, col_layer_regs, reg_grads[2])
        end

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
            if update_factors
                update!(opt, model.X, g[1])
            end
            if update_layers
                update!(opt, row_layers, g[2]) 
            end
            data_loss += batchloss
        end

        # Update X via regularizer gradients
        X_reg_loss = 0.0
        if update_factors
            X_reg_loss, X_reg_grads = Zygote.withgradient(X_regularizer, model.X, model.X_reg)
            update!(opt, model.X, X_reg_grads[1])
            # Update the X regularizer
            update!(opt, model.X_reg, X_reg_grads[2])
        end

        row_layer_reg_loss = 0.0
        if update_layers
            row_layer_reg_loss, reg_grads = Zygote.withgradient(row_layer_regularizer, row_layers, row_layer_regs)
            # Update the row layers via regularizer gradients
            update!(opt, row_layers, reg_grads[1])
            # Update the row layer regularizer's parameters
            update!(opt, row_layer_regs, reg_grads[2])
        end

        # Sum the loss components (halve the data loss
        # to account for row-pass and col-pass)
        loss = (data_loss + row_layer_reg_loss 
                + col_layer_reg_loss + X_reg_loss + Y_reg_loss)

        # Execute the callback (may or may not mutate the model)
        callback(model, epoch, data_loss, X_reg_loss, Y_reg_loss,
                               row_layer_reg_loss, col_layer_reg_loss)

        # Report the loss
        if (epoch % print_iter) == 0
            elapsed = time() - t_start
            vprint("Loss=",loss, " (", round(Int, elapsed), "s elapsed)\n")
        end

        # Check termination conditions
        loss_diff = prev_loss - loss 
        if loss_diff < abs_tol
            tol_iters += 1
            epoch += 1
            vprint("termination counter: ", tol_iters,"/",tol_max_iters ,"; abs_tol<",abs_tol, "\n"; level=0)
        elseif loss_diff/abs(loss) < abs_tol
            tol_iters += 1
            epoch += 1
            vprint("termination counter: ", tol_iters,"/",tol_max_iters ,"; rel_tol<",rel_tol, "\n"; level=0)
        else
            tol_iters = 0
            epoch += 1
        end
        prev_loss = loss

        if tol_iters >= tol_max_iters
            vprint("Reached max termination counter (", tol_max_iters, "). Terminating\n"; level=0)
            break
        end

    end
    if epoch >= max_epochs 
        vprint("Terminated: reached max_epochs=", max_epochs, "\n"; level=0)
    end

    #############################
    # Termination
    #############################

    return model
end


