

import StatsBase: fit!

AbstractOptimiser = Flux.Optimise.AbstractOptimiser


"""
    fit!(model::MatFacModel, D::AbstractMatrix;
         scale_columns=true, capacity=10^8, 
         opt=AdaGrad(), lr=0.01, max_epochs=1000,
         abs_tol=1e-9, rel_tol=1e-6, 
         verbosity=1, print_iter=10,
         callback=nothing,
         update_X=true,
         update_Y=true,
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
* `update_X`: whether to update the factor X (i.e., setting to false holds X fixed)
* `update_Y`: whether to update the factor Y (i.e., setting to false holds Y fixed)
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
              abs_tol::Number=1e-9, rel_tol::Number=1e-9,
              tol_max_iters::Number=3, 
              scale_column_losses=true,
              calibrate_losses=true,
              verbosity::Integer=1,
              print_iter::Integer=10,
              callback=nothing,
              update_X=true,
              update_Y=true,
              update_row_layers=true,
              update_col_layers=true,
              update_X_reg=true,
              update_Y_reg=true,
              update_row_layers_reg=true,
              update_col_layers_reg=true,
              )
    
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

    NdK = N/K
    MdK = M/K

    col_batch_size = div(capacity,M)
    row_batch_size = div(capacity,N)
    
    # Reweight the column losses if necessary
    if scale_column_losses
        vprint("Re-weighting column losses\n")
        rescale_column_losses!(model, D)
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
    X_regularizer = (X, reg) -> NdK*model.lambda_X*reg(X)
    Y_regularizer = (Y, reg) -> MdK*model.lambda_Y*reg(Y)

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
        d_loss = 0.0

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
                batchloss, grads = Zygote.withgradient(col_likelihood, 
                                                 model.Y,
                                                 col_layers_view,
                                                 model.noise_model)
                d_loss += batchloss
                if update_Y
                    update!(opt, model.Y, grads[1])
                end
                if update_col_layers
                    update!(opt, col_layers_view, grads[2])
                    update!(opt, model.noise_model, grads[3])
                end
            end
        end

        Y_reg_loss = 0.0
        if (update_Y | update_Y_reg)
            Y_reg_loss, Y_reg_grad = Zygote.withgradient(Y_regularizer, model.Y, model.Y_reg)
            if update_Y
                # Update Y via regularizer gradient
                update!(opt, model.Y, Y_reg_grad[1])
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
            # Update column layers via regularizer gradients
                update!(opt, col_layers, reg_grads[1])
            end
            if update_col_layers_reg
            # Update column layer regularizer's parameters
                update!(opt, col_layer_regs, reg_grads[2])
            end
        end

        ######################################
        # Iterate through the COLUMNS of data
        if (update_X | update_row_layers)
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
                d_loss += batchloss

                if update_X
                    update!(opt, model.X, g[1])
                end
                if update_row_layers
                    update!(opt, row_layers, g[2]) 
                end
            end
        end
        # Update X via regularizer gradients
        X_reg_loss = 0.0
        if (update_X | update_X_reg)
            X_reg_loss, X_reg_grads = Zygote.withgradient(X_regularizer, model.X, model.X_reg)
            if update_X
                update!(opt, model.X, X_reg_grads[1])
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
                update!(opt, row_layers, reg_grads[1])
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

        # Execute the callback (may or may not mutate the model)
        callback(model, epoch, d_loss, X_reg_loss, Y_reg_loss,
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


############################################################
# Rescaling column losses
############################################################

# Compute M-estimates of columns
function compute_M_estimates(model, D; kwargs...)

    K, M = size(model.X)
    N = size(model.Y, 2)

    model_copy = deepcopy(model)
    model_copy.X = similar(model.X, (1,M))
    model_copy.X .= 1
    model_copy.Y = similar(model.Y, (1,N))
    model_copy.Y .= 0 

    fit!(model_copy, D; scale_column_losses=false, 
                        update_X=false, 
                        update_row_layers=false,
                        update_col_layers=false,
                        update_X_reg=false,
                        update_Y_reg=false,
                        update_row_layers_reg=false,
                        update_col_layers_reg=false,
                        kwargs...)

    return model_copy.Y
end

# Compute columns' sum-squared-gradients of loss
# w.r.t. given M-estimates.
function column_ssq_grads(model, D, m_row)

    m_mat = repeat(m_row, size(D,1), 1)
    grads = Zygote.gradient(Z -> invlinkloss(model.noise_model, 
                                     model.col_transform(
                                         model.row_transform(Z
                                         )
                                     ),
                                 D; calibrate=false), m_mat
                            )[1]
    return sum(grads .* grads; dims=1)
end 


function batched_column_ssq_grads(model, col_M_estimates, D; capacity::Integer=10^8)
    N = size(D, 2)
    reduce_start = similar(D, (1,N))
    reduce_start .= 0
    ssq_grads = batched_reduce((v, mod, D) -> v .+ column_ssq_grads(mod, D, col_M_estimates),
                                   model, D; start=reduce_start, capacity=capacity)
    return ssq_grads
end


# Rescale the column losses in such a way that each
# column loss yields an X-gradient of similar magnitude
function rescale_column_losses!(model, D; capacity::Integer=10^8)

    K,N = size(model.Y)

    # Compute M-estimates for the columns
    col_M_estimates = compute_M_estimates(model, D; capacity=capacity, verbosity=-1, 
                                                    lr=1.0, max_epochs=1000, rel_tol=1e-9)
    # For each column, compute the sum of squared partial derivatives
    # of loss w.r.t. the M-estimates. (This turns out to be an appropriate
    # scaling factor.)
    ssq_grads = vec(batched_column_ssq_grads(model, col_M_estimates, D; capacity=capacity))

    M = column_nonnan(D)
    weights = sqrt.(M ./ ssq_grads)
    weights[ (!isfinite).(weights) ] .= 1
    weights = map(x -> max(x, 1e-2), weights)
    weights = map(x -> min(x, 10.0), weights)
    set_weight!(model.noise_model, vec(weights))

end


