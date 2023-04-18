

import StatsBase: fit!

AbstractOptimiser = Flux.Optimise.AbstractOptimiser


"""
    fit!(model::MatFacModel, D::AbstractMatrix;
         scale_columns=true, capacity=10^8, 
         opt=AdaGrad(), lr=0.01, max_epochs=1000,
         abs_tol=1e-9, rel_tol=1e-6, 
         verbosity=1, print_iter=10,
         keep_history=false,
         reg_relative_weighting=true,
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
* `lr` is learning rate. Default 0.01.
* `abs_tol` and `rel_tol` are standard convergence criteria.
* `max_epochs`: an upper bound on the number of epochs.
* `verbosity`: larger values make the method print more 
  information to stdout.
* `print_iter`: the number of iterations between printouts to stdout
* `keep_history`: Bool indicating whether to record the training loss history
* `reg_relative_weighting`: whether to reweight the regularizers 
                            s.t. they behave consistently across problem sizes
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
              t_start=nothing,
              epoch=1
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

    nthread = Threads.nthreads()
    capacity = min(capacity, M_D*N_D)
    col_batch_size = max(div(capacity,(M*nthread)),1)
    row_batch_size = max(div(capacity,(N*nthread)),1)
    
    # Reweight the column losses if necessary
    if scale_column_losses
        vprint("Re-weighting column losses...\n")
        rescale_column_losses!(model, D; verbosity=verbosity,
                                         prefix=string(print_prefix, "    "))
    end

    # Prep the row and column transformations 
    row_layers = make_viewable(model.row_transform)
    col_layers = make_viewable(model.col_transform)

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
    X_grads = [zero(model.X) for _=1:nthread]
    Y_grads = [zero(model.Y) for _=1:nthread]
    row_layer_grads = [deepcopy(rec_trainable(model.row_transform)) for _=1:nthread]
    col_layer_grads = [deepcopy(rec_trainable(model.col_transform)) for _=1:nthread]
    noise_model_grads = [deepcopy(rec_trainable(model.noise_model)) for _=1:nthread]

    # Define the row and column batchese
    row_batches = batch_iterations(M, row_batch_size)
    col_batches = batch_iterations(N, col_batch_size)

    #############################
    # Main loop 
    #############################
    if t_start == nothing
        t_start = time()
    end
    tol_iters = 0
    d_loss = zeros(nthread)
    term_code = "max_epochs"

    while epoch <= max_epochs

        if (epoch % print_iter) == 0
            vprint("Epoch ", epoch,":  ")
        end

        # Set losses to zero
        loss = 0.0
        d_loss .= 0
        d_loss_total = 0.0

        # Set gradients to zero
        for n=1:nthread 
            zero_out!(X_grads[n]) 
            zero_out!(Y_grads[n])
            zero_out!(row_layer_grads[n])
            zero_out!(col_layer_grads[n])
            zero_out!(noise_model_grads[n])
        end

        ######################################
        # Iterate through the ROWS of data
        if (update_Y | update_col_layers)
            
            Threads.@threads :static for row_batch in row_batches 
                th = Threads.threadid()

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
                    d_loss[th] += batch_loss
                end

                # Accumulate gradients
                if update_Y
                    binop!(.+, Y_grads[th], grads[1])
                end
                if update_col_layers
                    binop!(.+, col_layer_grads[th], grads[2])
                end
                if update_noise_models
                    binop!(.+, noise_model_grads[th], grads[3])
                end
            end
        end

        Y_reg_loss = 0.0
        if (update_Y | update_Y_reg)
            Y_reg_loss, Y_reg_grad = Zygote.withgradient(Y_regularizer, model.Y, model.Y_reg)
            if update_Y
                # Add Y's regularizer gradient
                binop!(.+, Y_grads[1], Y_reg_grad[1])
 
                # Update Y with the accumulated gradient
                accumulate_sum!(Y_grads)
                update!(opt, model.Y, Y_grads[1])
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
                binop!(.+, col_layer_grads[1], reg_grads[1])
                accumulate_sum!(col_layer_grads)
                update!(opt, model.col_transform, col_layer_grads[1])
            end
            if update_col_layers_reg
                # Update column layer regularizer's parameters
                update!(opt, col_layer_regs, reg_grads[2])
            end
        end
        if update_noise_models 
            # Update the noise model
            accumulate_sum!(noise_model_grads)
            update!(opt, model.noise_model, noise_model_grads[1])
        end

        ######################################
        # Iterate through the COLUMNS of data
        if (update_X | update_row_layers)
            Threads.@threads :static for col_batch in col_batches # BatchIter(N, col_batch_size)
               
                th = Threads.threadid()

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
                d_loss[th] += batchloss

                # Accumulate gradients
                if update_X
                    binop!(.+, X_grads[th], g[1])
                end
                if update_row_layers
                    binop!(.+, row_layer_grads[th], g[2]) 
                end
            end
        end

        # Update X via regularizer gradients
        X_reg_loss = 0.0
        if (update_X | update_X_reg)
            X_reg_loss, X_reg_grads = Zygote.withgradient(X_regularizer, model.X, model.X_reg)
            if update_X
                binop!(.+, X_grads[1], X_reg_grads[1])
       
                accumulate_sum!(X_grads) 
                update!(opt, model.X, X_grads[1])
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
                binop!(.+, row_layer_grads[1], reg_grads[1])

                accumulate_sum!(row_layer_grads)
                update!(opt, row_layers, row_layer_grads[1])
            end
            if update_row_layers_reg
                # Update the row layer regularizer's parameters
                update!(opt, row_layer_regs, reg_grads[2])
            end
        end

        # Sum the loss components (halve the data loss
        # to account for row-pass and col-pass)
        d_loss_total = sum(d_loss)
        loss = (d_loss_total + row_layer_reg_loss 
                + col_layer_reg_loss + X_reg_loss + Y_reg_loss)
        elapsed = time() - t_start

        history!(hist; data_loss=d_loss_total, 
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
                term_code = "abs_tol"
                vprint("termination counter: ", tol_iters,"/",tol_max_iters ,"; abs_tol=",abs_tol, "\n"; level=1)
            elseif loss_diff/abs(loss) < abs_tol
                tol_iters += 1
                term_code = "rel_tol"
                vprint("termination counter: ", tol_iters,"/",tol_max_iters ,"; rel_tol=",rel_tol, "\n"; level=1)
            else
                tol_iters = 0
            end
        else # Loss is higher than the best loss!
            if loss_diff < 0 # Loss has increased! 
                tol_iters += 1
                term_code = "loss_increase"
                vprint("termination counter: ", tol_iters,"/",tol_max_iters ,"; loss increase (Loss=", loss,")\n"; level=1)
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
    finalize_history!(hist; term_code=term_code, epochs=epoch)

    return hist
end


############################################################
# Compute M-estimates of columns
############################################################


function compute_M_estimates(model::MatFacModel, D::AbstractMatrix; capacity=10^8, keep_history=false, kwargs...)

    K, M = size(model.X)
    N = size(model.Y, 2)

    model_copy = deepcopy(model)
    model_copy.X = similar(model.X, (1,M))
    model_copy.X .= 1
    model_copy.Y = similar(model.Y, (1,N))

    # Initialize the M-estimates at the *means*
    # of the columns, after applying the link functions
    model_copy.Y[1,:] .= batched_link_mean(model.noise_model, D; capacity=capacity)
    model_copy.Y_reg = (x -> 0.0)
    
    model_copy.col_transform = (x -> x)
    model_copy.row_transform = (x -> x)

    h = fit!(model_copy, D; scale_column_losses=false, 
                            update_X=false, 
                            update_Y=true, 
                            update_row_layers=false,
                            update_col_layers=false,
                            update_X_reg=false,
                            update_Y_reg=false,
                            update_row_layers_reg=false,
                            update_col_layers_reg=false,
                            keep_history=keep_history,
                            kwargs...)

    nan_idx = (!isfinite).(model_copy.Y)
    model_copy.Y[nan_idx] .= 0
    if keep_history 
        return model_copy.Y, h
    else
        return model_copy.Y
    end
end


##############################################################
# Means and variances of loss
##############################################################

function batched_column_loss_sum(model::MatFacModel, D::AbstractMatrix; capacity=10^8, map_func=x->x)
    N = size(D,1)
    loss_start = similar(D, (1,N))
    loss_start .= 0

    total_loss = batched_mapreduce((m,d) -> sum(map_func(column_loss_sum(m,d)), dims=1),
                                   (s,L) -> s .+ L,
                                   model, D;
                                   start=loss_start,
                                   capacity=capacity)
    return total_loss
end


function batched_column_ssq_loss(model::MatFacModel, D::AbstractMatrix; capacity=10^8)
    return batched_column_loss_sum(model, D; capacity=capacity, map_func=x->x.*x)
end


function batched_column_loss_var(model::MatFacModel, D::AbstractMatrix; capacity=10^8)
    total_loss = batched_column_loss_sum(model, D; capacity=capacity)
    total_ssq_loss = batched_column_ssq_loss(model, D; capacity=capacity)
    M_vec = column_nonnan(D)
    mean_loss = vec(total_loss) ./ M_vec
    mean_ssq_loss = vec(total_ssq_loss) ./ M_vec

    mean_loss .*= mean_loss
    return mean_ssq_loss .- mean_loss
end


##############################################################
# Sums of squared gradients
##############################################################
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


function column_ssq_grads(model, D)
    Z = forward(model)
    grads = Zygote.gradient(A->invlinkloss(model.noise_model, A, D), Z)[1]
    return sum(grads .* grads; dims=1)
end


function batched_column_ssq_grads(model, col_M_estimates, D; capacity::Integer=10^8)
    N = size(D, 2)
    reduce_start = similar(D, (1,N))
    reduce_start .= 0
    ssq_grads = batched_mapreduce((mod, D) -> column_ssq_grads(mod, D, col_M_estimates),
                                  (x, y) -> x .+ y,
                                  model, D; start=reduce_start, capacity=capacity)
    return ssq_grads
end

function batched_column_ssq_grads(model, D; capacity::Integer=10^8, map_func=x->x)
    N = size(D, 2)
    reduce_start = similar(D, (1,N))
    reduce_start .= 0
    ssq_grads = batched_mapreduce((mod, D) -> map_func(column_ssq_grads(mod, D)),
                                  (x, y) -> x .+ y,
                                  model, D; start=reduce_start, capacity=capacity)
    return ssq_grads
end

#####################################################
# Loss rescaling
#####################################################

# Rescale the column losses in such a way that each
# column loss yields an X-gradient of similar magnitude
function rescale_column_losses!(model, D; capacity::Integer=10^8, verbosity=1, prefix="")

    K,N = size(model.Y)

    # Compute M-estimates for the columns
    col_M_estimates = compute_M_estimates(model, D; capacity=capacity, 
                                                    verbosity=verbosity, print_prefix=prefix, 
                                                    lr=0.25, max_epochs=1000, rel_tol=1e-9)
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


