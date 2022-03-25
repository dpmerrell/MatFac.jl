

#########################################
# COMPUTE SIMULATION NOISE PARAMETERS
#########################################


function decompose_xor_signal(output_mean; 
                              snr=10.0)

    output_var = output_mean*(1 - output_mean)
    b = 1 + 1/snr

    signal_var = (snr*0.125) * (b - sqrt(b^2 - 16*output_var/snr))
    noise_var = signal_var / snr

    signal_mean = 0.5*(1 - sqrt(1 - 4*signal_var))
    noise_mean = 0.5*(1 - sqrt(1 - 4*noise_var))
    return (signal_mean,), (noise_mean,)
end

function forward_xor_signal(input_mean, noise_moments)
    p = input_mean
    q = noise_moments[1]
    output_mean = p*(1-q) + q*(1-p)
    return output_mean
end


# Only valid for small output mean;
# relies on an exponential approximation
function decompose_logistic_signal(output_mean;
                                   input_mtv=10.0)

    input_mean = log(output_mean) * input_mtv / (input_mtv + 0.5)
    input_var = abs(input_mean) / input_mtv

    return (input_mean, input_var)
end

function forward_logistic_signal(input_mean, input_var)
    output_mean = exp(input_mean + 0.5*input_var)
    output_var = (exp(input_var) - 1.0)*exp(2.0*input_mean + input_var)
    return output_mean, output_var
end


# log-normality assumptions
function decompose_exp_signal(output_mean, output_var)

    M2 = output_mean*output_mean

    input_mean = log(M2/sqrt(M2 + output_var))
    input_var = log(1 + output_var/M2)

    return (input_mean, input_var)
end

function forward_exp_signal(input_mean, input_var)
    output_mean = exp(input_mean + 0.5*input_var)
    output_var = (exp(input_var) - 1.0)*exp(2.0*input_mean + input_var)
    return output_mean, output_var
end


function decompose_additive_signal(output_mean, output_var;
                                   snr=10.0,
                                   noise_mean=0.0)

    signal_var = output_var * snr / (1 + snr)
    noise_var = output_var - signal_var
    signal_mu = output_mean - noise_mean
    return (signal_mu, signal_var), (noise_mean, noise_var)
end


function decompose_mult_signal(output_mean, output_var;
                               noise_mean=1.0, 
                               snr=10.0)

    signal_mean = output_mean / noise_mean
    b = noise_mean*noise_mean + signal_mean*signal_mean/snr
    noise_var = (-b + sqrt(b*b + 4.0*output_var/snr))*0.5
    signal_var = noise_var*snr

    return (signal_mean, signal_var), (noise_mean, noise_var)
end


function forward_mult_signal(a_mean, a_var, b_mean, b_var)

    output_mean = a_mean*b_mean
    output_var = a_var*b_var + a_var*b_mean*b_mean + b_var*a_mean*a_mean
    return output_mean, output_var
end


function decompose_matfac_signal(out_mean, out_var;
                                 sigma_snr=20.0,
                                 mu_snr=10.0,
                                 delta_snr=10.0,
                                 theta_snr=10.0)

    notheta_signal,
    theta_noise = decompose_additive_signal(out_mean, out_var;
                                            snr=theta_snr,
                                            noise_mean=0.0) # noise is zero-mean
    println(string("NOTHETA_SIGNAL", notheta_signal))
    nodelta_signal,
    delta_noise = decompose_mult_signal(notheta_signal[1], 
                                        notheta_signal[2];
                                        noise_mean=1.0, # want E[delta]==1
                                        snr=delta_snr) # "noise" is exp-distributed
    println(string("NODELTA_SIGNAL", nodelta_signal))

    logdelta_noise = decompose_exp_signal(delta_noise[1],
                                          delta_noise[2])

    # set the mean-to-var ratio s.t. input signal has mean zero
    nomu_signal,
    mu_noise = decompose_additive_signal(nodelta_signal[1],
                                         nodelta_signal[2]; 
                                         snr=mu_snr,
                                         noise_mean=nodelta_signal[1])
    println(string("NOMU_SIGNAL", nomu_signal))
                                         # (Remaining signal should have zero-mean)
    
    sigma_snr = sigma_snr/nomu_signal[2]
    sigma_mean = sqrt(nomu_signal[2] - 1/sigma_snr)
    nosigma_signal,
    sigma_noise = decompose_mult_signal(nomu_signal[1], 
                                        nomu_signal[2];
                                        noise_mean=sigma_mean,
                                        snr=sigma_snr) # 
    println(string("NOSIGMA_SIGNAL", nosigma_signal))
    println("")
    logsigma_noise = decompose_exp_signal(sigma_noise[1],
                                          sigma_noise[2])

    return logsigma_noise, mu_noise,  
           logdelta_noise, theta_noise 

end


function forward_matfac_signal(z_mean, z_var,
                               logsigma_moments, mu_moments,
                               logdelta_moments, theta_moments)
    sigma_mean, sigma_var = forward_exp_signal(logsigma_moments...)
    z_mean, z_var = forward_mult_signal(z_mean, z_var, sigma_mean, sigma_var)
    z_mean += mu_moments[1]
    z_var += mu_moments[2]
    delta_mean, delta_var = forward_exp_signal(logdelta_moments...)
    z_mean, z_var = forward_mult_signal(z_mean, z_var, delta_mean, delta_var) 
    z_mean += theta_moments[1]
    z_var += theta_moments[2]
    return z_mean, z_var
end


function decompose_normal_data_signal(data_mean, data_var;
                               sigma_snr=10.0,
                               mu_snr=10.0,
                               delta_snr=10.0,
                               theta_snr=10.0,
                               sample_snr=10.0,
                               )

    nosample_signal,
    sample_noise = decompose_additive_signal(data_mean, data_var;
                                             snr=sample_snr,
                                             noise_mean=0.0) # Noise has zero-mean

    logsigma_noise,
    mu_noise, 
    logdelta_noise, 
    theta_noise = decompose_matfac_signal(nosample_signal[1], 
                                         nosample_signal[2];
                                         sigma_snr=sigma_snr,
                                         theta_snr=theta_snr,
                                         delta_snr=delta_snr,
                                         mu_snr=mu_snr,
                                         )

    return logsigma_noise, mu_noise,  
           logdelta_noise, theta_noise, 
           sample_noise
end


function decompose_bernoulli_data_signal(data_mean;
                                  sigma_snr=10.0,
                                  mu_snr=10.0,
                                  delta_snr=10.0,
                                  theta_snr=10.0,
                                  logistic_mtv=10.0,
                                  sample_snr=10.0,
                                  )
        
    nosample_signal,
    sample_noise = decompose_xor_signal(data_mean;
                                        snr=sample_snr)
    z_moments = decompose_logistic_signal(nosample_signal[1];
                                          input_mtv=logistic_mtv)
    logsigma_noise,
    mu_noise, 
    logdelta_noise, 
    theta_noise = decompose_matfac_signal(z_moments[1], 
                                          z_moments[2];
                                          sigma_snr=sigma_snr,
                                          theta_snr=theta_snr,
                                          delta_snr=delta_snr,
                                          mu_snr=mu_snr)

    return logsigma_noise, mu_noise,  
           logdelta_noise, theta_noise, 
           sample_noise
end


function decompose_all_data_signal(data_moments,
                                   feature_batch_ids,
                                   feature_losses;
                                   mu_snr=10.0,
                                   delta_snr=10.0,
                                   theta_snr=10.0,
                                   logistic_mtv=10.0,
                                   sample_snr=10.0)

    unq_col_batches = unique(feature_batch_ids)
    col_ranges = ids_to_ranges(feature_batch_ids)    
    batch_losses = [unique(feature_losses[c_rng])[1] for c_rng in col_ranges]
  
    sig_decomp_map = Dict("normal"=>m -> decompose_normal_data_signal(m[1],m[2],
                                                                      mu_snr=mu_snr,
                                                                      delta_snr=delta_snr,
                                                                      theta_snr=theta_snr,
                                                                      sample_snr=sample_snr),
                          "logistic"=>m -> decompose_bernoulli_data_signal(m[1],
                                                                           mu_snr=mu_snr,
                                                                           delta_snr=delta_snr,
                                                                           theta_snr=theta_snr,
                                                                           logistic_mtv=logistic_mtv,
                                                                           sample_snr=sample_snr),
                          "noloss"=>m -> decompose_normal_data_signal(0,1)
                         )

    # Compute variance contributions for each column batch
    batch_characteristics = [sig_decomp_map[b_l](m) for (m, b_l) in zip(data_moments, batch_losses)]

    return batch_characteristics
end

#######################################
# SIMULATE DATA
#######################################

function normal_prec(precision::AbstractMatrix; n_samples=1)

    N = size(precision,1)
    if n_samples > 1
        z = randn(N, n_samples)
    else
        z = randn(N)
    end
    fac = cholesky(precision)
    x = fac.UP\z
    
    return x
end


function sim_X(reg_mats)
    K = length(reg_mats)
    return permutedims(normal_prec(reg_mats[1]; n_samples=K)) 
end


function sim_Y(reg_mats)
    vecs = [normal_prec(mat) for mat in reg_mats]
    return permutedims(hcat(vecs...))
end


function sim_col_param(range_vec, moments_vec)
    vecs = Vector{Float64}[randn(length(r)).*sqrt(mv[2]) .+ mv[1] for (r,mv) in zip(range_vec, moments_vec)]
    return vcat(vecs...)
end

function sim_batch_param(mbatch_vec, moments_vec)
    return Vector{Float64}[randn(mb).*sqrt(mv[2]) .+ mv[1] for (mb, mv) in zip(mbatch_vec, moments_vec)]
end


function simulate_params(X_reg, Y_reg,
                         row_batch_ids,
                         col_batch_ids,
                         logsigma_moments_vec,
                         mu_moments_vec,
                         logdelta_moments_vec,
                         theta_moments_vec)
  
    K = length(X_reg)
    X = sim_X(X_reg)
    Y = sim_Y(Y_reg) ./ sqrt(K)

    col_ranges = ids_to_ranges(col_batch_ids)
    log_sigma = sim_col_param(col_ranges, logsigma_moments_vec)
    mu = sim_col_param(col_ranges, mu_moments_vec)

    mbatch_vec = Int64[length(unique(rbv)) for rbv in row_batch_ids]
    log_delta_values = sim_batch_param(mbatch_vec, logdelta_moments_vec)
    log_delta_values = Dict{String,Float64}[Dict{String,Float}(zip(unique(rbv),ldv)) 
                                            for (ldv, rbv) in zip(log_delta_values, row_batch_ids)]

    theta_values = sim_batch_param(mbatch_vec, theta_moments_vec)
    theta_values = Dict{String,Float64}[Dict{String,Float}(zip(unique(rbv),ldv)) 
                                        for (ldv, rbv) in zip(theta_values, row_batch_ids)]

    # Move everything to GPU
    X = CUDA.CuArray(X)
    Y = CUDA.CuArray(Y)
    log_sigma = CUDA.CuArray(log_sigma)
    mu = CUDA.CuArray(mu)

    # Assemble BatchMatrix objects
    log_delta = batch_matrix(log_delta_values, row_batch_ids, col_batch_ids)
    theta = batch_matrix(theta_values, row_batch_ids, col_batch_ids)

    # Return a ModelParams object
    return ModelParams(X, Y, mu, log_sigma, theta, log_delta)

end



function simulate_data(params, noise_models)

    unq_nm = unique(noise_models)
    nm_ranges = ids_to_ranges(noise_models)

    link_map = ColBlockMap(Function[LINK_FUNCTION_MAP[nm] for nm in unq_nm],
                           nm_ranges)

    A = batch_forward(params.X, params.Y, params.mu, params.log_sigma,
                      params.theta, params.log_delta, link_map)

    sampler_map = ColBlockMap(Function[SAMPLE_FUNCTION_MAP[nm] for nm in unq_nm],
                              nm_ranges)

    D = sampler_map(A)

    return D
    
end


function full_simulation(X_reg, Y_reg,
                         col_batch_moments,
                         feature_batch_ids,
                         sample_batch_ids,
                         feature_noise_models;
                         mu_snr=10.0,
                         delta_snr=10.0,
                         theta_snr=10.0,
                         logistic_mtv=10.0,
                         sample_snr=10.0)


    # Compute the contributions of different model
    # parameters to the data's overall means and variances
    batch_var_contribs = decompose_all_data_signal(col_batch_moments,
                                                   feature_batch_ids,
                                                   sample_batch_ids,
                                                   feature_noise_models;
                                                   mu_snr=mu_snr,
                                                   delta_snr=delta_snr,
                                                   theta_snr=theta_snr,
                                                   logistic_mtv=logistic_mtv,
                                                   sample_snr=sample_snr)

    # Simulate model parameters
    sim_model_params = simulate_params(X_reg, Y_reg, 
                                       feature_batch_ids,
                                       sample_batch_ids,
                                       batch_var_contribs)

    # Simulate the data from the parameters
    D = simulate_data(sim_model_params, feature_noise_models)

    # Return the parameters and data
    return sim_model_params, D
end




