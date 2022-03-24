

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


function decompose_logistic_signal(output_mean;
                                   input_mtv=10.0)

    input_mean = log(output_mean) * input_mtv / (input_mtv + 0.5)
    input_var = input_mean / input_mtv

    return (input_mean, input_var)
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
                                   noise_mtv=0.0)

    signal_var = output_var * snr / (1 + snr)
    noise_var = output_var - signal_var
    noise_mu = sqrt(noise_mtv * noise_var)
    signal_mu = output_mean - noise_mu
    return (signal_mu, signal_var), (noise_mu, noise_var)
end


function decompose_mult_signal(output_mean, output_var;
                               mean_ratio=10.0, 
                               snr=10.0)

    #signal_var = sqrt((snr/(1 + noise_mtv))*(output_var - output_mean*output_mean/noise_mtv))
    #noise_var = signal_var / snr
    #noise_mean = sqrt(noise_mtv * noise_var)
    #signal_mean = output_mean/noise_mean

    signal_mean = sqrt(mean_ratio * output_mean + 1e-9)
    noise_mean = signal_mean / mean_ratio

    b = noise_mean*noise_mean + signal_mean*signal_mean/snr
    
    noise_var = (sqrt(b*b + 4.0*output_var/snr) - b)*0.5
    signal_var = noise_var*snr 

    return (signal_mean, signal_var), (noise_mean, noise_var)
end


function forward_mult_signal(a_mean, a_var, b_mean, b_var)

    output_mean = a_mean*b_mean
    output_var = a_var*b_var + a_var*b_mean*b_mean + b_var*a_mean*a_mean
    return output_mean, output_var
end


function decompose_matfac_signal(out_mean, out_var;
                                 mu_snr=10.0,
                                 delta_snr=10.0,
                                 theta_snr=10.0
                                 )
    notheta_signal,
    theta_noise = decompose_additive_signal(out_mean, out_var;
                                            snr=theta_snr,
                                            noise_mtv=0.0) # noise is zero-mean

    nodelta_signal,
    delta_noise = decompose_mult_signal(notheta_signal[1], 
                                        notheta_signal[2];
                                        mean_ratio=notheta_signal[1], # want E[delta]==1
                                        snr=delta_snr) # "noise" is exp-distributed

    logdelta_noise = decompose_exp_signal(delta_noise[1],
                                          delta_noise[2])

    # set the mean-to-var ratio s.t. input signal has mean zero
    mu_noise_mtv = (mu_snr+1)*nodelta_signal[1]^2 / nodelta_signal[2]
    nomu_signal,
    mu_noise = decompose_additive_signal(nodelta_signal[1],
                                         nodelta_signal[2]; 
                                         snr=mu_snr,
                                         noise_mtv=mu_noise_mtv) # noise is zero-mean

    nosigma_signal,
    sigma_noise = decompose_mult_signal(nomu_signal[1], 
                                        nomu_signal[2];
                                        mean_ratio=1, # want Var[XY]=1 
                                        snr=1/nomu_signal[2]) # 
   
    logsigma_noise = decompose_exp_signal(sigma_noise[1],
                                          sigma_noise[2])

    return logsigma_noise, mu_noise,  
           logdelta_noise, theta_noise 

end


function decompose_normal_data_signal(data_mean, data_var;
                               mu_snr=10.0,
                               delta_snr=10.0,
                               theta_snr=10.0,
                               sample_snr=10.0,
                               )

    nosample_signal,
    sample_noise = decompose_additive_signal(data_var, sample_snr)

    logsigma_noise,
    mu_noise, 
    logdelta_noise, 
    theta_noise = decompose_matfac_signal(nosample_signal[1], 
                                         nosample_signal[2];
                                         theta_snr=theta_snr,
                                         delta_snr=delta_snr,
                                         mu_snr=mu_snr,
                                         )

    return logsigma_noise, mu_noise,  
           logdelta_noise, theta_noise, 
           sample_noise
end


function decompose_bernoulli_data_signal(data_mean;
                                  mu_snr=10.0,
                                  delta_snr=10.0,
                                  theta_snr=10.0,
                                  logistic_mtv=10.0,
                                  sample_snr=10.0,
                                  )
        
    nosample_signal,
    sample_noise = decompose_xor_signal(data_mean;
                                        snr=sample_snr)
    
    input_signal = decompose_logistic_signal(nosample_signal[1];
                                             input_mtv=logistic_mtv)
   
    logsigma_noise
    mu_noise, 
    logdelta_noise, 
    theta_noise = decompose_matfac_signal(input_var, input_mean;
                                          theta_snr=theta_snr,
                                          delta_snr=delta_snr,
                                          mu_snr=mu_snr,
                                          )

    return logsigma_noise, mu_noise,  
           logdelta_noise, theta_noise, 
           sample_noise
end


function decompose_all_data_signal(data_moments,
                                   feature_batch_ids,
                                   sample_batch_ids,
                                   feature_losses;
                                   mu_snr=10.0,
                                   delta_snr=10.0,
                                   theta_snr=10.0,
                                   logistic_mtv=10.0,
                                   sample_snr=10.0)

    unq_col_batches = unique(col_batch_ids)
    col_ranges = ids_to_ranges(col_batch_ids)    
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
    batch_characteristics = [sig_decomp_map[b_l](m) for (m, b_l) in zip(col_batch_moments, batch_losses)]

    return batch_characteristics
end

#######################################
# SIMULATE DATA
#######################################

function normal_prec(precision::AbstractMatrix; n_samples=1)

    N , _ = size(precision)
    if n_samples > 1
        z = CUDA.randn(N, n_samples)
    else
        z = CUDA.randn(N)
    end
    fac = cholesky(precision)
    x = fac.UP \ z
    
    return x
end


function sim_X(reg_mats, var)
    K = length(reg_mats)
    return normal_prec(reg_mats[1]; n_samples=K) 
end


function sim_Y(reg_mats, var)
    vecs = [normal_prec(mat) for mat in reg_mats]
    return hcat(vecs...)
end


function sim_mu(reg_mat, loc, var)
    return normal_prec(reg_mat)*sqrt(var) + loc
end


function sim_log_sigma(reg_mat, loc, var)
    return normal_prec(reg_mat)*sqrt(var) + loc
end


function sim_theta(N_vec, loc, var)
    return BMFVec[CUDA.randn(N).*sqrt(var) .+ loc for N in N_vec]
end


function sim_log_delta(N_vec, loc, var)
    return BMFVec[CUDA.randn(N).*sqrt(var) .+ loc for N in N_vec]
end


function simulate_params(X_reg, Y_reg,
                         mu_reg, mu_noise,
                         log_sigma_reg, log_sigma_noise,
                         theta, theta_noise,
                         log_delta, log_delta_noise)
  
    K = length(X_reg)

    X = sim_X(X_reg)
    Y = sim_Y(Y_reg) ./ sqrt(K)

    mu = sim_mu(mu_reg, mu_noise[1], mu_noise[2])
    log_sigma = sim_log_sigma(log_sigma_reg, 
                              log_sigma_noise[1],
                              log_sigma_noise[2])
    
    N_vec = Int[length(v) for v in theta.values]

    theta_copy = zero(theta)
    theta_copy.values = sim_theta(N_vec)
    log_delta_copy = zero(log_delta) 
    log_delta_copy.values = sim_log_delta(N_vec)

    return ModelParams(X, Y, mu, log_sigma, theta_copy, log_delta_copy)

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




