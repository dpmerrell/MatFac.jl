

function decompose_additive_variance(total_var, stn_ratio)

    signal_var = total_var * stn_ratio / (1 + stn_ratio)
    noise_var = total_var - signal_var
    return signal_var, noise_var
end


function decompose_xor_variance(total_var, stn_ratio)

    b = 1 + 1/stn_ratio

    signal_var = (stn_ratio*0.125) * (b - sqrt(b^2 - 16*total_var/stn_ratio))
    noise_var = signal_var / stn_ratio

    return signal_var, noise_var
end


function decompose_logistic_signal(output_mean, mtv_ratio)

    #output_p = 0.5*(1 - sqrt(1 - 4*output_var))
    input_mean = log(output_mean) * mtv_ratio / (mtv_ratio + 0.5)
    input_var = input_mean / mtv_ratio

    return input_mean, input_var
end


function decompose_mult_variance(output_mean, output_var;
                                 stn_ratio=1.0,
                                 noise_mtv=1.0)

    a = 1 - noise_mtv/stn_ratio
    b = output_mean
    c = - stn_ratio/(1 + noise_mtv)

    signal_var = (-b + sqrt(b^2 - 4*a*c))/(2*a)
    noise_var = signal_var/stn_ratio
    
    noise_mean = noise_mtv*noise_var
    signal_mean = output_mean - noise_mean

    return signal_mean, signal_var, noise_mean, noise_var
end


function decompose_model_variance(out_var, out_mean;
                                  theta_stn=10.0,
                                  delta_stn=10.0,
                                  mu_stn=10.0,
                                  sigma_stn=10.0
                                 )
    notheta_signal_var,
    notheta_noise_var = decompose_additive_variance(out_var, 
                                                    theta_stn)

    nodelta_signal_mean,
    nodelta_signal_var,
    nodelta_noise_mean,
    nodelta_noise_var = decompose_mult_variance(out_mean, notheta_signal_var;
                                                stn_ratio=delta_stn,
                                                noise_mtv=1.0)

    nomu_signal_var,
    nomu_noise_var = decompose_additive_variance(nodelta_signal_var, mu_stn)
   
    nosigma_signal_var = 1.0
    nosigma_stn = nomu_signal_var / nosigma_signal_var
  


    nosigma_signal_mean,
    _,
    nosigma_noise_mean,
    nosigma_noise_var = decompose_mult_variance(nodelta_signal_mean, 
                                                nomu_signal_var;
                                                stn_ratio=sigma_stn,
                                                noise_mtv=1.0)

    return nosigma_noise_var, nomu_noise_var,  
           nodelta_noise_var, notheta_noise_var 

end


function decompose_normal_data_variance(data_var, data_mean;
                                        sample_stn=10.0,
                                        theta_stn=10.0,
                                        delta_stn=10.0,
                                        mu_stn=10.0,
                                        sigma_stn=10.0
                                        )

    sample_signal_var,
    sample_noise_var = decompose_additive_variance(data_var, sample_stn)

    nosigma_noise_var,
    nomu_noise_var, 
    nodelta_noise_var, 
    notheta_noise_var = decompose_model_variance(sample_signal_var, data_mean;
                                                 theta_stn=theta_stn,
                                                 delta_stn=delta_stn,
                                                 mu_stn=mu_stn,
                                                 sigma_stn=sigma_stn
                                               )

    return nosigma_noise_var, nomu_noise_var,  
           nodelta_noise_var, notheta_noise_var, 
           sample_noise_var
end


function decompose_bernoulli_data_variance(data_var;
                                           logistic_mtv=10.0,
                                           sample_stn=10.0,
                                           theta_stn=10.0,
                                           delta_stn=10.0,
                                           mu_stn=10.0
                                          )
        
    sample_signal_var,
    sample_noise_var = decompose_xor_variance(data_var, sample_stn)

    sample_signal_mean = 0.5*(1 - sqrt(1 - 4*sample_signal_var))
    
    input_mean, 
    input_var = decompose_logistic_signal(sample_signal_mean, logistic_mtv)
    
    nomu_noise_var, 
    nodelta_noise_var, 
    notheta_noise_var, 
    notheta_noise_var = decompose_model_variance(input_var, input_mean;
                                                 theta_stn=theta_stn,
                                                 delta_stn=delta_stn,
                                                 mu_stn=mu_stn,
                                                 sigma_stn=sigma_stn
                                                 )

    return nosigma_noise_var, nomu_noise_var,  
           nodelta_noise_var, notheta_noise_var, 
           sample_noise_var
end


#function normal_prec(precision::AbstractMatrix; n_samples=1)
#
#    N , _ = size(precision)
#    if n_samples > 1
#        z = CUDA.randn(N, n_samples)
#    else
#        z = CUDA.randn(N)
#    end
#    fac = cholesky(precision)
#    x = fac.UP \ z
#    
#    return x
#end
#
#
#function X_sim_prior(reg_mats)
#    K = length(reg_mats)
#    return normal_prec(reg_mats[1]; n_samples=K) 
#end
#
#
#function Y_sim_prior(reg_mats)
#    vecs = [normal_prec(mat) for mat in reg_mats]
#    return hcat(vecs...)
#end
#
#
#function mu_sim_prior(reg_mat)
#    return normal_prec(reg_mat)
#end
#
#
#function log_sigma_sim_prior(reg_mat)
#    return normal_prec(reg_mat)
#end
#
#
#function theta_sim_prior(N_vec; std_theta=1.0)
#    return BMFVec[CUDA.randn(N) .+ (randn()*std_theta) for N in N_vec]
#end
#
#
#function log_delta_sim_prior(N_vec; std_delta=1.0)
#    return BMFVec[CUDA.randn(N) .+ (randn()*std_delta) for N in N_vec]
#end
#
#
#function simulate_params(X_reg, Y_reg, mu_reg, log_sigma_reg,
#                         theta, log_delta)
#   
#    X = X_sim_prior(X_reg)
#    Y = Y_sim_prior(Y_reg)
#
#    mu = mu_sim_prior(mu_reg)
#    log_sigma = log_sigma_sim_prior(log_sigma_reg)
#    
#    N_vec = Int[length(v) for v in theta.values]
#
#    theta_copy = zero(theta)
#    theta_copy.values = theta_sim_prior(N_vec)
#    log_delta_copy = zero(log_delta) 
#    log_delta_copy.values = log_delta_sim_prior(N_vec)
#
#    return ModelParams(X, Y, mu, log_sigma, theta_copy, log_delta_copy)
#
#end
#
#function simulate_data(X_reg, Y_reg, mu_reg, log_sigma_reg,
#                       theta, log_delta, noise_models)
#
#    params = simulate_params(X_reg, Y_reg, mu_reg, log_sigma_reg,
#                             theta, log_delta)
#
#    unq_nm = unique(noise_models)
#    nm_ranges = ids_to_ranges(noise_models)
#
#    link_map = ColBlockMap(Function[LINK_FUNCTION_MAP[nm] for nm in unq_nm],
#                           nm_ranges)
#
#    A = batch_forward(params.X, params.Y, params.mu, params.log_sigma,
#                      params.theta, params.log_delta, link_map)
#
#    sampler_map = ColBlockMap(Function[SAMPLE_FUNCTION_MAP[nm] for nm in unq_nm],
#                              nm_ranges)
#
#    D = sampler_map(A)
#
#    return params, D
#    
#end



