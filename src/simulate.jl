

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


function X_sim_prior(reg_mats)
    K = length(reg_mats)
    return normal_prec(reg_mats[1]; n_samples=K) 
end


function Y_sim_prior(reg_mats)
    vecs = [normal_prec(mat) for mat in reg_mats]
    return hcat(vecs...)
end


function mu_sim_prior(reg_mat)
    return normal_prec(reg_mat)
end


function log_sigma_sim_prior(reg_mat)
    return normal_prec(reg_mat)
end


function theta_sim_prior(N_vec; std_theta=1.0)
    return BMFVec[CUDA.randn(N) .+ (randn()*std_theta) for N in N_vec]
end


function log_delta_sim_prior(N_vec; std_delta=1.0)
    return BMFVec[CUDA.randn(N) .+ (randn()*std_delta) for N in N_vec]
end


function simulate_params(X_reg, Y_reg, mu_reg, log_sigma_reg,
                         theta, log_delta)
   
    X = X_sim_prior(X_reg)
    Y = Y_sim_prior(Y_reg)

    mu = mu_sim_prior(mu_reg)
    log_sigma = log_sigma_sim_prior(log_sigma_reg)
    
    N_vec = Int[length(v) for v in theta.values]

    theta_copy = zero(theta)
    theta_copy.values = theta_sim_prior(N_vec)
    log_delta_copy = zero(log_delta) 
    log_delta_copy.values = log_delta_sim_prior(N_vec)

    return ModelParams(X, Y, mu, log_sigma, theta_copy, log_delta_copy)

end


function simulate_data(X_reg, Y_reg, mu_reg, log_sigma_reg,
                       theta, log_delta, noise_models)

    params = simulate_params(X_reg, Y_reg, mu_reg, log_sigma_reg,
                             theta, log_delta)

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



