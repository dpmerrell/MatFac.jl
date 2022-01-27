

export BatchMatFacModel


mutable struct BatchMatFacModel

        X::BMFMat
        Y::BMFMat

        X_reg::Vector{BMFRegMat}
        Y_reg::Vector{BMFRegMat}

        mu::BMFVec
        log_sigma::BMFVec

        mu_reg::BMFRegMat
        log_sigma_reg::BMFRegMat

        theta::Matrix{<:Number}
        log_delta::Matrix{<:Number}

        sample_group_ids::Vector{<:Union{Number,String}}
        feature_group_ids::Vector{<:Union{Number,String}}

        feature_noise_models::Vector{String}

end


# Define an alias
BMFModel = BatchMatFacModel


# Define a convenience constructor
function BatchMatFacModel(X_reg, Y_reg, mu_reg, log_sigma_reg,
                          sample_block_ids, feature_block_ids,
                          feature_loss_names)

    M = size(X_reg[1], 1)
    N = size(Y_reg[1], 1)
    K = length(X_reg)

    @assert K == length(Y_reg)

    my_randn = CUDA.randn
    my_ones = CUDA.ones
    my_zeros = CUDA.zeros

    # These may be on the GPU or CPU
    X = my_randn(BMFFloat, K,M) ./ BMFFloat(10.0*sqrt(K))
    Y = my_randn(BMFFloat, K,N) ./ BMFFloat(10.0*sqrt(K))
    mu = my_randn(BMFFloat, N) ./ BMFFloat(100.0)
    log_sigma = my_zeros(BMFFloat, N)

    n_sample_blocks = length(unique(sample_block_ids))
    n_feature_blocks = length(unique(feature_block_ids))

    theta = randn(BMFFloat, n_sample_blocks, n_feature_blocks) ./ BMFFloat(100.0)
    log_delta = zeros(BMFFloat, n_sample_blocks, n_feature_blocks)

    return BatchMatFacModel(X, Y, X_reg, Y_reg, 
                            mu, log_sigma, mu_reg, log_sigma_reg,
                            theta, log_delta, 
                            sample_block_ids, feature_block_ids,
                            feature_loss_names)

end

