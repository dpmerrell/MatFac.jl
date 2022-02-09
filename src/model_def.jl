

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

        theta::Vector{Vector{<:Number}}
        log_delta::Vector{Vector{<:Number}}

        sample_group_ids::Vector{Vector{<:Union{Number,String}}}
        feature_group_ids::Vector{<:Union{Number,String}}

        feature_noise_models::Vector{String}

end


# Define an alias
BMFModel = BatchMatFacModel


# Define a convenience constructor
function BatchMatFacModel(X_reg, Y_reg, mu_reg, log_sigma_reg,
                          sample_block_dict, feature_block_ids,
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

    # Construct the batch parameter values
    unq_feature_blocks = unique(feature_block_ids)
    n_feature_blocks = length(unq_feature_blocks)
    n_sample_blocks_vec = [length(unique(sample_block_dict[k])) for k in unq_feature_blocks]
    theta_values = [randn(BMFFloat, nblocks) ./ BMFFloat(100.0) for nblocks in n_sample_blocks_vec ]
    log_delta_values = [zeros(BMFFloat, nblocks) for nblocks in n_sample_blocks_vec ]
   
    # Construct the vector of sample batch labels
    sample_block_ids = [copy(sample_block_dict[k]) for k in unq_feature_blocks]

    return BatchMatFacModel(X, Y, X_reg, Y_reg, 
                            mu, log_sigma, mu_reg, log_sigma_reg,
                            theta_values, log_delta_values, 
                            sample_block_ids, feature_block_ids,
                            feature_loss_names)

end


# Define some straightforward equality operators
function Base.:(==)(a::CuSparseMatrixCSC, b::CuSparseMatrixCSC)
    return SparseMatrixCSC(a) == SparseMatrixCSC(b)
end


function Base.:(==)(model_a::BMFModel, model_b::BMFModel)
    for fn in fieldnames(BMFModel)
        if !(getproperty(model_a, fn) == getproperty(model_b, fn)) 
            return false
        end
    end
    return true
end

