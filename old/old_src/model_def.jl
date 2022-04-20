

export BatchMatFacModel


mutable struct BatchMatFacModel

        ################################
        # Embedding matrices 
        # (and their quadratic regularizers)
        X::BMFMat
        Y::BMFMat
        X_reg::Vector{<:BMFRegMat}
        Y_reg::Vector{<:BMFRegMat}

        ################################
        # Feature parameters 
        # (and their quadratic regularizers)
        mu::BMFVec
        log_sigma::BMFVec
        mu_reg::BMFRegMat
        log_sigma_reg::BMFRegMat

        ################################
        # Batch parameters
        # (and their quadratic regularizers)
        theta_values::Vector{<:Dict}
        theta_reg::Number
        log_delta_values::Vector{<:Dict}
        log_delta_reg::Number
        sample_batch_ids::Vector{<:Vector{<:KeyType}}
        feature_batch_ids::Vector{<:KeyType}

        ################################
        # Noise models
        feature_noise_models::Vector{String}

end


# Define an alias
BMFModel = BatchMatFacModel


# Define a convenience constructor
function BatchMatFacModel(X_reg, Y_reg, mu_reg, log_sigma_reg,
                          sample_batch_dict::Dict{T,Vector{U}}, 
                          feature_batch_ids::Vector{T},
                          feature_loss_names;
                          theta_reg::Number=1.0, 
                          log_delta_reg::Number=1.0) where T<:KeyType where U<:KeyType

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

    # NOTE: the order of appearance in `feature_batch_ids`
    #       at construction determines the order of 
    #       obects in BatchMatrix quantities (theta, sigma)
    unq_feature_batches = unique(feature_batch_ids)
    n_feature_batches = length(unq_feature_batches)

    # Construct the batch parameters 
    theta_values = [Dict{U,BMFFloat}() for _=1:n_feature_batches]
    log_delta_values = [Dict{U,BMFFloat}() for _=1:n_feature_batches]
    for (j,ufb) in enumerate(unq_feature_batches)
        for sample_batch in unique(sample_batch_dict[ufb])
            theta_values[j][sample_batch] = randn(BMFFloat) ./ BMFFloat(100.0)
            log_delta_values[j][sample_batch] = zero(BMFFloat)
        end
    end

    # Construct the vector of sample batch labels
    sample_batch_ids = [copy(sample_batch_dict[k]) for k in unq_feature_batches]

    return BatchMatFacModel(X, Y, X_reg, Y_reg, 
                            mu, log_sigma, mu_reg, log_sigma_reg,
                            theta_values, theta_reg, 
                            log_delta_values, log_delta_reg, 
                            sample_batch_ids, feature_batch_ids,
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

