

function simulate(X::AbstractMatrix, Y::AbstractMatrix, 
                  log_sigma::AbstractVector, mu::AbstractVector,
                  feature_noises::AbstractVector,
                  col_batch_ids::AbstractVector,
                  row_batch_ids::AbstractVector,
                  bshift_noise_param::Number,
                  sample_noise_param::AbstractVector;
                  bscale_noise_param::Number=1e-4)

    K,M = size(X)
    K2, N = size(Y)

    # Some input validation
    @assert K2 == K
    @assert N == length(log_sigma)
    @assert N == length(mu)
    @assert N == length(sample_noise_param)
    @assert M == length(row_batches[1])

    # Build model
    model = BatchMatFacModel(M, N, K, col_batch_ids, 
                             row_batch_ids,
                             feature_noises)
    # Set some of its parameters
    model.cscale.log_sigma = log_sigma
    model.cshift.mu = mu

    model.bscale.log_delta.values = [randn(size(v)...).*batch_noise_param]
    model.bshift.theta.values = [randn(size(v)...).*batch_noise_param]

    # Run the model in forward-mode; and then
    # sample data from the appropriate 
    # "noise" distribution
    Z = model()

    A = simulate(model.noise_model, Z, sample_noise_param)

    return A
end


