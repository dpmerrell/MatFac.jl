

function simulate(X::AbstractMatrix, Y::AbstractMatrix, 
                  logsigma::AbstractVector, mu::AbstractVector,
                  col_batch_ids::AbstractVector,
                  row_batch_ids::AbstractVector,
                  bshift_noise_param::Number,
                  feature_noises::AbstractVector,
                  sample_noise_param::AbstractVector;
                  bscale_noise_param::Number=1e-4)

    K,M = size(X)
    K2, N = size(Y)

    # Some input validation
    @assert K2 == K
    @assert N == length(logsigma)
    @assert N == length(mu)
    @assert N == length(sample_noise_param)
    @assert M == length(row_batch_ids[1])

    # Build model
    model = BatchMatFacModel(M, N, K, col_batch_ids, 
                             row_batch_ids,
                             feature_noises)
    # Set some of its parameters
    model.cscale.logsigma = logsigma
    model.cshift.mu = mu

    model.bscale.logdelta.values = map(v->randn(size(v)...).*bscale_noise_param, model.bscale.logdelta.values)
    model.bshift.theta.values = map(v->randn(size(v)...).*bshift_noise_param, model.bshift.theta.values)

    # Run the model in forward-mode; and then
    # sample data from the appropriate 
    # "noise" distribution
    Z = model()

    A = simulate(model.noise_model, Z, sample_noise_param)

    return A
end


