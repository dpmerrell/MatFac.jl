
using Test, BatchMatFac, CUDA, Zygote, HDF5, SparseArrays, LinearAlgebra, ScikitLearnBase

BMF = BatchMatFac

function util_tests()

    @testset "Utility functions" begin

        @test BMF.is_contiguous([2,2,2,1,1,4,4,4])
        @test BMF.is_contiguous(["cat","cat","dog","dog","fish"])
        @test !BMF.is_contiguous([1,1,5,5,1,3,3,3])

        @test BMF.ids_to_ranges([2,2,2,1,1,4,4,4]) == [1:3, 4:5, 6:8]
        my_ranges = BMF.ids_to_ranges(["cat","cat","dog","dog","fish"])
        @test my_ranges == [1:2,3:4,5:5]

        @test BMF.subset_ranges(my_ranges, 2:5) == ([2:2,3:4,5:5], 1, 3)

        my_idx_vecs = BMF.ids_to_idx_vecs([1,1,1,2,2,1,2,3,3,1,2,3,3])
        @test my_idx_vecs == [[1,2,3,6,10], 
                              [4,5,7,11],
                              [8,9,12,13]]

        v_subset, kept_idx = BMF.subset_idx_vecs(my_idx_vecs, 5:11)
        @test kept_idx == [1,2,3]
        @test v_subset == [[6,10],
                           [5,7,11],
                           [8,9]]
        v_subset, kept_idx = BMF.subset_idx_vecs(my_idx_vecs, 1:7)
        @test kept_idx == [1,2]
        @test v_subset == [[1,2,3,6],
                           [4,5,7]]
    end

end


function batch_matrix_tests()

    @testset "BatchMatrix" begin
        
        row_batch_ids = [["cat","cat","dog","dog","dog","fish"],
                         ["cat","cat","dog","dog","dog","fish"]]
        col_batches = [1,1,2,2,2,2,2]
        r_matrix = [Dict("cat"=>1., "dog"=>2., "fish"=>3.),
                    Dict("cat"=>1., "dog"=>2., "fish"=>4.)]

        # Construction
        A = BMF.batch_matrix(deepcopy(r_matrix), row_batch_ids, col_batches)
    
        @test A == BMF.BatchMatrix(UnitRange[1:2,3:7],
                                   [[1,2,3],[1,2,3]],
                                   [[[1,2],[3,4,5],[6]],
                                    [[1,2],[3,4,5],[6]]],
                                   [[1.,2.,3.],[1.,2.,4.]]
                                              )
        @test size(A) == (6,7)

        # Getindex
        B = A[3:6, 1:7]

        @test B.unq_row_batches == [[2,3],[2,3]]
        @test B.row_batch_idx == [[[1,2,3],[4]],
                                  [[1,2,3],[4]]]
        @test B.col_batches == [1:2, 3:7]

        # Addition by dense matrix
        C = ones(6,7)
        D = C + A
        
        test_D = [2 2 2 2 2 2 2;
                  2 2 2 2 2 2 2;
                  3 3 3 3 3 3 3;
                  3 3 3 3 3 3 3;
                  3 3 3 3 3 3 3;
                  4 4 5 5 5 5 5]

        @test D == test_D

        # Multiplication by dense matrix
        D = (-C) * A
        test_D .-= C
        test_D .*= -1

        @test D == test_D

        # Additive row update (100% overlapping batches)
        #B.values = [[1., 1.], [1., 1.]]
        map!(x->1, B, B)
        BMF.row_add!(A, 3:6, B)
        @test A.values == [[1., 3., 4.], 
                           [1., 3., 5.]]

        # Additive row update (partially overlapping blocks)
        A = BMF.batch_matrix(deepcopy(r_matrix), row_batch_ids, col_batches)
        B = A[2:4, 1:7]
        B.values = [[2., 3.],
                    [-2.,-3.]]
        BMF.row_add!(A, 2:4, B)

        @test A.values == [[2., 4., 3.], 
                           [0., 0., 4.]]

        # Backpropagation for addition
        A = BMF.batch_matrix(deepcopy(r_matrix), row_batch_ids, col_batches)
        B = A[2:4, 1:7]
        B.values = [[2., 3.],
                    [-2., -3.]]
        BMF.row_add!(A, 2:4, B)
        D_view = view(test_D, 2:4, :)
        (grad_D, grad_B) = gradient((d,b)->sum(d+b), D_view, B)
        
        @test grad_D == ones(3,7)
        @test grad_B.values == [[2., 4.],
                                [5., 10.]] # Just the number of entries for each value 

        # Backpropagation for multiplication
        C = ones(3,7)
        (grad_C, grad_B) = gradient((c,b)->sum(c*b), C, B)
        
        @test grad_C == [ 2. 2. -2. -2. -2. -2. -2.; 
                          3. 3. -3. -3. -3. -3. -3.; 
                          3. 3. -3. -3. -3. -3. -3.]
        @test grad_B.values == [[2., 4.],
                                [5., 10.]]

    end

end


function col_block_map_tests()

    my_logistic =  x -> 1 ./ (1 .+ exp.(-x))
    my_exp = x -> exp.(x)
    my_noloss_link = x -> x

    my_sqerr = BMF.quad_loss #(x,y,m) -> 0.5.*(x .- y).^2
    my_binerr = BMF.logistic_loss #(x,y,m) -> - y .* log.(x) - (1 .- y) .* log.(1 .- x)
    my_noloss = BMF.noloss_loss

    col_block_ids = ["cat", "cat", "dog", "dog", "dog", "fish", "fish"]
    
    X = CUDA.zeros(2,7)
    Y = CUDA.ones(2,7)
    Y[:,1:2] .= 0.5
    Y[:,6:7] .= 0.0 

    @testset "ColBlockMap" begin
    
        # Construction
        my_cbm = BMF.ColBlockMap([my_logistic, my_exp, my_noloss_link], col_block_ids)
        @test my_cbm.col_blocks == [1:2, 3:5, 6:7]

        A = my_cbm(X)

        @test A == Y
        
        grad = gradient(x->sum(my_cbm(x)), X)[1]
        correct_grad = CUDA.ones(2,7)
        correct_grad[:,1:2] .= 0.25

        @test grad == correct_grad

    end

    D = CUDA.ones(2,7)
    test_losses = CUDA.zeros(2,7)
    test_losses[:,1:2] .= -log(0.5)
    test_losses[:,6:7] .= 0.0
    missing_data = CUDA.zeros(Bool, size(D)...)
    missing_mask = (0.5 .* missing_data)
    nonmissing = (!).(missing_data)

    @testset "ColBlockAgg" begin

        # Construction
        my_cba = BMF.ColBlockAgg([my_binerr, my_sqerr, my_noloss], col_block_ids)
        @test my_cba.col_blocks == [1:2, 3:5, 6:7]

        losses = my_cba(Y, D, missing_mask, nonmissing)
        @test losses == test_losses 

        grad = gradient(y-> sum(my_cba(y,D,missing_mask, nonmissing)), Y)[1]
        correct_grad = CUDA.zeros(2,7)
        correct_grad[:,1:2] .= -2.0
        
        @test grad == correct_grad

    end

end



function simulate_data(M, N, K, row_batches, n_logistic)
    
    X = BMF.BMFMat(CUDA.randn(K, M) .* 0.01)
    Y = BMF.BMFMat(CUDA.randn(K, N) .* 0.01)
    
    log_sigma = BMF.BMFVec(CUDA.randn(N) .* 0.01)
    mu = BMF.BMFVec(CUDA.randn(N) .* 0.01)

    n_batches = length(unique(row_batches))
    #theta_values = [randn(n_batches) .* 0.1 for _=1:2]
    #log_delta_values = [randn(n_batches) .* 0.1 for _=1:2]
    theta_values = [Dict(i => randn()*0.1 for i=1:n_batches) for _=1:2]
    log_delta_values = [Dict(i => randn()*0.1 for i=1:n_batches) for _=1:2]

    noise_models = [repeat(["logistic"], n_logistic);repeat(["normal"], N-n_logistic)]
    if n_logistic > 0
        noise_map = BMF.ColBlockMap([BMF.logistic_link, BMF.quad_link], noise_models)
    else
        noise_map = BMF.ColBlockMap([BMF.quad_link], noise_models)
    end

    row_batch_ids = [row_batches, row_batches]
    theta = BMF.batch_matrix(deepcopy(theta_values), row_batch_ids, noise_models)
    log_delta = BMF.batch_matrix(deepcopy(log_delta_values), row_batch_ids, noise_models)

    A = BMF.forward(X, Y, mu, log_sigma, theta, log_delta,
                   noise_map)

    if n_logistic > 0
        A_logistic = view(A, :, 1:n_logistic)
        A_logistic[A_logistic .>= 0.5] .= 1.0
        A_logistic[A_logistic .< 0.5] .= 0.0
    end

    return X, Y, log_sigma, mu, log_delta, theta, A
end


function generate_regularizers(M, N, K)
    
    X_reg = fill(BMF.CuSparseMatrixCSC{Float32}(SparseMatrixCSC(I(M))), K)
    Y_reg = fill(BMF.CuSparseMatrixCSC{Float32}(SparseMatrixCSC(I(N))), K)
    
    mu_reg = BMF.CuSparseMatrixCSC{Float32}(SparseMatrixCSC(I(N)))
    sigma_reg = BMF.CuSparseMatrixCSC{Float32}(SparseMatrixCSC(I(N)))
    
    return X_reg, Y_reg, mu_reg, sigma_reg
end


getvalues(a::BMF.BatchMatrix) = [collect(values(d)) for d in a.values]
getvalues(a::AbstractArray) = a


function all_equal(mp::BMF.ModelParams, x::Number)

    for pn in propertynames(mp)
        arr = getvalues(getproperty(mp, pn))
        # Check whether any entry is *not* approximately x
        if !reduce((a,b)-> a & all(b), map(a->isapprox.(a,x), arr) ; init=true) 
            return false
        end
    end

    return true
end


function model_params_tests()

    M = 10
    N = 20
    K = 5

    X = zeros(K,M)
    Y = zeros(K,N)
    mu = zeros(N)
    log_sigma = zeros(N)

    theta = BMF.batch_matrix([Dict(1=>0., 2=>0.),Dict(1=>0., 2=>0.)],
                             [[fill(1,5); fill(2,5)], [fill(1,5); fill(2,5)]],
                             [fill(1,10); fill(2,10)])
    log_delta = BMF.batch_matrix([Dict(1=>0., 2=>0.),Dict(1=>0., 2=>0.)],
                             [[fill(1,5); fill(2,5)], [fill(1,5); fill(2,5)]],
                             [fill(1,10); fill(2,10)])

    a = BMF.ModelParams(X,Y,mu,log_sigma,theta,log_delta)

    @testset "ModelParams" begin

        # map! test
        map!(x -> x .+ 3.14, a, a)
        @test all_equal(a, 3.14)
        
        # zero test
        b = zero(a)
        @test all_equal(b, 0.0) & all_equal(a, 3.14)

    end

end


function model_core_tests()
    
    M = 1000
    N = 2000
    K = 100
    n_batches = 4

    X = CUDA.zeros(K,M)
    Y = CUDA.zeros(K,N)
    log_sigma = CUDA.zeros(N)
    mu = CUDA.zeros(N)

    row_batches = repeat(1:n_batches, inner=div(M,n_batches))
    row_batch_ranges = BMF.ids_to_ranges(row_batches)
    n_logistic = div(N,2)
    col_batches = repeat(1:2, inner=n_logistic)
    col_batch_ranges = BMF.ids_to_ranges(col_batches)

    bm_values = [Dict(i => 0. for i=1:n_batches) for _=1:2]
    bm_row_batches = [row_batches, row_batches]

    log_delta = BMF.batch_matrix(deepcopy(bm_values), bm_row_batches, col_batches)
    theta = BMF.batch_matrix(deepcopy(bm_values), bm_row_batches, col_batches)

    feature_link_map = BMF.ColBlockMap(Function[BMF.logistic_link, BMF.quad_link],
                                       col_batch_ranges)

    @testset "Forward" begin

        # Computation
        A = BMF.forward(X, Y, mu, log_sigma, 
                        theta, log_delta,
                        feature_link_map)
       
        test_A = CUDA.zeros(M,N)
        test_A[:,1:n_logistic] .= 0.5

        @test A == test_A

        # Gradient
        curried_forward_sum = (X, Y, mu, log_sigma,
                               theta, log_delta) -> sum(BMF.forward(X, Y, mu, log_sigma, 
                                                                    theta, log_delta,
                                                                    feature_link_map))
       
        grad_X, grad_Y, grad_mu, grad_log_sigma,
        grad_theta, grad_log_delta = gradient(curried_forward_sum, X, Y, mu,
                                                                   log_sigma,
                                                                   theta,
                                                                   log_delta)

        test_grad_X = CUDA.zeros(K, M)
        @test grad_X == test_grad_X

        test_grad_Y = CUDA.zeros(K, N)
        @test grad_Y == test_grad_Y

        test_grad_theta = [zeros(n_batches) for _=1:2]
        test_grad_theta[1] .= 0.25 * n_logistic * div(M,n_batches)
        test_grad_theta[2] .= n_logistic * div(M,n_batches)
        #test_grad_theta = [Dict(k=>test_grad_theta[j][k] for k=1:n_batches) for j=1:2]
        @test grad_theta.values == test_grad_theta
       
        test_grad_sigma = CUDA.zeros(N)
        @test grad_log_sigma == test_grad_sigma

        test_grad_delta = [zeros(n_batches) for _=1:2]
        #test_grad_delta = [Dict(k=>test_grad_delta[j][k] for k=1:n_batches) for j=1:2]
        @test grad_log_delta.values == test_grad_delta

    end

    feature_loss_map = BMF.ColBlockAgg(Function[BMF.logistic_loss, BMF.quad_loss],
                                       col_batch_ranges)

    D = CUDA.zeros(M,N)
    #D[:,1:n_logistic] .= 0.5
    missing_data = CUDA.zeros(Bool, size(D)...)
    missing_mask = (0.5 .* missing_data)
    nonmissing = (!).(missing_data)

    @testset "Likelihood" begin

        # Computation
        loss = BMF.neg_log_likelihood(X, Y, mu, log_sigma, 
                                      theta, log_delta,
                                      feature_link_map, 
                                      feature_loss_map, D, 
                                      missing_mask,
                                      nonmissing)
        @test isapprox(loss, (n_logistic * M * log(2))/(M*N))

        # Gradient
        new_D = CUDA.zeros(M,N)
        new_D[:,1:n_logistic] .= 0.5
        curried_loss = (X, Y, mu, log_sigma, 
                        theta, log_delta) -> BMF.neg_log_likelihood(X, Y, mu, log_sigma, 
                                                                    theta, log_delta,
                                                                    feature_link_map, 
                                                                    feature_loss_map, new_D,
                                                                    missing_mask,
                                                                    nonmissing)
        grad_X, grad_Y, grad_mu, grad_log_sigma,
        grad_theta, grad_log_delta = gradient(curried_loss, X, Y, mu, log_sigma,
                                                            theta, log_delta)
      
        # Compare autograd gradients against known, true gradients
        @test grad_X == CUDA.zeros(K,M)
        @test grad_Y == CUDA.zeros(K,N)
        test_grad_mu = CUDA.zeros(N)
        @test grad_mu == test_grad_mu 
        @test grad_log_sigma == CUDA.zeros(N)
        test_grad_theta = [zeros(n_batches) for _=1:2]
        #test_grad_theta = [Dict(k=>test_grad_theta[j][k] for k=1:n_batches) for j=1:2]
        @test grad_theta.values == test_grad_theta
        #test_grad_delta = [Dict(k=> 0. for k=1:n_batches) for j=1:2]
        test_grad_delta = deepcopy(test_grad_theta)
        @test grad_log_delta.values == test_grad_delta 

    end

    X_reg, Y_reg, mu_reg, log_sigma_reg = generate_regularizers(M, N, K)
    theta_reg = 1.0
    log_delta_reg = 1.0

    @testset "Priors" begin

        # Computation
        loss = BMF.neg_log_prior(X, X_reg, 
                                 Y, Y_reg, 
                                 mu, mu_reg, 
                                 log_sigma, log_sigma_reg,
                                 theta, theta_reg,
                                 log_delta, log_delta_reg)
        @test loss == 0

        # Gradient
        curried_prior = (X, Y, mu, log_sigma, 
                         theta, log_delta) -> BMF.neg_log_prior(X, X_reg,
                                                                Y, Y_reg,
                                                                mu, mu_reg,
                                                                log_sigma, 
                                                                log_sigma_reg,
                                                                theta, theta_reg,
                                                                log_delta, 
                                                                log_delta_reg)

        grad_X, grad_Y, grad_mu, grad_log_sigma,
        grad_theta, grad_log_delta = gradient(curried_prior, X, Y,
                                              mu, log_sigma, theta, log_delta)

        # These should all have zero gradient
        @test grad_X == zero(grad_X)
        @test grad_Y == zero(grad_Y)
        @test grad_mu == zero(grad_mu)
        @test grad_log_sigma == zero(grad_log_sigma)
        @test grad_theta == zero(grad_theta)
        @test grad_log_delta == zero(grad_log_delta)
    end
end


function adagrad_tests()

    M = 10
    N = 20
    K = 5
    n_batches = 2
    n_logistic = div(N,2)
    
    X = zeros(K,M)
    Y = zeros(K,N)

    mu = zeros(N)
    log_sigma = zeros(N)

    bm_values = [Dict(i => 0. for i=1:n_batches) for _=1:2]
    row_batches = repeat(1:n_batches, inner=div(M,n_batches))
    bm_row_batches = [row_batches, row_batches]
    bm_col_batches = repeat(1:2, inner=n_logistic)
    
    log_delta = BMF.batch_matrix(deepcopy(bm_values), bm_row_batches, bm_col_batches)
    theta = BMF.batch_matrix(deepcopy(bm_values), bm_row_batches, bm_col_batches)

    params = BMF.ModelParams(X,Y,mu,log_sigma,theta,log_delta)

    adagrad = BMF.adagrad_updater(params; epsilon=0.01)

    grads = map(x-> x .+ 1, params)


    @testset "AdaGrad" begin
       
        lr = 0.01
        adagrad(params, grads; lr=lr)
        @test all_equal(adagrad.sum_sq, 1.01)
        @test all_equal(params, -lr/sqrt(1.01))
        @test all_equal(grads, 1.0)

    end

end


function fit_tests()

    M = 20
    N = 10
    K = 5
    n_batches = 4

    X_reg, Y_reg, mu_reg, sigma_reg = generate_regularizers(M, N, K)
    
    n_logistic = div(N,2)
    feature_loss_names = [repeat(["logistic"],n_logistic); repeat(["normal"],N-n_logistic)] 
    
    sample_batch_ids = repeat(1:n_batches, inner=div(M,n_batches))
    sample_batch_dict = Dict([k => sample_batch_ids for k in ("logistic","normal")])

    my_model = BatchMatFacModel(X_reg, Y_reg, mu_reg, sigma_reg,
                                sample_batch_dict, feature_loss_names,
                                feature_loss_names)

    test_X, test_Y, 
    test_log_sigma, test_mu, 
    test_log_delta, test_theta, A = simulate_data(M, N, K, sample_batch_ids, n_logistic)

    fit!(my_model, A; max_epochs=200, capacity=div(M*N,2), lr=0.01, verbose=false)

    # Just put an empty test here to show we run to completion
    @testset "Fit" begin
        @test true
    end

end



function io_tests()

    test_hdf_path = "test.hdf"

    # Some simpler objects
    my_scalar = 3.14159
    str_vec = ["cat", "dog", "fish"]
    num_vec = [1.0, 2.0, 3.0]
    spmat = sprand(10,10,0.2)
    spmat_vec = [sprand(10,10,0.2) for _=1:5]

    # The model object
    M = 20
    N = 10
    K = 5
    n_batches = 4
    X_reg, Y_reg, mu_reg, sigma_reg = generate_regularizers(M, N, K)
    sample_batch_ids = repeat(1:n_batches, inner=div(M,n_batches))
    sample_batch_dict = Dict([k => sample_batch_ids for k in ("logistic","normal")])
    n_logistic = div(N,2)
    feature_loss_names = [repeat(["logistic"],n_logistic); repeat(["normal"],N-n_logistic)] 
    my_model = BatchMatFacModel(X_reg, Y_reg, mu_reg, sigma_reg,
                                sample_batch_dict, feature_loss_names,
                                feature_loss_names)


    # Write tests
    @testset "File IO" begin
        h5open(test_hdf_path, "w") do file
      
            write(file, "/my_scalar", my_scalar)
            write(file, "/str_vec", str_vec)
            write(file, "/num_vec", num_vec)
            write(file, "/spmat", spmat)
            write(file, "/spmat_vec", spmat_vec)
            write(file, "/my_model", my_model)

        end

        h5open(test_hdf_path, "r") do file
            
            sc = BMF.readtype(file, "/my_scalar", Float64)
            @test sc == my_scalar

            sv = BMF.readtype(file, "/str_vec", Vector{String})
            @test sv == str_vec

            nv = BMF.readtype(file, "/num_vec", Vector{Float64})
            @test nv == num_vec

            sp = BMF.readtype(file, "/spmat", SparseMatrixCSC)
            @test sp == spmat

            spv = BMF.readtype(file, "/spmat_vec", Vector{<:SparseMatrixCSC})
            @test spv == spmat_vec

            mdl = BMF.readtype(file, "/my_model", BMF.BMFModel)
            @test mdl == my_model
        end
    end

    rm(test_hdf_path)

end

function simulation_tests()

    #@testset "Simulation SNR tests" begin

    #    # XOR noise for binary signal
    #    (xor_signal,), (xor_noise,) = BMF.decompose_xor_signal(0.5; snr=10.0)
    #    @test isapprox(xor_signal*(1-xor_signal)/(xor_noise * (1-xor_noise)), 10.0)
    #    @test isapprox(xor_signal*(1-xor_noise) + (1-xor_signal)*xor_noise, 0.5)

    #    # backtrack mean/var through logistic function
    #    logistic_mean, logistic_var = BMF.decompose_logistic_signal(0.01; input_mtv=10.0)
    #    @test isapprox(abs(logistic_mean)/logistic_var, 10.0)
    #    @test logistic_var > 0.0
    #    test_mean, test_var = BMF.forward_logistic_signal(logistic_mean, logistic_var)
    #    @test isapprox(test_mean, 0.01, atol=0.01)

    #    # Backtrack mean/var through exp function 
    #    exp_mean, exp_var = BMF.decompose_exp_signal(1.0, 0.0001)
    #    @test isapprox(exp_mean, 0.0, atol=1e-3)
    #    @test isapprox(exp_var, 0.0001, atol=1e-5)
    #    @test exp_var > 0.0

    #    # Additive noise for real-valued signal
    #    (sig_mean, sig_var),
    #    (noise_mean, noise_var) = BMF.decompose_additive_signal(3.14, 2.7; 
    #                                                            snr=10.0,
    #                                                            noise_mean=1.5)
    #    @test isapprox(sig_var/noise_var, 10.0)
    #    @test isapprox(noise_mean, 1.5)
    #    @test isapprox(noise_mean+sig_mean, 3.14)
    #    @test isapprox(noise_var+sig_var, 2.7)
    #    @test sig_var > 0.0
    #    @test noise_var > 0.0

    #    # Multiplicative noise
    #    (sig_mean, sig_var), 
    #    (noise_mean, noise_var) = BMF.decompose_mult_signal(12.34, 32.1;
    #                                                        noise_mean=1.0,
    #                                                        snr=10.0)
    #    @test isapprox(sig_var/noise_var, 10.0)
    #    @test isapprox(noise_mean, 1.0)
    #    @test isapprox(sig_mean*noise_mean, 12.34)
    #    @test isapprox(sig_var*noise_var + noise_var*(sig_mean^2) + sig_var*(noise_mean^2), 32.1)
    #    @test sig_var > 0.0
    #    @test noise_var > 0.0

    #    # Matrix factorization
    #    logsigma_moments, 
    #    mu_moments,  
    #    logdelta_moments, 
    #    theta_moments = BMF.decompose_matfac_signal(-3.14, 9.0;
    #                                                sigma_snr=20.0,
    #                                                mu_snr=1000.0,
    #                                                delta_snr=10.0,
    #                                                theta_snr=10.0)
    #    z_mean, z_var = BMF.forward_matfac_signal(0.0, 1.0,
    #                                              logsigma_moments, mu_moments,
    #                                              logdelta_moments, theta_moments)
    #    @test isapprox(z_mean, -3.14)
    #    @test isapprox(z_var, 9.0)

    #   
    #    # Matrix factorization -- normal data
    #    logsigma_moments, mu_moments,  
    #    logdelta_moments, theta_moments, 
    #    sample_moments = BMF.decompose_normal_data_signal(12.34, 32.1; 
    #                                                      mu_snr=100.0,
    #                                                      delta_snr=10.0,
    #                                                      theta_snr=10.0,
    #                                                      sample_snr=10.0,
    #                                                      )
    #    z_mean, z_var = BMF.forward_matfac_signal(0.0, 1.0,
    #                                              logsigma_moments, mu_moments,
    #                                              logdelta_moments, theta_moments)
    #    z_mean += sample_moments[1]
    #    z_var += sample_moments[2]
    #    @test isapprox(z_mean, 12.34)
    #    @test isapprox(z_var, 32.1)

    #    # Matrix factorization -- bernoulli data
    #    logsigma_moments, mu_moments,  
    #    logdelta_moments, theta_moments, 
    #    sample_moments = BMF.decompose_bernoulli_data_signal(0.00001;
    #                              mu_snr=100.0,
    #                              delta_snr=10.0,
    #                              theta_snr=10.0,
    #                              logistic_mtv=10.0,
    #                              sample_snr=100.0,
    #                              )
    #    z_mean, z_var = BMF.forward_matfac_signal(0.0, 1.0,
    #                                              logsigma_moments, mu_moments,
    #                                              logdelta_moments, theta_moments)
    #    logistic_mean, logistic_var = BMF.forward_logistic_signal(z_mean, z_var)
    #    (data_mean, ) = BMF.forward_xor_signal(logistic_mean, sample_moments)
    #    
    #    # This will only hold in a *very* rough sense.
    #    @test isapprox(data_mean, 0.00001, atol=0.001)

    #    # Compute signal contributions for 
    #    # a dataset with column batches
    #    data_moments = [(3.14, 2.7), (0.001,), (12.34, 32.1)]
    #    feature_batch_ids = ["cat", "cat", "dog", "dog", "dog", "fish"]
    #    feature_losses = ["normal", "normal", "logistic", "logistic", "logistic", "normal"]
    #    col_batch_characteristics = BMF.decompose_all_data_signal(data_moments,
    #                                                              feature_batch_ids,
    #                                                              feature_losses;
    #                                                              mu_snr=100.0,
    #                                                              delta_snr=10.0,
    #                                                              theta_snr=10.0,
    #                                                              logistic_mtv=7.0,
    #                                                              sample_snr=10.0)
    #    cat_char, dog_char, fish_char = col_batch_characteristics

    #    # "cat" batch
    #    z_mean, z_var = BMF.forward_matfac_signal(0.0, 1.0,
    #                                              cat_char[1], cat_char[2],
    #                                              cat_char[3], cat_char[4])
    #    z_mean += cat_char[5][1]
    #    z_var += cat_char[5][2]
    #    @test isapprox(z_mean, data_moments[1][1])
    #    @test isapprox(z_var, data_moments[1][2])

    #    # "dog" batch
    #    z_mean, z_var = BMF.forward_matfac_signal(0.0, 1.0,
    #                                              dog_char[1], dog_char[2],
    #                                              dog_char[3], dog_char[4])
    #    logistic_mean, logistic_var = BMF.forward_logistic_signal(z_mean, z_var)
    #    (data_mean,) = BMF.forward_xor_signal(logistic_mean, sample_moments)
    #    @test isapprox(data_mean, data_moments[2][1], atol=0.002)
    #    
    #    # "fish" batch
    #    z_mean, z_var = BMF.forward_matfac_signal(0.0, 1.0,
    #                                              fish_char[1], fish_char[2],
    #                                              fish_char[3], fish_char[4])
    #    z_mean += fish_char[5][1]
    #    z_var += fish_char[5][2]
    #    @test isapprox(z_mean, data_moments[3][1])
    #    @test isapprox(z_var, data_moments[3][2])
    #end

    @testset "Data Simulation" begin

        M = 1000
        N = 2000
        K = 10
        n_col_batches = 2
        n_batches = 10

        col_batch_ids = repeat([string("colbatch_",i) for i=1:n_col_batches],inner=div(N,n_col_batches))
        row_batch_ids = Vector{String}[vcat([repeat([string("rowbatch_",i,"_",j) for i=1:n_batches],
                                                    inner=div(M,n_batches)
                                                   ) for j=1:n_col_batches
                                            ]...
                                           )
                                      ]

        Q_X = sprandn(M,M, 0.0005)
        P_X = Q_X*transpose(Q_X) + sparse(1.0.*I(M)) 
        X_reg = SparseMatrixCSC[deepcopy(P_X) for _=1:K]

        Q_Y = sprandn(N,N, 0.0005)
        P_Y = Q_Y*transpose(Q_Y) + sparse(1.0.*I(N)) 
        Y_reg = SparseMatrixCSC[deepcopy(P_Y) for _=1:K]

        logsigma_moments_vec = [(0.0, 0.25), (12.3, 32.1)]
        mu_moments_vec = [(-5.0, 0.25), (20.0, 50.0)]

        # Generate normal-distributed values, parameterized
        # by precision matrix
        mat = BMF.normal_prec(P_X; n_samples=K)
        @test size(mat) == (M,K)
        v = BMF.normal_prec(P_X)
        @test size(v) == (M,)

        # Simulate model parameters
        params = BMF.simulate_params(X_reg, Y_reg,
                                     row_batch_ids,
                                     col_batch_ids,
                                     logsigma_moments_vec,
                                     mu_moments_vec,
                                     logdelta_moments_vec,
                                     theta_moments_vec)
        @test size(params.X) == (K,M)
        @test size(params.Y) == (K,N)
        @test size(params.mu) == (N,)
        @test size(params.sigma) == (N,)
    end

end


function main()
   
    #util_tests()
    #batch_matrix_tests()
    #col_block_map_tests()
    #model_params_tests()
    #model_core_tests()
    #adagrad_tests()
    #fit_tests()
    #io_tests()
    simulation_tests()


end

main()


