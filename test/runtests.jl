
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

        my_idx_dict = BMF.ids_to_idx_dict([1,1,1,2,2,1,2,3,3,1,2,3,3])
        @test my_idx_dict == Dict(1 => [1,2,3,6,10], 
                                  2 => [4,5,7,11],
                                  3 => [8,9,12,13])

        d_subset = BMF.subset_idx_dict(my_idx_dict, 5:11)
        @test d_subset == Dict(1 => [6,10],
                               2 => [5,7,11],
                               3 => [8,9])
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
        A = BMF.batch_matrix(r_matrix, row_batch_ids, col_batches)
    
        @test A == BMF.BatchMatrix(r_matrix, [Dict("cat"=>[1,2],"dog"=>[3,4,5],"fish"=>[6]),
                                              Dict("cat"=>[1,2],"dog"=>[3,4,5],"fish"=>[6])],
                                              UnitRange[1:2,3:7])
        @test size(A) == (6,7)

        # Getindex
        B = A[3:6, 1:7]

        @test B.row_batch_dicts == [Dict("dog"=>[1,2,3], "fish"=>[4]),
                                    Dict("dog"=>[1,2,3], "fish"=>[4])]
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
        @test A.values == [Dict("cat"=>1., "dog"=>3., "fish"=>4.), 
                           Dict("cat"=>1., "dog"=>3., "fish"=>5.)]

        # Additive row update (partially overlapping blocks)
        A = BMF.batch_matrix(r_matrix, row_batch_ids, col_batches)
        B = A[2:4, 1:7]
        B.values = [Dict("cat"=>2., "dog"=>3.),
                    Dict("cat"=>-2.,"dog"=>-3.)]
        BMF.row_add!(A, 2:4, B)

        @test A.values == [Dict("cat"=>2., "dog"=>4., "fish"=>3.), 
                           Dict("cat"=>0., "dog"=>0., "fish"=>4.)]

        # Backpropagation for addition
        A = BMF.batch_matrix(r_matrix, row_batch_ids, col_batches)
        B = A[2:4, 1:7]
        #B.values = [2. -2.; 3. -3.]
        B.values = [Dict("cat"=>2., "dog"=>3.),
                    Dict("cat"=>-2.,"dog"=> -3.)]
        BMF.row_add!(A, 2:4, B)
        D_view = view(test_D, 2:4, :)
        (grad_D, grad_B) = gradient((d,b)->sum(d+b), D_view, B)
        
        @test grad_D == ones(3,7)
        @test grad_B.values == [Dict("cat"=>2., "dog"=>4.),
                                Dict("cat"=>5., "dog"=>10.)] # Just the number of entries for each value 

        # Backpropagation for multiplication
        C = ones(3,7)
        (grad_C, grad_B) = gradient((c,b)->sum(c*b), C, B)
        
        @test grad_C == [ 2. 2. -2. -2. -2. -2. -2.; 
                          3. 3. -3. -3. -3. -3. -3.; 
                          3. 3. -3. -3. -3. -3. -3.]
        @test grad_B.values == [Dict("cat"=>2., "dog"=>4.),
                                Dict("cat"=>5., "dog"=>10.)]

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

    @testset "ColBlockAgg" begin

        # Construction
        my_cba = BMF.ColBlockAgg([my_binerr, my_sqerr, my_noloss], col_block_ids)
        @test my_cba.col_blocks == [1:2, 3:5, 6:7]

        losses = my_cba(Y, D, missing_data)
        @test losses == test_losses 

        grad = gradient(y-> sum(my_cba(y,D,missing_data)), Y)[1]
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
    theta = BMF.batch_matrix(theta_values, row_batch_ids, noise_models)
    log_delta = BMF.batch_matrix(log_delta_values, row_batch_ids, noise_models)

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

    log_delta = BMF.batch_matrix(bm_values, bm_row_batches, col_batches)
    theta = BMF.batch_matrix(bm_values, bm_row_batches, col_batches)

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
        #test_grad_Y = [Dict(k=>test_grad_Y[k,j] for k=1:K) for j=1:N]
        @test grad_Y == test_grad_Y

        test_grad_theta = [zeros(n_batches) for _=1:2]
        test_grad_theta[1] .= 0.25 * n_logistic * div(M,n_batches)
        test_grad_theta[2] .= n_logistic * div(M,n_batches)
        test_grad_theta = [Dict(k=>test_grad_theta[j][k] for k=1:n_batches) for j=1:2]
        @test grad_theta.values == test_grad_theta
       
        test_grad_sigma = CUDA.zeros(N)
        @test grad_log_sigma == test_grad_sigma

        test_grad_delta = [zeros(n_batches) for _=1:2]
        test_grad_delta = [Dict(k=>test_grad_delta[j][k] for k=1:n_batches) for j=1:2]
        @test grad_log_delta.values == test_grad_delta

    end

    feature_loss_map = BMF.ColBlockAgg(Function[BMF.logistic_loss, BMF.quad_loss],
                                       col_batch_ranges)

    D = CUDA.zeros(M,N)
    #D[:,1:n_logistic] .= 0.5
    missing_data = zeros(Bool, size(D)...)

    @testset "Likelihood" begin

        # Computation
        loss = BMF.neg_log_likelihood(X, Y, mu, log_sigma, 
                                      theta, log_delta,
                                      feature_link_map, 
                                      feature_loss_map, D, missing_data)
        @test isapprox(loss, n_logistic * M * log(2))

        # Gradient
        new_D = CUDA.zeros(M,N)
        new_D[:,1:n_logistic] .= 0.5
        curried_loss = (X, Y, mu, log_sigma, 
                        theta, log_delta) -> BMF.neg_log_likelihood(X, Y, mu, log_sigma, 
                                                                    theta, log_delta,
                                                                    feature_link_map, 
                                                                    feature_loss_map, new_D,
                                                                    missing_data)
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
        test_grad_theta = [Dict(k=>test_grad_theta[j][k] for k=1:n_batches) for j=1:2]
        @test grad_theta.values == test_grad_theta
        test_grad_delta = [Dict(k=> 0. for k=1:n_batches) for j=1:2]
        @test grad_log_delta.values == test_grad_delta 

    end

    X_reg, Y_reg, mu_reg, log_sigma_reg = generate_regularizers(M, N, K)

    @testset "Priors" begin

        # Computation
        loss = BMF.neg_log_prior(X, X_reg, 
                                 Y, Y_reg, 
                                 mu, mu_reg, 
                                 log_sigma, log_sigma_reg)
        @test loss == 0

        # Gradient
        curried_prior = (X, Y, mu, log_sigma) -> BMF.neg_log_prior(X, X_reg,
                                                                   Y, Y_reg,
                                                                   mu, mu_reg,
                                                                   log_sigma, 
                                                                   log_sigma_reg)

        grad_X, grad_Y, grad_mu, grad_log_sigma = gradient(curried_prior, X, Y,
                                                           mu, log_sigma)
        # These should all have zero gradient
        @test grad_X == zero(grad_X)
        @test grad_Y == zero(grad_Y)
        @test grad_mu == zero(grad_mu)
        @test grad_log_sigma == zero(grad_log_sigma)

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
    
    log_delta = BMF.batch_matrix(bm_values, bm_row_batches, bm_col_batches)
    theta = BMF.batch_matrix(bm_values, bm_row_batches, bm_col_batches)
    #theta = BMF.BatchMatrix([zeros(4) for _=1:2],
    #                        [[1:5,6:10,11:15,16:20] for _=1:2],
    #                        [1:10,11:20])
    #log_delta = BMF.BatchMatrix([zeros(4) for _=1:2],
    #                            [[1:5,6:10,11:15,16:20] for _=1:2],
    #                            [1:10,11:20])

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

    fit!(my_model, A; max_epochs=200, capacity=M*N, lr=0.01)

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


function main()
   
    #util_tests()
    #batch_matrix_tests()
    col_block_map_tests()
    #model_params_tests()
    #model_core_tests()
    #adagrad_tests()
    #fit_tests()
    #io_tests()

end

main()


