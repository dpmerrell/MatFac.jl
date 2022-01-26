
using Test, BatchMatFac, CUDA, Zygote, HDF5, SparseArrays, LinearAlgebra

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
    end

end


function block_matrix_tests()

    @testset "BlockMatrix" begin
        
        row_blocks = ["cat","cat","dog","dog","dog","fish"]
        col_blocks = [1,1,2,2,2,2,2]
        r_matrix = [1. 1.; 2. 2.; 3. 4.]

        # Construction
        A = BMF.block_matrix(r_matrix, row_blocks, col_blocks)
    
        @test A == BMF.BlockMatrix(r_matrix, UnitRange[1:2,3:5,6:6], 
                                             UnitRange[1:2,3:7])
        @test size(A) == (6,7)

        # view
        A_copy = BMF.BlockMatrix(copy(A.values), A.row_ranges, A.col_ranges)
        A_view = view(A_copy, 3:6, 1:7)
        @test A_view.row_ranges == [1:3, 4:4]
        @test A_view.col_ranges == [1:2, 3:7]

        A_view.values .+= 1.0
        @test A_copy.values[2:3,:] == (A.values[2:3,:] .+ 1.0)

        # Getindex
        B = A[3:6, 1:7]

        @test B.row_ranges == [1:3, 4:4]
        @test B.col_ranges == [1:2, 3:7]

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

        # Additive row update (100% overlapping blocks)
        B.values = [1. 1.; 1. 1.]
        BMF.row_add!(A, 3:6, B)
        @test A.values == [1. 1.; 3. 3.; 4. 5.]

        # Additive row update (partially overlapping blocks)
        A = BMF.block_matrix(r_matrix, row_blocks, col_blocks)
        B = A[2:4, 1:7]
        B.values = [2. -2.; 3. -3.]
        BMF.row_add!(A, 2:4, B)

        @test A.values == [2. 0.; 4. 0.; 3. 4.]

        # Additive column update (100% overlapping blocks)
        A = BMF.block_matrix(r_matrix, row_blocks, col_blocks)
        B = A[1:6, 2:4]
        B.values = [1. 1.; 1. 1.; 1. 1.]
        BMF.col_add!(A, 2:4, B)
        @test A.values == [1.5 1.4; 2.5 2.4; 3.5 4.4]

        ## Additive column update (partially overlapping blocks)
        #A = BMF.block_matrix(r_matrix, row_blocks, col_blocks)
        #B = A[2:4, 1:7]
        #B.values = [2. -2.; 3. -3.]
        #BMF.row_add!(A, 2:4, B)

        #@test A.values == [2. 0.; 4. 0.; 3. 4.]
     

        # Backpropagation for addition
        A = BMF.block_matrix(r_matrix, row_blocks, col_blocks)
        B = A[2:4, 1:7]
        B.values = [2. -2.; 3. -3.]
        BMF.row_add!(A, 2:4, B)
        D_view = view(test_D, 2:4, :)
        (grad_D, grad_B) = gradient((d,b)->sum(d+b), D_view, B)
        
        @test grad_D == ones(3,7)
        @test grad_B == [2. 5.; 4. 10.] # Just the number of entries for each value 

        # Backpropagation for multiplication
        C = ones(3,7)
        (grad_C, grad_B) = gradient((c,b)->sum(c*b), C, B)
        
        @test grad_C == [ 2. 2. -2. -2. -2. -2. -2.; 
                          3. 3. -3. -3. -3. -3. -3.; 
                          3. 3. -3. -3. -3. -3. -3.]
        @test grad_B == [ 2. 5.; 4. 10.] # Just the sum of entries for each value 

    end

end


function col_block_map_tests()

    my_logistic =  x -> 1 ./ (1 .+ exp.(-x))
    my_exp = x -> exp.(x)

    my_sqerr = (x,y) -> 0.5.*(x .- y).^2
    my_binerr = (x,y) -> - y .* log.(x) - (1 .- y) .* log.(1 .- x)

    col_block_ids = ["cat", "cat", "dog", "dog", "dog"]
    
    X = zeros(2,5)
    Y = ones(2,5)
    Y[:,1:2] .= 0.5
    

    @testset "ColBlockMap" begin
    
        # Construction
        my_cbm = BMF.ColBlockMap([my_logistic, my_exp], col_block_ids)
        @test my_cbm.col_blocks == [1:2, 3:5]

        A = my_cbm(X)

        @test A == Y
        
        grad = gradient(x->sum(my_cbm(x)), X)[1]
        correct_grad = ones(2,5)
        correct_grad[:,1:2] .= 0.25

        @test grad == correct_grad


    end

    D = ones(2,5)
    test_losses = zeros(2,5)
    test_losses[:,1:2] .= -log(0.5)
    
    @testset "ColBlockAgg" begin

        # Construction
        my_cba = BMF.ColBlockAgg([my_binerr, my_sqerr], col_block_ids)
        @test my_cba.col_blocks == [1:2, 3:5]

        losses = my_cba(Y, D)
        @test losses == test_losses 

        grad = gradient(y-> sum(my_cba(y,D)), Y)[1]
        correct_grad = zeros(2,5)
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
    theta = randn(n_batches, 2) .* 0.1
    log_delta = randn(n_batches, 2) .* 0.1

    noise_models = [repeat(["logistic"], n_logistic);repeat(["normal"], N-n_logistic)]
    if n_logistic > 0
        noise_map = BMF.ColBlockMap([BMF.logistic_link, BMF.quad_link], noise_models)
    else
        noise_map = BMF.ColBlockMap([BMF.quad_link], noise_models)
    end

    row_batch_ranges = BMF.ids_to_ranges(row_batches)
    col_batch_ranges = BMF.ids_to_ranges(noise_models) 

    A = BMF.forward(X, Y, mu, log_sigma, theta, log_delta,
                   row_batch_ranges, col_batch_ranges,
                   noise_map)

    if n_logistic > 0
        A_logistic = view(A, :, 1:n_logistic)
        A_logistic[A_logistic .>= 0.5] .= 1.0
        A_logistic[A_logistic .< 0.5] .= 0.0
    end

    return X, Y, log_sigma, mu, log_delta, theta, A
end


function generate_regularizers(M, N, K)
    
    X_reg = fill(BMF.BMFRegMat(SparseMatrixCSC(I(M))), K)
    Y_reg = fill(BMF.BMFRegMat(SparseMatrixCSC(I(N))), K)
    
    mu_reg = BMF.BMFRegMat(SparseMatrixCSC(I(N)))
    sigma_reg = BMF.BMFRegMat(SparseMatrixCSC(I(N)))
    
    return X_reg, Y_reg, mu_reg, sigma_reg
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

    log_delta = zeros(BMF.BMFFloat, n_batches, 2)
    theta = zeros(BMF.BMFFloat, n_batches, 2)

    feature_link_map = BMF.ColBlockMap(Function[BMF.logistic_link, BMF.quad_link],
                                       col_batch_ranges)

    @testset "Forward" begin

        # Computation
        A = BMF.forward(X, Y, mu, log_sigma, 
                        theta, log_delta,
                        row_batch_ranges,
                        col_batch_ranges,
                        feature_link_map)
       
        test_A = CUDA.zeros(M,N)
        test_A[:,1:n_logistic] .= 0.5

        @test A == test_A

        # Gradient
        curried_forward_sum = (X, Y, mu, log_sigma,
                               theta, log_delta) -> sum(BMF.forward(X, Y, mu, log_sigma, 
                                                                    theta, log_delta,
                                                                    row_batch_ranges,
                                                                    col_batch_ranges,
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

        test_grad_theta = zeros(n_batches, 2)
        test_grad_theta[:,1] .= 0.25 * n_logistic * div(M,n_batches)
        test_grad_theta[:,2] .= n_logistic * div(M,n_batches)
        @test grad_theta == test_grad_theta
       
        test_grad_sigma = CUDA.zeros(N)
        @test grad_log_sigma == test_grad_sigma

        test_grad_delta = zeros(n_batches,2)
        @test grad_log_delta == test_grad_delta

    end


    feature_loss_map = BMF.ColBlockAgg(Function[BMF.logistic_loss, BMF.quad_loss],
                                       col_batch_ranges)

    D = CUDA.zeros(M,N)
    #D[:,1:n_logistic] .= 0.5

    @testset "Likelihood" begin

        # Computation
        loss = BMF.neg_log_likelihood(X, Y, mu, log_sigma, 
                                      theta, log_delta,
                                      row_batch_ranges,
                                      col_batch_ranges,
                                      feature_link_map, 
                                      feature_loss_map, D)
        @test isapprox(loss, n_logistic * M * log(2))

        # Gradient
        new_D = CUDA.zeros(M,N)
        new_D[:,1:n_logistic] .= 0.5
        curried_loss = (X, Y, mu, log_sigma, 
                        theta, log_delta) -> BMF.neg_log_likelihood(X, Y, mu, log_sigma, 
                                                                    theta, log_delta,
                                                                    row_batch_ranges,
                                                                    col_batch_ranges,
                                                                    feature_link_map, 
                                                                    feature_loss_map, new_D)
        grad_X, grad_Y, grad_mu, grad_log_sigma,
        grad_theta, grad_log_delta = gradient(curried_loss, X, Y, mu, log_sigma,
                                                            theta, log_delta)
      
        # Compare autograd gradients against true (hand-calculated) gradients
        @test grad_X == CUDA.zeros(K,M)
        @test grad_Y == CUDA.zeros(K,N)
        test_grad_mu = CUDA.zeros(N)
        test_grad_mu[1:n_logistic] .= Float32(M * 0.5)
        @test grad_mu == test_grad_mu 
        @test grad_log_sigma == CUDA.zeros(N)
        test_grad_theta = zeros(n_batches,2)
        test_grad_theta[:,1] .= Float32(0.5 * n_logistic * div(M,n_batches))
        @test grad_theta == test_grad_theta
        @test grad_log_delta == zeros(n_batches, 2)

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



function fit_tests()

    #M = 1000
    #N = 2000
    #K = 100
    M = 20
    N = 10
    K = 5
    n_batches = 4

    X_reg, Y_reg, mu_reg, sigma_reg = generate_regularizers(M, N, K)
    
    sample_batch_ids = repeat(1:n_batches, inner=div(M,n_batches))

    n_logistic = div(N,2)
    #n_logistic = 0
    feature_loss_names = [repeat(["logistic"],n_logistic); repeat(["normal"],N-n_logistic)] 
    
    my_model = BatchMatFacModel(X_reg, Y_reg, mu_reg, sigma_reg,
                                sample_batch_ids, feature_loss_names,
                                feature_loss_names)

    test_X, test_Y, 
    test_log_sigma, test_mu, 
    test_log_delta, test_theta, A = simulate_data(M, N, K, sample_batch_ids, n_logistic)

    println("A")
    println(A)

    println("TEST THETA")
    println(test_theta)
    println("TEST_LOG_DELTA")
    println(test_log_delta)

    BMF.fit!(my_model, A; max_epochs=100, capacity=M*N, lr=0.0005)#div(M*N,2))

    println(my_model)
    #function fit!(model::BatchMatFacModel, A::AbstractMatrix;
    #              capacity::Integer=1e8, max_epochs::Integer=1000, 
    #              lr::Real=0.01f0, abs_tol::Real=1e-9, rel_tol::Real=1e-9)


end




function main()
   
    #util_tests()
    #block_matrix_tests()
    #col_block_map_tests()
    #model_core_tests()
    fit_tests()

end

main()


