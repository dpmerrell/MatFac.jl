
using Test, BatchMatFac, CUDA, Zygote, HDF5, SparseArrays, LinearAlgebra

BMF = BatchMatFac

function util_tests()

    @testset "Utility functions" begin

        @test BMF.is_contiguous([2,2,2,1,1,4,4,4])
        @test BMF.is_contiguous(["cat","cat","dog","dog","fish"])
        @test !BMF.is_contiguous([1,1,5,5,1,3,3,3])

        @test BMF.ids_to_ranges([2,2,2,1,1,4,4,4]) == [1:3, 4:5, 6:8]
        @test BMF.ids_to_ranges(["cat","cat","dog","dog","fish"]) == [1:2,3:4,5:5]

    end

end


function block_matrix_tests()

    @testset "BlockMatrix" begin
        
        row_blocks = ["cat","cat","dog","dog","dog","fish"]
        col_blocks = [1,1,2,2,2,2,2]
        r_matrix = [1. 1.; 2. 2.; 3. 4.]

        # Construction
        A = BMF.block_matrix(r_matrix, row_blocks, col_blocks)
    
        @test A == BMF.BlockMatrix(r_matrix, UnitRange[1:2,3:5,6:6], UnitRange[1:2,3:7])
        @test size(A) == (6,7)

        # Getindex
        B = A[3:6]

        @test B.row_ranges == [1:3, 4:4]
        @test B.col_ranges == A.col_ranges
        
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

        # Additive update (100% overlapping blocks)
        B.values = [1. 1.; 1. 1.]
        BMF.add!(A, 3:6, B)
        @test A.values == [1. 1.; 3. 3.; 4. 5.]

        # Additive update (partially overlapping blocks)
        A = BMF.block_matrix(r_matrix, row_blocks, col_blocks)
        B = A[2:4]
        B.values = [2. -2.; 3. -3.]
        BMF.add!(A, 2:4, B)

        @test A.values == [2. 0.; 4. 0.; 3. 4.]

        # Backpropagation for addition
        D_view = view(test_D, 2:4, :)
        (grad_D, grad_B) = gradient((d,b)->sum(d+b), D_view, B)
        
        @test grad_D == ones(3,7)
        @test grad_B.values == [2. 5.; 4. 10.] # Just the number of entries for each value 

        # Backpropagation for multiplication
        C = ones(3,7)
        (grad_C, grad_B) = gradient((c,b)->sum(c*b), C, B)
        
        @test grad_C == [ 2. 2. -2. -2. -2. -2. -2.; 3. 3. -3. -3. -3. -3. -3.; 3. 3. -3. -3. -3. -3. -3.]
        @test grad_B.values == [ 2. 5.; 4. 10.] # Just the sum of entries for each value 

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


function main()
   
    #util_tests()
    #block_matrix_tests()
    col_block_map_tests()

    #M = 10000
    #N = 20000
    #K = 100
    #
    #X_reg = fill(CUDA.CUSPARSE.CuSparseMatrixCSC(SparseMatrixCSC{Float32,Int64}(I(M))), K)
    #Y_reg = fill(CUDA.CUSPARSE.CuSparseMatrixCSC(SparseMatrixCSC{Float32,Int64}(I(N))), K)
    #
    #mu_reg = CUDA.CUSPARSE.CuSparseMatrixCSC(SparseMatrixCSC{Float32,Int64}(I(N)))
    #sigma_reg = CUDA.CUSPARSE.CuSparseMatrixCSC(SparseMatrixCSC{Float32,Int64}(I(N)))
    #
    #sample_batch_ids = repeat(1:10, inner=div(M,10))
    #feature_batch_ids = repeat(1:10, inner=div(N,10))
    
    #feature_loss_names = [repeat(["logistic"],div(N,2)); repeat(["normal"],div(N,2))] 
    #
    #my_model = BatchMatFacModel(X_reg, Y_reg, mu_reg, sigma_reg,
    #                            sample_batch_ids, feature_batch_ids,
    #                            feature_loss_names)
    
    #println("SUCCESSFULLY BUILT MODEL")
    #
    #println("THETA")
    #println(my_model.theta)
    #println("DELTA")
    #println(my_model.delta)
    #
    #A = forward(my_model)
    #println("SUCCESSFULLY RAN FORWARD")
    #println(size(A))
    
    #fake_data = randn(M,N)
    #
    #fake_X = CUDA.randn(K,M)
    #
    #function arr_summary(arr)
    #    return string(typeof(arr), " ", size(arr))
    #end
    #
    #for minibatch in BatchMatFac.DataLoader([fake_data, transpose(fake_X)])
    #    #data_batch = CUDA.CuArray(minibatch[1])
    #    data_batch = minibatch[1]
    #    X_batch = minibatch[2]
    #    println(arr_summary(data_batch), "\t", arr_summary(X_batch))
    #end
    
    #loss = total_loss(my_model, fake_data)
    #println("SUCCESSFULLY COMPUTED TOTAL LOSS")
    #println(loss)
    #
    #grad = gradient(model -> total_loss(model, fake_data), my_model)
    #println("SUCCESSFULLY COMPUTED GRADIENT")
    #println(grad)

end

main()
