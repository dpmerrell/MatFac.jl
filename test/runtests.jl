
using BatchMatFac, Test, Zygote

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

        my_ind_mat = BMF.ids_to_ind_mat([1,1,1,2,2,1,2,3,3,1,2,3,3])
        @test my_ind_mat == Bool[1 0 0;
                                 1 0 0;
                                 1 0 0;
                                 0 1 0;
                                 0 1 0;
                                 1 0 0;
                                 0 1 0;
                                 0 0 1;
                                 0 0 1;
                                 1 0 0;
                                 0 1 0;
                                 0 0 1;
                                 0 0 1] 

    end

end

function model_core_tests()

    @testset "Model Core" begin

        M = 20
        N = 30
        K = 4

        view_N = 10

        ###################################
        # Matrix Product layer
        mp = BMF.MatProd(M,N,K)
        @test size(mp.X) == (K,M)
        @test size(mp.Y) == (K,N)
        xy = mp()
        @test xy == transpose(mp.X)*mp.Y

        mp_view = view(mp, :, 1:view_N)
        @test size(mp_view.X) == (K,M)
        @test size(mp_view.Y) == (K,10)
        view_xy = mp_view()
        @test view_xy == transpose(mp.X)*mp.Y[:,1:10]
        
        ###################################
        # Column Scale
        cscale = BMF.ColScale(N)
        @test size(cscale.logsigma) == (N,)
        @test cscale(xy) == xy .* transpose(exp.(cscale.logsigma))
        
        cscale_view = view(cscale,1:view_N)
        @test cscale_view(view_xy) == view_xy .* transpose(exp.(cscale.logsigma[1:view_N]))

        ###################################
        # Column Shift
        cshift = BMF.ColShift(N)
        @test size(cshift.mu) == (N,)
        @test cshift(xy) == xy .+ transpose(cshift.mu)
        
        cshift_view = view(cshift,1:view_N)
        @test cshift_view(view_xy) == view_xy .+ transpose(cshift.mu[1:view_N])

    end
end


function batch_array_tests()
    
    @testset "Batch Arrays" begin

        col_batches = ["cat", "cat", "cat", "dog", "dog", "fish"]
        row_batches = [[1,1,1,2,2], [1,1,2,2,2], [1,1,1,1,2]]
        values = [Dict(1=>3.14, 2=>2.7), Dict(1=>0.0, 2=>0.5), Dict(1=>-1.0, 2=>1.0)]
        A = zeros(5,6)

        ##############################
        # Constructor
        ba = BMF.BatchArray(col_batches, row_batches, values)
        @test ba.col_ranges == [1:3, 4:5, 6:6]
        test_row_batches = [Bool[1 0; 1 0; 1 0; 0 1; 0 1],
                            Bool[1 0; 1 0; 0 1; 0 1; 0 1],
                            Bool[1 0; 1 0; 1 0; 1 0; 0 1]]
        @test ba.row_batches == test_row_batches 
        @test ba.values == [[3.14, 2.7], [0.0, 0.5], [-1.0, 1.0]]

        ##############################
        # View
        ba_view = view(ba, 2:4, 2:5)
        @test ba_view.col_ranges == [1:2, 3:4]
        @test ba_view.row_batches == [test_row_batches[1][2:4,:],
                                      test_row_batches[2][2:4,:]]
        @test ba_view.values == [[3.14, 2.7],[0.0,0.5]]

        ###############################
        # zero
        ba_zero = zero(ba)
        @test ba_zero.col_ranges == ba.col_ranges
        @test ba_zero.row_batches == ba.row_batches
        @test ba_zero.values == [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]

        ###############################
        # Addition
        Z = A + ba
        test_mat = [3.14 3.14 3.14 0.0 0.0 -1.0;
                    3.14 3.14 3.14 0.0 0.0 -1.0;
                    3.14 3.14 3.14 0.5 0.5 -1.0;
                    2.7  2.7  2.7  0.5 0.5 -1.0;
                    2.7  2.7  2.7  0.5 0.5  1.0]
        @test Z == test_mat
        (A_grad, ba_grad) = gradient((x,y)->sum(x+y), A, ba)
        @test A_grad == ones(size(A)...)
        @test ba_grad.values == [[9., 6.], [4., 6.], [4., 1.]]

        ################################
        # Multiplication
        Z = ones(5,6) * ba
        @test Z == test_mat
        (A_grad, ba_grad) = gradient((x,y)->sum(x*y), ones(5,6), ba)
        @test A_grad == Z
        @test ba_grad.values == [[9.0, 6.0],[4.0, 6.0],[4.0, 1.0]]

        ################################
        # Exponentiation
        ba_exp = exp(ba)
        @test (ones(5,6) * ba_exp) == exp.(test_mat)
        (ba_grad,) = gradient(x->sum(ones(5,6) * exp(x)), ba)
        @test ba_grad.values == [[9. * exp(3.14), 6. * exp(2.7)],
                                 [4. * exp(0.0), 6. * exp(0.5)],
                                 [4. * exp(-1.0), 1. * exp(1.0)]]

    end
end


function main()
    util_tests()
    model_core_tests()
    batch_array_tests()
end


main()

