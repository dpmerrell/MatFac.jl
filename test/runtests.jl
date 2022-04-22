
using BatchMatFac, Test, Flux, Zygote

BMF = BatchMatFac
    
logistic(x) = 1 / (1 + exp(-x))

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

        n_col_batches = 2
        n_row_batches = 4

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

        ##################################
        # Batch Scale
        col_batches = repeat([string("colbatch",i) for i=1:n_col_batches], inner=div(N,n_col_batches))
        row_batches = [repeat([string("rowbatch",i) for i=1:n_row_batches], inner=div(M,n_row_batches)) for j=1:n_col_batches]
        bscale = BMF.BatchScale(col_batches, row_batches)
        @test size(bscale.logdelta.values) == (n_col_batches,)
        @test size(bscale.logdelta.values[1]) == (n_row_batches,)
        @test bscale(xy)[1:div(M,n_row_batches),1:div(N,n_col_batches)] == xy[1:div(M,n_row_batches),1:div(N,n_col_batches)] .* exp(bscale.logdelta.values[1][1])

        bscale_grads = Zygote.gradient((f,x)->sum(f(x)), bscale, xy)
        @test bscale_grads[1].logdelta.values[end][end] == sum(xy[16:20,16:30].*exp(bscale.logdelta.values[end][end]))
        @test bscale_grads[2] == zeros(M,N) + exp(bscale.logdelta)

        ##################################
        # Batch Shift
        bshift = BMF.BatchShift(col_batches, row_batches)
        @test size(bshift.theta.values) == (n_col_batches,)
        @test size(bshift.theta.values[1]) == (n_row_batches,)
        @test bshift(xy)[1:div(M,n_row_batches),1:div(N,n_col_batches)] == xy[1:div(M,n_row_batches),1:div(N,n_col_batches)] .+ bshift.theta.values[1][1]

        bshift_grads = Zygote.gradient((f,x)->sum(f(x)), bshift, xy)
        @test bshift_grads[1].theta.values == [ones(n_row_batches).*(div(M,n_row_batches)*div(N,n_col_batches)) for j=1:n_col_batches]
        @test bshift_grads[2] == ones(M,N)

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
        (A_grad, ba_grad) = Zygote.gradient((x,y)->sum(x+y), A, ba)
        @test A_grad == ones(size(A)...)
        @test ba_grad.values == [[9., 6.], [4., 6.], [4., 1.]]

        ################################
        # Multiplication
        Z = ones(5,6) * ba
        @test Z == test_mat
        (A_grad, ba_grad) = Zygote.gradient((x,y)->sum(x*y), ones(5,6), ba)
        @test A_grad == Z
        @test ba_grad.values == [[9.0, 6.0],[4.0, 6.0],[4.0, 1.0]]

        ################################
        # Exponentiation
        ba_exp = exp(ba)
        @test (ones(5,6) * ba_exp) == exp.(test_mat)
        (ba_grad,) = Zygote.gradient(x->sum(ones(5,6) * exp(x)), ba)
        @test ba_grad.values == [[9. * exp(3.14), 6. * exp(2.7)],
                                 [4. * exp(0.0), 6. * exp(0.5)],
                                 [4. * exp(-1.0), 1. * exp(1.0)]]

    end
end


function noise_model_tests()


    @testset "Noise models" begin
        M = 20
        N = 40

        # Normal noise model
        normal_data = ones(M,N)
        normal_Z = zeros(M,N)
        nn = BMF.NormalNoise()
        @test BMF.invlink(nn, normal_Z) == normal_Z
        @test BMF.loss(nn, BMF.invlink(nn, normal_Z), normal_data) == 0.5.*ones(M,N)
        @test Flux.gradient(x->sum(BMF.loss(nn, BMF.invlink(nn, x),normal_data)), normal_Z)[1] == -ones(M,N)
        @test Flux.gradient(x->BMF.invlinkloss(nn, x,normal_data), normal_Z)[1] == -ones(M,N)


        # Logistic noise model
        logistic_data = ones(M,N)
        logistic_Z = zeros(M,N)
        ln = BMF.BernoulliNoise()
        @test BMF.invlink(ln, logistic_Z) == 0.5 .* ones(M,N)
        @test Flux.gradient(x->sum(BMF.invlink(ln, x)), logistic_Z)[1] == BMF.invlink(ln,logistic_Z).*(1 .- BMF.invlink(ln, logistic_Z))
        @test BMF.loss(ln, BMF.invlink(ln, logistic_Z), logistic_data) == log(2.0).*ones(M,N)
        logistic_A = BMF.invlink(ln, logistic_Z)
        @test Flux.gradient(x->sum(BMF.loss(ln, x, logistic_data)), logistic_A)[1] == -2.0.*ones(M,N)
        @test Flux.gradient(x->BMF.invlinkloss(ln, x, logistic_data), logistic_Z)[1] == (logistic_A .- logistic_data)


        # Poisson noise model
        poisson_data = ones(M,N)
        poisson_Z = zeros(M,N)
        pn = BMF.PoissonNoise()
        @test BMF.invlink(pn, poisson_Z) == ones(M,N)
        @test BMF.loss(pn, BMF.invlink(pn, poisson_Z), poisson_data) == ones(M,N)
        @test Flux.gradient(x->sum(BMF.loss(pn, BMF.invlink(pn, x), logistic_data)), poisson_Z)[1] == zeros(M,N)
        @test Flux.gradient(x->BMF.invlinkloss(pn, x, logistic_data), poisson_Z)[1] == zeros(M,N)

        # Ordinal noise model
        ordinal_data = [1 2 3;
                        1 2 3]
        ordinal_Z = [0 0 0;
                     0 0 0]
        on = BMF.OrdinalNoise([-Inf, -1, 1, Inf])
        @test BMF.invlink(on, ordinal_Z) == ordinal_Z 
        @test BMF.loss(on, ordinal_Z, ordinal_data) == [-log.(logistic(-1)-logistic(-Inf)) -log.(logistic(1)-logistic(-1)) -log.(logistic(Inf)-logistic(1));
                                              -log.(logistic(-1)-logistic(-Inf)) -log.(logistic(1)-logistic(-1)) -log.(logistic(Inf)-logistic(1))]
        thresh_grad, z_grad = Flux.gradient((noise, x)->sum(BMF.loss(noise, BMF.invlink(noise, x), ordinal_data)), on, ordinal_Z)
        
        lgrad(lt,rt,z) = logistic(lt-z)*(1 - logistic(lt-z))/(logistic(rt-z)-logistic(lt-z))
        rgrad(lt,rt,z) = -logistic(rt-z)*(1 - logistic(rt-z))/(logistic(rt-z)-logistic(lt-z))
        test_thresh_grad = [2*lgrad(-Inf,-1,0), 
                              2*(rgrad(-Inf,-1,0)+lgrad(-1,1,0)), 
                              2*(rgrad(-1,1,0)+lgrad(1,Inf,0)), 
                              2*rgrad(1,Inf,0)]
        test_z_grad = [1-logistic(-1)-logistic(-Inf) 1-logistic(1)-logistic(-1) 1-logistic(Inf)-logistic(1);
                         1-logistic(-1)-logistic(-Inf) 1-logistic(1)-logistic(-1) 1-logistic(Inf)-logistic(1)]

        @test thresh_grad.ext_thresholds == test_thresh_grad 
        @test z_grad == test_z_grad 

        thresh_grad, z_grad = Flux.gradient((noise,x)->BMF.invlinkloss(noise, x, ordinal_data), on, ordinal_Z)
        @test thresh_grad.ext_thresholds == test_thresh_grad
        @test z_grad == test_z_grad


        # Composite noise models
        composite_Z = zeros(M,N)
        composite_data = zeros(M,N)
        composite_data[:,21:30] .= 1
        composite_data[:,31:40] .= 2

        noise_models = repeat(["normal","bernoulli","poisson", "ordinal3"], inner=10)
        cn = BMF.CompositeNoise(noise_models)

        composite_A = BMF.invlink(cn, composite_Z)
        @test composite_A[:,1:10] == BMF.invlink(nn, composite_Z[:,1:10])
        @test composite_A[:,11:20]== BMF.invlink(ln, composite_Z[:,11:20])
        @test composite_A[:,21:30]== BMF.invlink(pn, composite_Z[:,21:30])
        @test composite_A[:,31:40]== BMF.invlink(cn.noises[4], composite_Z[:,31:40])

        composite_l = BMF.loss(cn, composite_A, composite_data)
        @test composite_l[:,1:10] == BMF.loss(nn, composite_A[:,1:10] , composite_data[:,1:10])
        @test composite_l[:,11:20]== BMF.loss(ln, composite_A[:,11:20], composite_data[:,11:20])
        @test composite_l[:,21:30]== BMF.loss(pn, composite_A[:,21:30], composite_data[:,21:30])
        @test composite_l[:,31:40]== BMF.loss(cn.noises[4], composite_A[:,31:40], composite_data[:,31:40])

        grad_cn, grad_Z = Flux.gradient((noise,x)->BMF.invlinkloss(noise, x, composite_data), 
                                        cn, composite_Z)
        test_grad_Z = zeros(M,N)
        test_grad_Z[:,11:20] .= 0.5
        @test grad_Z == grad_Z
        @test isapprox(grad_cn.noises[4].ext_thresholds, [0.0, 200.0*logistic(-0.5)*(1 - logistic(-0.5))/(logistic(0.5) - logistic(-0.5)), 
                                                          -200.0*logistic(0.5)*(1 - logistic(0.5))/(logistic(0.5) - logistic(-0.5)) ,0.0])
        #println("GRAD CN")
        #println(grad_cn)
        #println("GRAD Z")
        #println(grad_Z)
    end

end



function model_tests()

    @testset "Model" begin
        
        M = 20
        N = 40
        K = 5
        n_col_batches = 4
        n_row_batches = 4
        n_logistic = div(N,4)
        n_ordinal = div(N,4)
        n_poisson = div(N,4)
        n_normal = N - (n_logistic + n_ordinal + n_poisson)

        col_batches = repeat([string("colbatch",i) for i=1:n_col_batches], inner=div(N,n_col_batches))
        row_batches = [repeat([string("rowbatch",i) for i=1:n_row_batches], inner=div(M,n_row_batches)) for j=1:n_col_batches]
        col_losses = [repeat(["bernoulli"], n_logistic);
                      repeat(["ordinal5"], n_ordinal);
                      repeat(["poisson"], n_poisson);
                      repeat(["normal"], n_normal)]

        model = BMF.BatchMatFacModel(M,N,K, col_batches, row_batches, col_losses)

        @test size(model.mp.X) == (K,M)
        @test map(typeof, model.noise_models.noises) == (BMF.BernoulliNoise, BMF.OrdinalNoise, BMF.PoissonNoise, BMF.NormalNoise)
        @test size(model()) == (M,N)
        @test size(model) == (M,N)

    end
end


function fit_tests()

    @testset "Fit" begin
        
        M = 20
        N = 40
        K = 5
        n_col_batches = 4
        n_row_batches = 4
        n_logistic = div(N,4)
        n_ordinal = div(N,4)
        n_poisson = div(N,4)
        n_normal = N - (n_logistic + n_ordinal + n_poisson)

        col_batches = repeat([string("colbatch",i) for i=1:n_col_batches], inner=div(N,n_col_batches))
        row_batches = [repeat([string("rowbatch",i) for i=1:n_row_batches], inner=div(M,n_row_batches)) for j=1:n_col_batches]
        col_losses = [repeat(["bernoulli"], n_logistic);
                      repeat(["ordinal5"], n_ordinal);
                      repeat(["poisson"], n_poisson);
                      repeat(["normal"], n_normal)]
        model = BMF.BatchMatFacModel(M,N,K, col_batches, row_batches, col_losses)


    end

end



function main()
    
    #util_tests()
    #model_core_tests()
    #batch_array_tests()
    noise_model_tests()
    model_tests()

end


main()

