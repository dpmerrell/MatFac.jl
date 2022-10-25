
using MatFac, Test, Flux, Zygote, CSV, DataFrames, Functors, Statistics, StatsBase

MF = MatFac
    
logistic(x) = 1 / (1 + exp(-x))

mutable struct TestStruct
    a::Float64
    b::NamedTuple
    c::String
end
@functor TestStruct
Flux.trainable(ts::TestStruct) = (a=ts.a, b=ts.b)

function util_tests()

    @testset "Utility functions" begin

        opt = Flux.Optimise.AdaGrad()
        opt_copy = deepcopy(opt)
        params = (W=randn(10,20),
                  child=(a=randn(10),
                         b=randn(10))
                  )
        params_copy = deepcopy(params)
        grads = (W=ones(10,20),
                 child=(a=ones(10),
                        b=nothing)
                )

        params_tuple = (randn(10,20), "cat", randn(10))
        params_tuple_copy = deepcopy(params_tuple)
        grads_tuple = (nothing, nothing, ones(10))

       
        ####################################### 
        # Flux.update! extensions
        Flux.update!(opt, params, nothing) 
        @test all(params.W .== params_copy.W)
        @test all(params.child.a .== params_copy.child.a)
        @test all(params.child.b .== params_copy.child.b)
        @test opt.eta == opt_copy.eta

        Flux.update!(opt, params, grads)
        @test isapprox(params.W, params_copy.W .- (opt.eta ./ (sqrt.(grads.W.*grads.W) .+ opt.epsilon) ) .* grads.W ) 
        @test isapprox(params.child.b, params_copy.child.b)

        Flux.update!(opt, params_tuple, grads_tuple)
        @test isapprox(params_tuple[3], params_tuple_copy[3] .- (opt.eta ./ (sqrt.(grads_tuple[3].*grads_tuple[3]) .+ opt.epsilon) ) .* grads_tuple[3] )
        @test params_tuple[2] == "cat"

        #######################################
        # Functors: fmap, fmapstructure extensions
        test_struct = (A=randn(10), child=())
        result = fmapstructure(x -> 2.0 .* x, test_struct)
        @test isapprox(result.A, test_struct.A .* 2.0)
        @test result.child == test_struct.child
        result = fmap(x -> 2.0 .* x, test_struct)
        @test isapprox(result.A, test_struct.A .* 2.0)
        @test result.child == test_struct.child

        #######################################
        # binop!
        params = deepcopy(params_copy)
        MF.binop!(.+, params, grads)
        @test isapprox(params.W, params_copy.W .+ grads.W)
        @test isapprox(params.child.a, params_copy.child.a .+ grads.child.a)
        @test isapprox(params.child.b, params_copy.child.b)

        #######################################
        # rec_trainable
        ts = TestStruct(3.14, (W=randn(10,20), name="dog"), "fish")

        trn_params = MF.rec_trainable(ts)
        @test trn_params.a == ts.a
        @test trn_params.b.W == ts.b.W
        @test trn_params.b.name == ()

        ##################################
        # tozero
        result = fmapstructure(MF.tozero, params)
        @test isapprox(result.W, zero(result.W))
        @test isapprox(result.child.a, zero(result.child.a))
        @test isapprox(result.child.b, zero(result.child.b))

        ##########################################
        # replace_if, tozero!, tonan!, toone!
        test_array = rand([0,NaN], 10, 20)
        test_array_copy = copy(test_array)
        nan_idx = isnan.(test_array)
        MF.tozero!(test_array, nan_idx)
        @test isapprox(test_array, zero(test_array)) 
        MF.tonan!(test_array, nan_idx)
        @test all(map((a,b) -> isequal(a,b), test_array, test_array_copy))
        MF.toone!(test_array, nan_idx)
        @test sum(test_array) == sum(nan_idx)
        MF.tonan!(test_array, nan_idx)

        #########################################
        # batched_reduce
        s = MF.batched_reduce((s, D) -> s + sum(isnan.(D)), test_array; capacity=20)
        @test s == sum(isnan.(test_array))

        col_counts = MF.batched_reduce((r, D) -> r .+ sum(isnan.(D), dims=1), test_array; capacity=20)
        @test isapprox(col_counts, sum(isnan.(test_array), dims=1))

        ########################################## 
        # column_nonnan
        @test isapprox(MF.column_nonnan(test_array), vec(sum((!isnan).(test_array), dims=1)))

        #########################################
        # batched_column_meanvar
        A = randn(10,20)
        batched_means, batched_vars = MF.batched_column_meanvar(A; capacity=20)
        @test isapprox(batched_means, vec(mean(A; dims=1)))
        @test isapprox(batched_vars, vec(var(A; dims=1, corrected=false)))

        #########################################
        # batched_column_mean_loss
        noise_model = MF.NormalNoise(20)
        col_losses = MF.batched_column_mean_loss(noise_model, A)
        @test isapprox(col_losses, vec(0.5 .* var(A; dims=1, corrected=false)))

        #########################################
        # batched_data_loss
        matfac = MF.MatFacModel(10, 20, 1, "normal")
        matfac.X .= 0
        matfac.Y .= 0
        total_loss = MF.batched_data_loss(matfac, A; capacity=20)
        @test isapprox(total_loss, 0.5*sum(A .* A))

    end

end


function noise_model_tests()


    @testset "Noise models" begin
        M = 20
        N = 40

        # Normal noise model
        normal_data = ones(M,N)
        normal_Z = zeros(M,N)
        nn = MF.NormalNoise(N)
        @test MF.invlink(nn, normal_Z) == normal_Z
        @test MF.loss(nn, MF.invlink(nn, normal_Z), normal_data) == 0.5.*ones(M,N)
        @test Flux.gradient(x->sum(MF.loss(nn, MF.invlink(nn, x),normal_data)), normal_Z)[1] == -ones(M,N)
        @test Flux.gradient(x->MF.invlinkloss(nn, x,normal_data), normal_Z)[1] == -ones(M,N)


        # Logistic noise model
        logistic_data = ones(M,N)
        logistic_Z = zeros(M,N)
        ln = MF.BernoulliNoise(N)
        @test MF.invlink(ln, logistic_Z) == 0.5 .* ones(M,N)
        @test Flux.gradient(x->sum(MF.invlink(ln, x)), logistic_Z)[1] == MF.invlink(ln,logistic_Z).*(1 .- MF.invlink(ln, logistic_Z))
        @test isapprox(MF.loss(ln, MF.invlink(ln, logistic_Z), logistic_data), log(2.0).*ones(M,N), atol=1e-6)
        logistic_A = MF.invlink(ln, logistic_Z)
        @test Flux.gradient(x->sum(MF.loss(ln, x, logistic_data)), logistic_A)[1] == -2.0.*ones(M,N)
        @test Flux.gradient(x->MF.invlinkloss(ln, x, logistic_data), logistic_Z)[1] == (logistic_A .- logistic_data)


        # Poisson noise model
        poisson_data = ones(M,N)
        poisson_Z = zeros(M,N)
        pn = MF.PoissonNoise(N)
        @test MF.invlink(pn, poisson_Z) == ones(M,N)
        @test isapprox(MF.loss(pn, MF.invlink(pn, poisson_Z), poisson_data), ones(M,N), atol=1e-6)
        @test Flux.gradient(x->sum(MF.loss(pn, MF.invlink(pn, x), logistic_data)), poisson_Z)[1] == zeros(M,N)
        @test Flux.gradient(x->MF.invlinkloss(pn, x, logistic_data), poisson_Z)[1] == zeros(M,N)

        # Ordinal noise model
        ordinal_data = [1. 2. 3.;
                        1. 2. 3.]
        ordinal_Z = [0. 0. 0.;
                     0. 0. 0.]
        on = MF.OrdinalNoise(3, [-Inf, -1., 1., Inf])
        @test MF.invlink(on, ordinal_Z) == ordinal_Z 
        @test isapprox(MF.loss(on, ordinal_Z, ordinal_data), [-log.(logistic(-1)-logistic(-Inf)) -log.(logistic(1)-logistic(-1)) -log.(logistic(Inf)-logistic(1));
                                                               -log.(logistic(-1)-logistic(-Inf)) -log.(logistic(1)-logistic(-1)) -log.(logistic(Inf)-logistic(1))],
                       atol=1e-6)
        thresh_grad, z_grad = Flux.gradient((noise, x)->sum(MF.loss(noise, MF.invlink(noise, x), ordinal_data)), on, ordinal_Z)
        
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

        thresh_grad, z_grad = Flux.gradient((noise,x)->MF.invlinkloss(noise, x, ordinal_data), on, ordinal_Z)
        @test thresh_grad.ext_thresholds == test_thresh_grad
        @test z_grad == test_z_grad


        # Composite noise models
        composite_Z = zeros(M,N)
        composite_data = zeros(M,N)
        composite_data[:,21:30] .= 1
        composite_data[:,31:40] .= 2

        noise_model = repeat(["normal","bernoulli","poisson", "ordinal3"], inner=10)
        cn = MF.CompositeNoise(noise_model)

        composite_A = MF.invlink(cn, composite_Z)
        @test composite_A[:,1:10] == MF.invlink(nn, composite_Z[:,1:10])
        @test composite_A[:,11:20]== MF.invlink(ln, composite_Z[:,11:20])
        @test composite_A[:,21:30]== MF.invlink(pn, composite_Z[:,21:30])
        @test composite_A[:,31:40]== MF.invlink(cn.noises[4], composite_Z[:,31:40])

        composite_l = MF.loss(cn, composite_A, composite_data)
        @test composite_l[:,1:10] == MF.loss(cn.noises[1], composite_A[:,1:10] , composite_data[:,1:10])
        @test composite_l[:,11:20]== MF.loss(cn.noises[2], composite_A[:,11:20], composite_data[:,11:20])
        @test composite_l[:,21:30]== MF.loss(cn.noises[3], composite_A[:,21:30], composite_data[:,21:30])
        @test composite_l[:,31:40]== MF.loss(cn.noises[4], composite_A[:,31:40], composite_data[:,31:40])

        grad_cn, grad_Z = Flux.gradient((noise,x)->MF.invlinkloss(noise, x, composite_data), 
                                        cn, composite_Z)
        test_grad_Z = zeros(M,N)
        test_grad_Z[:,11:20] .= 0.5
        test_grad_ordinal = [0.0, 200.0*logistic(-0.5)*(1 - logistic(-0.5))/(logistic(0.5) - logistic(-0.5)),
                             -200.0*logistic(0.5)*(1 - logistic(0.5))/(logistic(0.5) - logistic(-0.5)) ,0.0]
        @test isapprox(grad_Z, test_grad_Z)
        @test isapprox(grad_cn.noises[4].ext_thresholds, test_grad_ordinal) 
                                                         

        # GPU tests
        cn_d = gpu(cn)
        composite_Z_d = gpu(composite_Z)
        composite_data_d = gpu(composite_data)
        test_grad_Z_d = gpu(test_grad_Z)
        test_grad_ordinal_d = gpu(test_grad_ordinal)
        grad_cn_d, grad_Z_d = Flux.gradient((noise,x)->MF.invlinkloss(noise, x, composite_data_d), 
                                            cn_d, composite_Z_d)
        @test isapprox(grad_Z_d, test_grad_Z_d)
        @test isapprox(grad_cn_d.noises[4].ext_thresholds, test_grad_ordinal_d)
    end

end



function model_tests()

    @testset "Model" begin
        
        M = 20
        N = 40
        K = 5
        n_loss_types = 4
        #n_row_batches = 4
        n_logistic = div(N,4)
        n_ordinal = div(N,4)
        n_poisson = div(N,4)
        n_normal = N - (n_logistic + n_ordinal + n_poisson)

        col_losses = [repeat(["bernoulli"], n_logistic);
                      repeat(["ordinal5"], n_ordinal);
                      repeat(["poisson"], n_poisson);
                      repeat(["normal"], n_normal)]

        model = MF.MatFacModel(M,N,K,col_losses)

        @test size(model.X) == (K,M)
        @test map(typeof, model.noise_model.noises) == (MF.BernoulliNoise, MF.OrdinalNoise, MF.PoissonNoise, MF.NormalNoise)
        @test size(model()) == (M,N)
        @test size(model) == (M,N)

        # GPU
        model_d = gpu(model)
        @test isapprox(transpose(model_d.X)*model_d.Y, gpu(transpose(model.X)*model.Y))
        @test isapprox(model_d(), gpu(model()))
    end
end


function update_tests()

    @testset "Update" begin

        # Array update
        A = ones(10,10)
        start_A = deepcopy(A)

        f = x -> sum(x.^2)
        grads = Zygote.gradient(f, A)
        opt = Flux.Optimise.AdaGrad()

        Flux.Optimise.update!(opt, A, grads[1])
        @test !isapprox(A, start_A)

        # Tuple update
        B = (ones(10,10), ones(5,5))
        start_B = deepcopy(B)
        B_grad = (-ones(10,10), -ones(5,5))

        Flux.Optimise.update!(opt, B, B_grad)
        @test !isapprox(B[1], start_B[1])
        @test !isapprox(B[2], start_B[2])
        
        # NamedTuple update
        C = (cat=ones(10,10), dog=ones(5,5))
        start_C = deepcopy(C)
        C_grad = (cat=-ones(10,10), dog=-ones(5,5), fish=nothing)

        Flux.Optimise.update!(opt, C, C_grad)
        @test !isapprox(C[1], start_C[1])
        @test !isapprox(C[2], start_C[2])
        
    end
end

function fit_tests()

    @testset "Fit" begin
        
        M = 20
        N = 40
        K = 3
        n_loss_types = 4
        n_logistic = div(N,n_loss_types)
        n_ordinal = div(N,n_loss_types)
        n_poisson = div(N,n_loss_types)
        n_normal = div(N,n_loss_types) 

        col_losses = [repeat(["bernoulli"], n_logistic);
                      repeat(["normal"], n_normal);
                      repeat(["poisson"], n_poisson);
                      repeat(["ordinal5"], n_ordinal)];
        

        composite_data = zeros(M,N)
        composite_data[:,21:30] .= 1
        composite_data[:,31:40] .= 3

        to_null = rand(Bool, M, N)
        composite_data[to_null] .= NaN

        #################################
        # CPU TESTS
        model = MF.MatFacModel(M,N,K, col_losses)
        X_start = deepcopy(model.X)
        Y_start = deepcopy(model.Y)
        thresholds_start = deepcopy(model.noise_model.noises[4].ext_thresholds)

        # test whether the fit! function can run to
        # completion (under max_epochs condition)
        fit!(model, composite_data; verbosity=1, lr=0.05, max_epochs=10)
        @test true

        # test whether the parameters were modified
        @test !isapprox(model.X, X_start)
        @test !isapprox(model.Y, Y_start)
        @test !isapprox(model.noise_model.noises[4].ext_thresholds, thresholds_start)
        
        # test whether the parameters contain NaN
        @test all(isfinite.(model.X))
        @test all(isfinite.(model.Y))
        @test all(isfinite.(model.noise_model.noises[4].ext_thresholds[2:4]))
 
        #################################
        # GPU TESTS
        model = MF.MatFacModel(M,N,K, col_losses)
        model = gpu(model)
        composite_data = gpu(composite_data)
        X_start = deepcopy(model.X)
        Y_start = deepcopy(model.Y)
        thresholds_start = deepcopy(model.noise_model.noises[4].ext_thresholds)

        # test whether the fit! function can run to
        # completion (under max_epochs condition)
        fit!(model, composite_data; verbosity=1, lr=0.05, max_epochs=10)
        @test true

        # test whether the parameters were modified
        @test !isapprox(cpu(model.X), cpu(X_start))
        @test !isapprox(cpu(model.Y), cpu(Y_start))
        @test !isapprox(cpu(model.noise_model.noises[4].ext_thresholds), cpu(thresholds_start))
        
    end
end

function callback_tests()

    @testset "Callbacks" begin

        M = 20
        N = 40
        K = 3
        n_loss_types = 4
        n_logistic = div(N,n_loss_types)
        n_ordinal = div(N,n_loss_types)
        n_poisson = div(N,n_loss_types)
        n_normal = div(N,n_loss_types) 

        col_losses = [repeat(["bernoulli"], n_logistic);
                      repeat(["normal"], n_normal);
                      repeat(["poisson"], n_poisson);
                      repeat(["ordinal5"], n_ordinal)];
        

        composite_data = zeros(M,N)
        composite_data[:,21:30] .= 1
        composite_data[:,31:40] .= 3

        #################################
        # CPU TESTS
        model = MF.MatFacModel(M,N,K, col_losses)
        X_start = deepcopy(model.X)
        Y_start = deepcopy(model.Y)
        thresholds_start = deepcopy(model.noise_model.noises[4].ext_thresholds)

        # Construct a HistoryCallback 
        hcb = MF.HistoryCallback()

        # test whether the HistoryCallback records history correctly
        fit!(model, composite_data; verbosity=1, lr=0.05, max_epochs=10, callback=hcb)
        @test length(hcb.history) == 10


    end

end

function io_tests()

    @testset "IO tests" begin

        test_bson_path = "model_test.bson"

        M = 20
        N = 40
        K = 3
        n_loss_types = 4
        n_logistic = div(N,n_loss_types)
        n_ordinal = div(N,n_loss_types)
        n_poisson = div(N,n_loss_types)
        n_normal = div(N,n_loss_types) 

        col_losses = [repeat(["bernoulli"], n_logistic);
                      repeat(["normal"], n_normal);
                      repeat(["poisson"], n_poisson);
                      repeat(["ordinal5"], n_ordinal)];
        model = MF.MatFacModel(M,N,K, col_losses)
       
        MF.save_model(test_bson_path, model)

        new_model = MF.load_model(test_bson_path)

        @test new_model.X == model.X
        @test new_model.Y == model.Y
        @test new_model.noise_model.noises[4].ext_thresholds == model.noise_model.noises[4].ext_thresholds

        rm(test_bson_path)
    end
end


function main()
   
    util_tests() 
    noise_model_tests()
    model_tests()
    update_tests()
    fit_tests()
    callback_tests()
    io_tests()

end

main()


