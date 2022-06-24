
using MatFac, Test, Flux, Zygote, CSV, DataFrames

MF = MatFac
    
logistic(x) = 1 / (1 + exp(-x))


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
        ordinal_data = [1 2 3;
                        1 2 3]
        ordinal_Z = [0 0 0;
                     0 0 0]
        on = MF.OrdinalNoise(3, [-Inf, -1, 1, Inf])
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
        opt = Flux.Optimise.ADAGrad()

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
    
    noise_model_tests()
    model_tests()
    update_tests()
    fit_tests()
    io_tests()

end

main()


