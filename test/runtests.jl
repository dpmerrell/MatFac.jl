
using BatchMatFac, CUDA, Zygote, HDF5, SparseArrays, LinearAlgebra

println("Hello World")

M = 10000
N = 20000
K = 100

X_reg = fill(CUDA.CUSPARSE.CuSparseMatrixCSC(SparseMatrixCSC{Float32,Int64}(I(M))), K)
Y_reg = fill(CUDA.CUSPARSE.CuSparseMatrixCSC(SparseMatrixCSC{Float32,Int64}(I(N))), K)

mu_reg = CUDA.CUSPARSE.CuSparseMatrixCSC(SparseMatrixCSC{Float32,Int64}(I(N)))
sigma_reg = CUDA.CUSPARSE.CuSparseMatrixCSC(SparseMatrixCSC{Float32,Int64}(I(N)))

sample_batch_ids = repeat(1:10, inner=div(M,10))
feature_batch_ids = repeat(1:10, inner=div(N,10))

feature_loss_names = [repeat(["logistic"],div(N,2)); repeat(["normal"],div(N,2))] 

my_model = BatchMatFacModel(X_reg, Y_reg, mu_reg, sigma_reg,
                            sample_batch_ids, feature_batch_ids,
                            feature_loss_names)

println("SUCCESSFULLY BUILT MODEL")

println("THETA")
println(my_model.theta)
println("DELTA")
println(my_model.delta)

A = forward(my_model)
println("SUCCESSFULLY RAN FORWARD")
println(size(A))

fake_data = CUDA.randn(M,N)

loss = total_loss(my_model, fake_data)
println("SUCCESSFULLY COMPUTED TOTAL LOSS")
println(loss)

grad = gradient(model -> total_loss(model, fake_data), my_model)
println("SUCCESSFULLY COMPUTED GRADIENT")
println(grad)


