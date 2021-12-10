
using BatchMatFac, CUDA, Zygote, HDF5, SparseArrays, LinearAlgebra

println("Hello World")

M = 10000
N = 20000
K = 100

X_reg = fill(SparseMatrixCSC(I(M)), K)
Y_reg = fill(SparseMatrixCSC(I(N)), K)

mu_reg = SparseMatrixCSC(I(M))
sigma_reg = SparseMatrixCSC(I(M))

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
println(A)
