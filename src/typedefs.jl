
BMFFloat = Float32
BMFMat = CuArray{BMFFloat}
BMFRegMat = CUDA.CUSPARSE.CuSparseMatrixCSC{BMFFloat}
BMFVec = CuVector{BMFFloat}
BMFData = AbstractMatrix


