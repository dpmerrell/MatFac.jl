

export save_hdf, load_hdf

import Base: write


####################################
# MatFacModel
####################################

function write(hdf_file::Union{HDF5.File,HDF5.Group}, path::String, model::BMFModel)
    for prop in propertynames(model)
        write(hdf_file, string(path, "/", prop), getproperty(model, prop))
    end
end


function readtype(hdf_file::Union{HDF5.File,HDF5.Group}, path::String, ::Type{BMFModel})
    
end


function matfac_from_hdf(hdf_file, path::String)
    # Factors
    X = hdf_file[string(path,"/X")][:,:]
    Y = hdf_file[string(path,"/Y")][:,:]

    # Covariate coefficients
    if "instance_covariate_coeff" in keys(hdf_file[string(path,"/")])
        covariate_coeff = hdf_file[string(path,"/instance_covariate_coeff")][:,:]
    end
    covariate_coeff_reg = spmat_from_hdf(hdf_file, string(path,"/instance_covariate_coeff_reg"))

    # Offsets
    instance_offset = hdf_file[string(path,"/instance_offset")][:]
    feature_offset = hdf_file[string(path,"/feature_offset")][:]

    # Offset Regularizers
    instance_offset_reg = spmat_from_hdf(hdf_file, string(path, "/instance_offset_reg"))
    feature_offset_reg = spmat_from_hdf(hdf_file, string(path, "/feature_offset_reg"))

    # Precisions
    feature_precision = hdf_file[string(path,"/feature_precision")][:]

    # Losses
    loss_names = hdf_file[string(path,"/loss_types")][:] 
    loss_scales = hdf_file[string(path,"/loss_scales")][:] 
    losses = Loss[eval(Meta.parse(lname))(lscale) for (lname, lscale) in zip(loss_names, loss_scales)]

    # Instance regularizers
    inst_reg_gp = hdf_file[string(path, "/inst_reg")]
    inst_reg_mats = SparseMatrixCSC[spmat_from_hdf(inst_reg_gp, k) for k in sort(keys(inst_reg_gp))]
   
    # Feature regularizers
    feat_reg_gp = hdf_file[string(path, "/feat_reg")]
    feat_reg_mats = SparseMatrixCSC[spmat_from_hdf(inst_reg_gp, k) for k in sort(keys(feat_reg_gp))]

    # Factor importances
    factor_importances = hdf_file[string(path,"/factor_importances")][:]

    return MatFacModel(X, Y, inst_reg_mats, feat_reg_mats,
                             instance_offset, instance_offset_reg, 
                             feature_offset, feature_offset_reg,
                             feature_precision,
                             covariate_coeff, covariate_coeff_reg,
                             losses, factor_importances) 
end


##################################
# Vector of matrices
##################################

function Base.write(hdf_file::Union{HDF5.File, HDF5.Group},
                    path::String, vec::Vector{T}) where T <: AbstractMatrix
    for (i, mat) in enumerate(vec)
        write(hdf_file, string(path, "/", i), mat)
    end
end


##################################
# Sparse Matrices
##################################

function Base.write(hdf_file::Union{HDF5.File, HDF5.Group}, 
                    path::String, mat::SparseMatrixCSC)

    write(hdf_file, string(path, "/colptr"), mat.colptr)
    write(hdf_file, string(path, "/m"), mat.m)
    write(hdf_file, string(path, "/n"), mat.n)
    write(hdf_file, string(path, "/nzval"), mat.nzval)
    write(hdf_file, string(path, "/rowval"), mat.rowval)
end


function Base.write(hdf_file, path::String, mat::CUDA.CUSPARSE.CuSparseMatrixCSC)
    write(hdf_file, path, SparseMatrixCSC(mat))
end


function spmat_from_hdf(hdf_file, path::String)
    colptr = read(hdf_file, string(path, "/colptr"))
    m = read(hdf_file, string(path, "/m"))
    n = read(hdf_file, string(path, "/n"))
    nzval = read(hdf_file, string(path, "/nzval"))
    rowval = read(hdf_file, string(path, "/rowval"))

    return SparseMatrixCSC(m, n, colptr, rowval, nzval)
end


#################################
# `readtype` function
#################################


function readtype(hdf_file::Union{HDF5.File,HDF5.Group}, 
                  path::String, ::Type{AbstractArray{T}}) where T <: Number
    return read(hdf_file, path)
end

function readtype(hdf_file::Union{HDF5.File,HDF5.Group}, 
                  path::String, ::Type{AbstractArray{T}}) where T <: String
    return read(hdf_file, path)
end




