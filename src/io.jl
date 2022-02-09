

export save_hdf, model_from_hdf

import Base: write


####################################
# TO HDF5
####################################

# A model or CPU sparse matrix
function Base.write(hdf_file::Union{HDF5.File,HDF5.Group}, path::String, 
                    obj::SparseMatrixCSC)
    for prop in propertynames(obj)
        write(hdf_file, string(path, "/", prop), getproperty(obj, prop))
    end
end


# A CUDA sparse matrix
function Base.write(hdf_file::Union{HDF5.File,HDF5.Group}, path::String, 
                    obj::CuSparseMatrixCSC)
    write(hdf_file, path, SparseMatrixCSC(obj))
end


# A CuArray
function Base.write(hdf_file::Union{HDF5.File,HDF5.Group}, path::String, 
                obj::CuArray{<:Union{Number,String}}) 
    write(hdf_file, path, Array(obj))
end


# A vector of sparse matrices
function Base.write(hdf_file::Union{HDF5.File, HDF5.Group}, path::String, 
                    vec::AbstractVector{<:Union{SparseMatrixCSC,CuSparseMatrixCSC,Vector}})

    for (i, obj) in enumerate(vec)
        write(hdf_file, string(path, "/", i), obj)
    end
end


function Base.write(hdf_file::Union{HDF5.File,HDF5.Group}, path::String, obj::BMFModel)
    for prop in propertynames(obj)
        write(hdf_file, string(path, "/", prop), getproperty(obj, prop))
    end
end


###################################
# FROM HDF5
###################################



# Base case: scalar values
function readtype(hdf_file, path::String, t::Type{T}) where T <: Union{Number,String}
    return read(hdf_file[path])
end

# Base case: ordinary data arrays
function readtype(hdf_file, path::String, t::Type{T}) where T <: DenseArray{<:Union{Number,String}}
    return read(hdf_file[path])
end

# Base case: CUDA arrays
function readtype(hdf_file, path::String, t::Type{T}) where T <: CuArray{<:Union{Number,String}}
    return t(read(hdf_file[path]))
end

# Recursive case: Sparse matrix
function readtype(hdf_file, path::String, t::Type{<:SparseMatrixCSC})

    field_values = []
    for fn in fieldnames(t)
        push!(field_values, read(hdf_file[string(path, "/", fn)]))
    end

    return t(field_values...)
end


# Recursive case: CUDA sparse matrix 
function readtype(hdf_file, path::String, t::Type{<:CuSparseMatrixCSC})
    return t(readtype(hdf_file, path, SparseMatrixCSC))
end


# Recursive case: Vector of sparse matrices
function readtype(hdf_file, path::String, ::Type{Vector{T}}) where T <: Union{SparseMatrixCSC,CuSparseMatrixCSC,Vector}
    
    result = T[]
    for idx in sort(keys(hdf_file[path]))
        push!(result, readtype(hdf_file, string(path,"/",idx), T))
    end

    return result
end

# Recursive case: Model 
function readtype(hdf_file, path::String, t::Type{BMFModel})

    field_values = []
    for (fn, ft) in zip(fieldnames(t), fieldtypes(t))
        push!(field_values, readtype(hdf_file, string(path, "/", fn), ft))
    end

    return t(field_values...)
end

function model_from_hdf(hdf_file, path::String)

    return readtype(hdf_file, path, BMFModel)

end



