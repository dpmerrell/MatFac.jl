

########################################
# UNIFIED INTERFACE FOR ARITHMETIC OPS
########################################

# Unary operations

function my_sq(x::AbstractArray{T}) where T<:Number
    return x.*x
end

function my_sqrt(x::AbstractArray{T}) where T<:Number
    return sqrt.(x)
end

function my_sq(v::Vector{Dict{T,U}}) where T where U<:Number
    return Dict{T,U}[map(x->x*x, d) for d in v] 
end
        
function my_sqrt(v::Vector{Dict{T,U}}) where T where U<:Number
    return Dict{T,U}[map(sqrt, d) for d in v] 
end


# Binary operations

function binop(op::Function, a::Vector{AbstractArray},
                             b::Vector{AbstractArray})
    return [op(u,v) for (u,v) in zip(a,b)]
end

function binop(op::Function, a::AbstractArray{<:Number},
                             b::AbstractArray{<:Number})
    return op(a,b)
end

function binop!(op::Function, a::Vector{<:AbstractArray},
                              b::Vector{<:AbstractArray})
    for (u,v) in zip(a,b)
        u .= op(u,v)
    end
end

function binop(func::Function, d1::Dict{T,U}, d2::Dict{T,U}) where T where U<:Number
    return Dict{T,U}(k => func(a, d2[k]) for (k,a) in d1)
end

function binop(func::Function, d1::Dict{T,U}, x::Number) where T where U<:Number
    return Dict{T,U}(k => func(a, x) for (k,a) in d1)
end

function binop!(func::Function, d1::Dict{T,U}, d2::Dict{T,U}) where T where U<:Number
    for (k,a) in d1
        d1[k] = func(a, d2[k])
    end
end

function binop!(func::Function, d1::Dict{T,U}, x::Number) where T where U<:Number
    for (k,a) in d1
        d1[k] = func(a, x)
    end
end

function binop!(op::Function, a::Vector{Dict{T,U}},
                b::Vector{Dict{T,U}}) where T where U <:Number
    for (a_d, b_d) in zip(a,b)
        binop!(op, a_d, b_d)
    end
end

function binop!(op::Function, a::Vector{Dict{T,U}},
                b::V) where T where U <:Number where V<:Number
    for a_d in a
        binop!(op, a_d, b)
    end
end

function binop(op::Function, a::Vector{Dict{T,U}},
               b::Vector{Dict{T,U}}) where T where U <:Number
    result = Dict{T,U}[]
    for (a_d, b_d) in zip(a,b)
        push!(result, binop(op, a_d, b_d))
    end
    return result
end

function binop(op::Function, a::Vector{Dict{T,U}},
               b::Number) where T where U <:Number
    result = Dict{T,U}[]
    for a_d in a
        push!(result, binop(op, a_d, b))
    end
    return result
end

function binop!(op::Function, a::AbstractArray{<:Number},
                              b::AbstractArray{<:Number})
    a .= op(a,b)
end

function binop(op::Function, a::Vector{<:AbstractArray},
                             k::Number)
    return [op(u,k) for u in a]
end

function binop(op::Function, a::AbstractArray{<:Number},
                             k::Number)
    return op(a,k)
end

function binop!(op::Function, a::Vector{<:AbstractArray},
                              k::Number)
    for u in a
        u .= op(u,k)
    end
end

function binop!(op::Function, a::AbstractArray{<:Number},
                              k::Number)
    a .= op(a,k)
end



