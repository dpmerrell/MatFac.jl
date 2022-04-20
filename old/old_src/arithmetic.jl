

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

function my_sq(v::Vector{Vector{U}}) where U<:Number
    return Vector{U}[x .* x for x in v] 
end
        
function my_sqrt(v::Vector{Vector{U}}) where U<:Number
    return Vector{U}[sqrt.(x) for x in v] 
end

########################################
# Binary operations
########################################

# Mutators
function binop!(op::Function, a::Vector{<:AbstractArray},
                              b::Vector{<:AbstractArray})
    for (u,v) in zip(a,b)
        u .= op(u,v)
    end
end


function binop!(op::Function, a::AbstractArray{<:Number},
                              b::AbstractArray{<:Number})
    a .= op(a,b)
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

# Pure functions
function binop(op::Function, a::AbstractArray{<:Number},
                             b::Number)
    return op(a,b)
end

function binop(op::Function, a::Vector{<:AbstractArray},
                             k::Number)
    return [op(u,k) for u in a]
end

