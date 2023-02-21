
import Base: view


###############################################
# Wrapper for layers that are not `view`-able
###############################################

mutable struct NoViewWrapper
    callable
end

function Base.view(nvw::NoViewWrapper, idx...)
    return nvw
end

function Base.getindex(nvw::NoViewWrapper, idx...)
    return nvw
end

function (nvw::NoViewWrapper)(args...)
    return nvw.callable(args...)
end

# Anything that doesn't have `view` defined
# should be wrapped.
function make_viewable(obj)
    T = typeof(obj)
    if (hasmethod(view, Tuple{T,Any}) | 
        hasmethod(view, Tuple{T,Any,Any})|
        hasmethod(view, Tuple{T,AbstractRange})|
        hasmethod(view, Tuple{T,AbstractRange,AbstractRange})) 
       return obj
    else
        return NoViewWrapper(obj)
    end
end

# Function should be idempotent
function make_viewable(obj::NoViewWrapper)
    return obj
end

