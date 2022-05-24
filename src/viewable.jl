
import Base: view

mutable struct ViewableFunction
    callable::Function
end

function view(vf::ViewableFunction, idx...)
    return vf
end

function (vf::ViewableFunction)(args...)
    return vf.callable(args...)
end

#Flux.trainable(vf::ViewableFunction) = ()

# We only want to make *Functions* viewable.
# Everything else should either be
# (1) a mutable struct with `view` defined; or
# (2) something else that causes a problem.
function make_viewable(obj)
    return obj
end

function make_viewable(obj::Function)
    return ViewableFunction(obj)
end

# Function should be idempotent
function make_viewable(obj::ViewableFunction)
    return obj
end

