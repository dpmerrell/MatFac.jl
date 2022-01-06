


NOISE_MODELS = ["normal","logistic","poisson"]


####################################
# LINK FUNCTIONS
####################################

function quad_link!(Z::AbstractArray)
    return Z
end


function rrule!(::typeof(quad_link!), Z::AbstractArray)

    quad_link!(Z)
    function quad_link_pullback!(A_bar)
        return # Do nothing!
    end
    return quad_link_pullback!
end


function logistic_link!(Z::AbstractArray)
    Z .*= -1
    Z .= exp.(Z)
    Z .+= 1
    Z .= 1/Z
    return Z 
end


function rrule!(::typeof(logistic_link!), Z::AbstractArray, output_buffer::AbstractArray)
    logistic_link!(Z)
    output_buffer .= Z
    function logistic_link_pullback!(A_bar)
        A_bar .*= output_buffer .* (1 .- output_buffer)
        return
    end
    return logistic_link_pullback!
end


function poisson_link!(Z::AbstractArray)
    Z .= exp.(Z)
    return Z
end


function rrule!(::typeof(poisson_link!), Z::AbstractArray, output_buffer::AbstractArray)
    poisson_link!(Z)
    output_buffer .= Z
    function poisson_link_pullback!(A_bar)
        A_bar .*= output_buffer 
        return
    end
    return poisson_link_pullback!
end


LINK_FUNCTION_MAP = Dict("normal"=>quad_link!,
                         "logistic"=>logistic_link!,
                         "poisson"=>poisson_link!
                         )


####################################
# LOSS FUNCTIONS
####################################

function quad_loss!(A::AbstractArray, D::AbstractArray)
    A .-= D
    A .= A.^2
    return A
end

function rrule!(::typeof(quad_loss!), A::AbstractArray, D::AbstractArray, buffer::AbstractArray)
    A .-= D
    buffer .= A
    A .= A.^2

    function quad_loss_pullback!(err_bar)
        err_bar .*= 2.0f0 .* buffer 
    end

    return quad_loss_pullback!
end


function logistic_loss!(A::AbstractArray, D::AbstractArray, buffer::AbstractArray)
    buffer .= 1 .- A

    A .= log.(A)
    A .*= D

    buffer .= log.(buffer)
    buffer .*= (1 .- D)

    A .+= buffer
    return  A 
end

function rrule!(::typeof(logistic_loss!), A::AbstractArray, D::AbstractArray, buffer::AbstractArray)
    
    logistic_loss!(A, D, buffer)

    function logistic_loss_pullback!(err_bar)
        err_bar .*= (1 .- D)./(1 .- A) - D./A
    end

    return logistic_loss_pullback!
end


function poisson_loss!(A::AbstractArray, D::AbstractArray)
    A .-= (D.*log.(A))
    return A 
end


function rrule!(::typeof(poisson_loss!), A::AbstractArray, D::AbstractArray)
    poisson_loss!(A,D)

    function poisson_loss_pullback!(err_bar)
        err_bar .*= (1 .- D./A)
    end
    return poisson_loss_pullback!

end


LOSS_FUNCTION_MAP = Dict("normal"=>quad_loss!,
                         "logistic"=>logistic_loss!,
                         "poisson"=>poisson_loss!
                         )



