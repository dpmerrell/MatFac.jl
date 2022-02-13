
export total_loss


####################################
# LINK FUNCTIONS
####################################

function quad_link(Z::AbstractArray)
    return Z
end


function logistic_link(Z::AbstractArray)
    # We shrink the value slightly toward 0.5
    # in order to prevent NaNs.
    return 0.5f0 .+ (0.9999f0 .*(1 ./ (1 .+ exp.(-Z)) .- 0.5))
end


function ChainRules.rrule(::typeof(logistic_link), Z)
    A = logistic_link(Z)

    function logistic_link_pullback(A_bar)
        return ChainRules.NoTangent(), A_bar .* (A .* (1 .- A))
    end
    return A, logistic_link_pullback
end


function poisson_link(Z::AbstractArray)
    return exp.(Z)
end


function noloss_link(Z::AbstractArray)
    return Z
end


LINK_FUNCTION_MAP = Dict("normal"=>quad_link,
                         "logistic"=>logistic_link,
                         "poisson"=>poisson_link,
                         "noloss"=>noloss_link,
                        )


####################################
# LOSS FUNCTIONS
####################################

function quad_loss(A::AbstractArray, D::AbstractArray, 
                   missing_mask::AbstractArray,
                   nonmissing::AbstractArray)
    return BMFFloat(0.5).*(A .- D).^2
end

function ChainRules.rrule(::typeof(quad_loss), A, D, missing_mask,
                          nonmissing)
   
    diff = A .- D
    diff .*= nonmissing

    function quad_loss_pullback(loss_bar)
        return ChainRules.NoTangent(), loss_bar.*diff, ChainRules.NoTangent(),
                                                       ChainRules.NoTangent(),
                                                       ChainRules.NoTangent()
    end

    return BMFFloat(0.5).*(diff.^2), quad_loss_pullback 
end


function logistic_loss(A::AbstractArray, D::AbstractArray,
                       missing_mask::AbstractArray,
                       nonmissing::AbstractArray)
    loss = -D .* log.(A) .- (1 .- D) .* log.( 1 .- A)
    return loss
end


function ChainRules.rrule(::typeof(logistic_loss), A, D, missing_mask,
                          nonmissing)
    
    A .*= nonmissing
    A .+= missing_mask
    
    loss = logistic_loss(A,D,missing_mask,nonmissing)

    function logistic_loss_pullback(loss_bar)
        A_bar = loss_bar .* (-D./A .+ (1 .- D)./(1 .- A))
        return ChainRules.NoTangent(), A_bar, ChainRules.NoTangent(),
                                              ChainRules.NoTangent(),
                                              ChainRules.NoTangent()
    end
    return loss, logistic_loss_pullback 
end


function poisson_loss(A::AbstractArray, D::AbstractArray, 
                      missing_mask::AbstractArray,
                      nonmissing::AbstractArray)
    return A .- D.*log.(A)
end


function ChainRules.rrule(::typeof(poisson_loss), A, D, missing_mask,
                          nonmissing)
    
    A .*= nonmissing
    A .+= missing_mask

    loss = poisson_loss(A, D, missing_mask, nonmissing)

    function poisson_loss_pullback(loss_bar)
        A_bar = loss_bar .* (1 .- D ./ A)
        return ChainRules.NoTangent(), A_bar, ChainRules.NoTangent(),
                                              ChainRules.NoTangent(),
                                              ChainRules.NoTangent()
    end

    return loss, poisson_loss_pullback 
end


function noloss_loss(A::AbstractArray, D::AbstractArray,
                     missing_mask::AbstractArray,
                     nonmissing::AbstractArray)
    return zero(A)
end


function ChainRules.rrule(::typeof(noloss_loss), A, D, missing_mask,
                          nonmissing)
    
    function noloss_loss_pullback(loss_bar)
        return ChainRules.NoTangent(), zero(loss_bar), ChainRules.NoTangent(),
                                                       ChainRules.NoTangent(),
                                                       ChainRules.NoTangent()
    end

    return zero(A), noloss_loss_pullback 
end

LOSS_FUNCTION_MAP = Dict("normal"=>quad_loss,
                         "logistic"=>logistic_loss,
                         "poisson"=>poisson_loss,
                         "noloss"=>noloss_loss
                         )


