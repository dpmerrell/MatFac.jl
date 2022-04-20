
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
                         "ternary"=>logistic_link,
                         "poisson"=>poisson_link,
                         "noloss"=>noloss_link
                        )


####################################
# LOSS FUNCTIONS
####################################

function quad_loss(A::AbstractArray, D::AbstractArray, missing_mask,
                   nonmissing)
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


function logistic_loss(A::AbstractArray, D::AbstractArray, missing_mask,
                       nonmissing)
    loss = -D .* log.(A) .- (1 .- D) .* log.( 1 .- A)
    return loss
end


function ChainRules.rrule(::typeof(logistic_loss), A, D, missing_mask,
                          nonmissing)
    
    A .*= nonmissing
    A .+= missing_mask
    
    loss = logistic_loss(A,D, missing_mask, nonmissing)

    function logistic_loss_pullback(loss_bar)
        A_bar = loss_bar .* (-D./A .+ (1 .- D)./(1 .- A))
        return ChainRules.NoTangent(), A_bar, ChainRules.NoTangent(),
                                              ChainRules.NoTangent(),
                                              ChainRules.NoTangent()
    end
    return loss, logistic_loss_pullback 
end


function ternary_loss(A::AbstractArray, D::AbstractArray, missing_mask, nonmissing)

    loss = -0.5f0.*(D .+ 1).*log.(A) .- 0.5f0.*(1 .- D).*log.(1 .- A)
    return loss 
end


function ChainRules.rrule(::typeof(ternary_loss), A, D, missing_mask,
                          nonmissing)
    A .*= nonmissing
    A .+= missing_mask

    # Data with value 0 should not 
    # contribute to the loss
    A[D .== 0] .= 0.5f0

    loss = ternary_loss(A, D, missing_mask, nonmissing)

    function ternary_loss_pullback(loss_bar)
        A_bar = loss_bar .* (-0.5f0.*(D .+ 1)./A .+ 0.5f0.*(1 .- D)./(1 .- A))
        return ChainRules.NoTangent(), A_bar, ChainRules.NoTangent(),
                                              ChainRules.NoTangent(),
                                              ChainRules.NoTangent()
    end
    return loss, ternary_loss_pullback
end


function poisson_loss(A::AbstractArray, D::AbstractArray, missing_mask, nonmissing) 
    return A .- D.*log.(A)
end


function ChainRules.rrule(::typeof(poisson_loss), A, D, missing_mask,
                          nonmissing)
    
    A .*= nonmissing
    A .+= missing_mask

    loss = poisson_loss(A, D)

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
                         "ternary"=>ternary_loss,
                         "poisson"=>poisson_loss,
                         "noloss"=>noloss_loss
                         )


###################################
# Random Sampling
###################################

function normal_sample(A; scale=1.0)
    return A .+ (scale .* CUDA.randn(size(A)...))
end


function bernoulli_sample(A; scale=1.0)
    D = CUDA.rand(size(A)...)
    D .= (D .<= A)
    return D
end

function ternary_sample(A; scale=0.01)
    X = (CUDA.rand(size(A)...) .<= A)
    Y = 2.0f0.*(CUDA.rand(size(A)...) .<= scale) .- 1
    return X .* Y
end

#TODO
#function poisson_sample(A; scale=1.0)
#    D = CUDA.rand_poisson(size(A); lambda=scale)
#end

function noloss_sample(A; scale=1.0)
    return A
end

SAMPLE_FUNCTION_MAP = Dict("normal"=> normal_sample,
                           "logistic"=> bernoulli_sample,
                           "ternary"=> ternary_sample,
                           "noloss"=> noloss_sample)


