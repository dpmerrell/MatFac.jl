

import Base: view


################################
# Normal noise model

# inverse link

mutable struct NormalInvLink end

function (ni::NormalInvLink)(A::AbstractMatrix)
    return A
end

function view(ni::NormalInvLink, idx)
    return ni
end


# loss function
mutable struct NormalLoss end

function (nl::NormalLoss)(Z::AbstractMatrix, D::AbstractMatrix)
    diff = (Z .- D)
    return 0.5 .* diff .* diff
end


function ChainRules.rrule(nl::NormalLoss, Z, D)

    diff = (Z .- D)
    loss = diff.*diff
    nonnan = (!isnan).(D)
    loss .*= nonnan

    function normalloss_pullback(loss_bar)
        Z_bar = loss_bar .* diff
        return ChainRulesCore.NoTangent(), Z_bar, 
               ChainRulesCore.NoTangent()
    end

    return loss, normalloss_pullback
end


function view(nl::NormalLoss, idx)
    return nl
end



################################
# Bernoulli (logistic) noise model

# inverse link function

sigmoid(x) = 1 ./ (1 .+ exp.(-x))

mutable struct BernoulliInvLink end


function (bi::BernoulliInvLink)(A::AbstractMatrix)
    return sigmoid(A) 
end


function ChainRules.rrule(bi::BernoulliInvLink, A)

    Z = bi(A)

    function bernoulli_link_pullback(Z_bar)
        return ChainRulesCore.NoTangent(), Z_bar .* Z .* (1 .- Z),
               ChainRulesCore.NoTangent()
    end

    return Z, bernoulli_link_pullback
end


function view(bi::BernoulliInvLink, idx)
    return bi
end


# Loss function

mutable struct BernoulliLoss end

function (bl::BernoulliLoss)(Z::AbstractMatrix, D::AbstractMatrix)
    return - D.*log.(Z) - (1 .- D).*log.(1 .- Z)
end


function ChainRules.rrule(bl::BernoulliLoss, Z, D)
    
    result = bl(Z,D)
    nonnan = (!isnan).(D)
    result .*= nonnan

    function bernoulli_loss_pullback(loss_bar)
        Z_bar = loss_bar.*(-D./Z + (1 .- D) ./ (1 .- Z))
        Z_bar .*= nonnan
        return ChainRulesCore.NoTangent(), Z_bar,
               ChainRulesCore.NoTangent()
    end

    return result, bernoulli_loss_pullback 
end


################################
# Poisson noise model

# inverse link function

mutable struct PoissonInvLink end


function (pil::PoissonInvLink)(A::AbstractMatrix)
    return exp.(A)
end


function view(pil::PoissonInvLink, idx)
    return pil
end


# Loss function

mutable struct PoissonLoss end

function (pil::PoissonLoss)(Z::AbstractMatrix, D::AbstractMatrix)
    return Z .- D.*log.(Z)
end


function ChainRules.rrule(pil::PoissonLoss, Z, D)
    
    result = pil(Z,D)
    nonnan = (!isnan).(D)
    result .*= nonnan

    function bernoulli_loss_pullback(loss_bar)
        Z_bar = loss_bar.*( 1 .- D ./ Z)
        Z_bar .*= nonnan
        return ChainRulesCore.NoTangent(), Z_bar,
               ChainRulesCore.NoTangent()
    end

    return result, bernoulli_loss_pullback 
end



####################################################
# Ordinal noise model

# Inverse link function

mutable struct OrdinalInvLink end


function (ol::OrdinalInvLink)(A)
    return A
end

function view(ol::OrdinalInvLink, idx)
    return ol
end


# Loss function

mutable struct OrdinalLoss
    ext_thresholds::AbstractVector{<:Number}
end

@functor OrdinalLoss



function (ol::OrdinalLoss)(Z::AbstractMatrix, D::AbstractMatrix)

    l_thresh = ol.ext_thresholds[D]
    r_thresh = ol.ext_thresholds[D .+ UInt8(1)]
    loss = log.(sigmoid(r_thresh .- Z) .- sigmoid(l_thresh .- Z))

    return loss
end


function ChainRules.rrule(ol::OrdinalLoss, Z, D)

    N_bins = length(ol.ext_thresholds)-1
    l_thresh = ol.ext_thresholds[D]
    r_thresh = ol.ext_thresholds[D .+ UInt8(1)]
    sig_r = sigmoid(r_thresh .- Z)
    sig_l = sigmoid(l_thresh .- Z)

    function ordinal_loss_pullback(loss_bar)
        ol_r_bar = loss_bar .* -sig_r .* (1 .- sig_r) ./ (sig_r .- sig_l)
        ol_l_bar = loss_bar .* sig_l .* (1 .- sig_l) ./ (sig_r .- sig_l)
        ol_bar = zeros(N_bins + 1)
        for i=1:N_bins
            relevant_idx = (D .== i) 
            ol_bar[i] += sum(ol_l_bar .* relevant_idx)
            ol_bar[i+1] += sum(ol_r_bar .* relevant_idx)
        end
        ol_bar = convert(typeof(ol.ext_thresholds), ol_bar)

        Z_bar = loss_bar .* (sig_r .+ sig_l .- 1) 
        return ol_bar, Z_bar, ChainRulesCore.NoTangent()
    end
    
    loss = -log.(sig_r - sig_l)

    return loss, ordinal_loss_pullback
end


####################################
## HELPER FUNCTIONS
#
#
#function partition(ext_thresholds::Vector, data::CuArray)
#    result = CUDA.zeros(UInt8, size(data))
#    return abstract_partition!(result, ext_thresholds, data)
#end
#
#
#function partition(ext_thresholds::Vector, data::Matrix)
#    result = zeros(UInt8, size(data))
#    return abstract_partition!(result, ext_thresholds, data)
#end
#
#
#function abstract_partition!(result, ext_thresholds, data)
#    
#    N = length(ext_thresholds)
#    for (i, thresh) in enumerate(ext_thresholds[1:N-1])
#        result[(thresh .<= data) .& (data .< ext_thresholds[i+1])] .= UInt8(i)
#    end
#
#    return result
#end



