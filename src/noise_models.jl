

import Base: view

#########################################################
# Normal noise model
#########################################################
mutable struct NormalNoise 
    weight::AbstractVector
end


@functor NormalNoise
Flux.trainable(nn::NormalNoise) = ()

function NormalNoise(N::Integer)
    return NormalNoise(ones(N))
end

# link
function link(::NormalNoise, A::AbstractMatrix)
    return A
end

# inverse link
function invlink(::NormalNoise, A::AbstractMatrix)
    return A
end


function loss(nn::NormalNoise, Z::AbstractMatrix, D::AbstractMatrix; kwargs...)
    diff = (Z .- D)
    diff .*= diff
    diff .*= 0.5
    diff .*= transpose(nn.weight)
    return diff
end

mean(l) = sum(l) / prod(size(l))

function ChainRulesCore.rrule(::typeof(loss), nn::NormalNoise, Z, D; kwargs...)

    diff = (Z .- D)
    nanvals = (!isfinite).(diff)
    tozero!(diff, nanvals)
    l = diff.*diff
    l .*= 0.5
    l .*= transpose(nn.weight)

    function loss_normal_pullback(loss_bar)
        Z_bar = loss_bar .* diff .* transpose(nn.weight)
        return ChainRulesCore.NoTangent(), 
               ChainRulesCore.NoTangent(),
               Z_bar, 
               ChainRulesCore.NoTangent()
    end

    return l, loss_normal_pullback
end

invlinkloss(nn::NormalNoise, Z, D; kwargs...) = sum(loss(nn::NormalNoise, Z, D; kwargs...))

function Base.view(nn::NormalNoise, idx)
    return NormalNoise(Base.view(nn.weight, idx))
end

function Base.getindex(nn::NormalNoise, idx)
    return NormalNoise(Base.getindex(nn.weight, idx))
end


function link_col_sqerr(nn::NormalNoise, model, D::AbstractMatrix; capacity=10^8, kwargs...)
    return batched_link_col_sqerr(model, D; capacity=capacity, kwargs...) 
end


#########################################################
# Bernoulli (logistic) noise model
#########################################################

mutable struct BernoulliNoise 
    weight::AbstractVector
end

@functor BernoulliNoise
Flux.trainable(bn::BernoulliNoise) = ()

function BernoulliNoise(N::Integer)
    return BernoulliNoise(ones(N))
end

# link function
function logit_kernel(x::T) where T <: Number
    return T(log( x/(1-x) ))
end

function logit(X)
    return map(logit_kernel, X)
end

function link(::BernoulliNoise, A::AbstractMatrix)
    return logit(A)
end

# inverse link function
function sigmoid_kernel(x::T) where T <: Number
    return T(1 / (1 + exp(-x)))
end

function sigmoid(X)
    return map(sigmoid_kernel, X)
end

# Efficient custom rrule
function ChainRulesCore.rrule(::typeof(sigmoid), x)
    
    y = sigmoid(x)

    function sigmoid_pullback(y_bar)
        x_bar = 1 .- y
        x_bar .*= y
        x_bar .*= y_bar
        return ChainRulesCore.NoTangent(), x_bar
    end

    return y, sigmoid_pullback
end

function invlink(bn::BernoulliNoise, A::AbstractMatrix)
    return sigmoid(A) 
end


# Loss function

function cross_entropy_kernel(p::T, d::Number) where T <: Number
    d_t = T(d)
    eps_t = T(1e-15)
    return -d_t * log(p + eps_t) - (1-d_t) * log(1 - p + eps_t) 
end


function loss(bn::BernoulliNoise, Z::AbstractMatrix, D::AbstractMatrix; calibrate=false)
    Z2 = map(cross_entropy_kernel, Z, D)

    # The `calibrate` kwarg indicates whether to subtract
    # the minimum possible loss value from the result.
    if calibrate
        Z2 .-= map(cross_entropy_kernel, D, D)
    end
    Z2 .*= transpose(bn.weight)
    return Z2 
end


function ChainRulesCore.rrule(::typeof(loss), bn::BernoulliNoise, Z, D; kwargs...)
    
    result = loss(bn,Z,D; kwargs...)
    nanvals = (!isfinite).(result)
    tozero!(result, nanvals)
        
    Z_bar = (-D./Z - (1 .- D) ./ (1 .- Z))
    Z_bar .*= transpose(bn.weight)
    tozero!(Z_bar, nanvals)

    function loss_bernoulli_pullback(loss_bar)
        return ChainRulesCore.NoTangent(),
               ChainRulesCore.NoTangent(), 
               loss_bar .* Z_bar,
               ChainRulesCore.NoTangent()
    end

    return result, loss_bernoulli_pullback 
end


invlinkloss(bn::BernoulliNoise, Z, D; kwargs...) = sum(loss(bn, invlink(bn, Z), D; kwargs...))


function ChainRulesCore.rrule(::typeof(invlinkloss), bn::BernoulliNoise, Z, D; kwargs...)

    A = sigmoid(Z)
    diff = A .- D
    nanvals = (!isfinite).(diff) # Handle missing values
    diff .*= transpose(bn.weight)

    function invlinkloss_bernoulli_pullback(loss_bar)

        Z_bar = loss_bar .* diff
        tozero!(Z_bar, nanvals)

        return ChainRulesCore.NoTangent(),
               ChainRulesCore.NoTangent(), 
               Z_bar,
               ChainRulesCore.NoTangent()
    end
    larr = loss(bn, A, D; kwargs...)
    tozero!(larr, nanvals)

    return sum(larr), invlinkloss_bernoulli_pullback
end


function Base.view(bn::BernoulliNoise, idx)
    return BernoulliNoise(Base.view(bn.weight, idx))
end

function Base.getindex(bn::BernoulliNoise, idx)
    return BernoulliNoise(Base.getindex(bn.weight, idx))
end

function link_col_sqerr(bn::BernoulliNoise, model, D::AbstractMatrix; capacity=10^8, kwargs...)
    return vec(batched_column_ssq_grads(model, D; capacity=capacity))
end

#function link_col_sqerr(bn::BernoulliNoise, model, D::AbstractMatrix; capacity=10^8, kwargs...)
#    #N = size(D,2)
#    #p_vec = cpu(column_nanmeans(D))
#    #mu_vec = logit(p_vec)
#    #scale_vec = similar(D, N)
#    #scale_vec .= mu_vec ./ quantile(Normal(), p_vec)
#    return scale_vec.^2
#end

#########################################################
# Squared hinge "noise" (for binary variables)
#########################################################

mutable struct SquaredHingeNoise
    weight::AbstractVector
end

@functor SquaredHingeNoise
Flux.trainable(shn::SquaredHingeNoise) = ()

function SquaredHingeNoise(N::Integer)
    return SquaredHingeNoise(ones(N))
end

function link(shn::SquaredHingeNoise, D::AbstractMatrix)
    result = copy(D)
    result[D .== 0] .= -1

    return result
end

function invlink(shn::SquaredHingeNoise, A::AbstractMatrix)
    return A
end

function loss(shn::SquaredHingeNoise, A::AbstractMatrix, D::AbstractMatrix; kwargs...)

    nan_idx = (!isfinite).(D)
    result = zero(A)

    # Convert data to +/- 1 
    S = sign.(D .- Float32(0.5))

    # We'll zero out the entries where
    # (1) A and data have the same sign and
    # (2) A has absolute value > 1
    mask_idx = (!).(S.*A .> 1)

    result .= (A .- S)
    result .*= result
    result .*= Float32(0.5)
    
    result .*= mask_idx

    result .*= transpose(shn.weight)

    result[nan_idx] .= 0
    return result
end


function ChainRulesCore.rrule(::typeof(loss), shn::SquaredHingeNoise, A::AbstractMatrix, D::AbstractMatrix; kwargs...)

    nan_idx = (!isfinite).(D)

    S = sign.(D .- Float32(0.5))
    mask_idx = (!).(S.*A .> 1)
    A_grad = (A .- S)
    A_grad .*= mask_idx
    A_grad[nan_idx] .= 0

    result = Float32(0.5) .* A_grad .* A_grad

    function sq_hinge_loss_pullback(loss_bar)
        A_bar = (loss_bar .* A_grad) .* transpose(shn.weight)
        return NoTangent(), NoTangent(),
               A_bar, NoTangent()
    end

    result .*= transpose(shn.weight) 

    return result, sq_hinge_loss_pullback
end

invlinkloss(shn::SquaredHingeNoise, A::AbstractMatrix, D::AbstractMatrix; kwargs...) = sum(loss(shn, A, D; kwargs...))

function Base.view(pn::SquaredHingeNoise, idx)
    return SquaredHingeNoise(Base.view(pn.weight, idx))
end

function link_col_sqerr(shn::SquaredHingeNoise, model, D::AbstractMatrix; capacity=10^8, kwargs...)
    #return batched_link_col_sqerr(model, D; capacity=capacity, kwargs...)
    #M, N = size(D)
    #result = similar(D, N)
    #result .= M
    #return result
    return column_nonnan(D)
end

#function link_col_sqerr(shn::SquaredHingeNoise, model, D::AbstractMatrix; capacity=10^8, kwargs...)
#    N = size(D,2)
#    p_vec = cpu(column_nanmeans(D))
#    mu_vec = logit(p_vec)
#    scale_vec = similar(D, N)
#    scale_vec .= mu_vec ./ quantile(Normal(), p_vec)
#    return scale_vec.^2
#end

#########################################################
# Poisson noise model
#########################################################

mutable struct PoissonNoise 
    weight::AbstractVector
end

@functor PoissonNoise
Flux.trainable(pn::PoissonNoise) = ()

function PoissonNoise(N::Integer)
    return PoissonNoise(ones(N))
end

# link function

function link(pn::PoissonNoise, A::AbstractMatrix)
    return log.(A)
end

# inverse link function

function invlink(pn::PoissonNoise, A::AbstractMatrix)
    return exp.(A)
end

# Loss function
function poisson_loss_kernel(z::T, d::Number) where T <: Number
    d_t = T(d)
    eps_t = T(1e-15)
    return z - d_t*log(z + eps_t)
end

function loss(pn::PoissonNoise, Z::AbstractMatrix, D::AbstractMatrix; calibrate=false)
    
    pl = map(poisson_loss_kernel, Z, D) 
    if calibrate
        pl .-= map(poisson_loss_kernel, D, D)
    end
    pl .*= transpose(pn.weight)
    return pl 

end


function ChainRulesCore.rrule(::typeof(loss), pn::PoissonNoise, Z, D; kwargs...)
    
    result = loss(pn, Z, D; kwargs...)
    nanvals = (!isfinite).(D)
    tozero!(result, nanvals)
        
    Z_bar = (1 .- D ./ Z)
    Z_bar .*= transpose(pn.weight)
    tozero!(Z_bar, nanvals)

    function loss_poisson_pullback(loss_bar)
        return ChainRulesCore.NoTangent(),
               ChainRulesCore.NoTangent(), 
               loss_bar .* Z_bar,
               ChainRulesCore.NoTangent()
    end

    return result, loss_poisson_pullback 
end


invlinkloss(pn::PoissonNoise, Z, D; kwargs...) = sum(loss(pn, invlink(pn, Z), D; kwargs...))


function ChainRulesCore.rrule(::typeof(invlinkloss), pn::PoissonNoise, Z, D; kwargs...)

    nanvals = (!isfinite).(D)
    A = invlink(pn, Z)
        
    diff = A .- D
    diff .*= transpose(pn.weight)
    tozero!(diff, nanvals)

    function invlinkloss_poisson_pullback(result_bar)
        return ChainRulesCore.NoTangent(),
               ChainRulesCore.NoTangent(), 
               result_bar .* diff,
               ChainRulesCore.NoTangent()
    end

    larr = loss(pn, A, D; kwargs...)
    tozero!(larr, nanvals)
    result = sum(larr)

    return result, invlinkloss_poisson_pullback
end


function Base.view(pn::PoissonNoise, idx)
    return PoissonNoise(Base.view(pn.weight, idx))
end

function Base.getindex(pn::PoissonNoise, idx)
    return PoissonNoise(Base.getindex(pn.weight, idx))
end

poisson_sample(z) = rand(Poisson(z))


function link_col_sqerr(pn::PoissonNoise, model, D::AbstractMatrix; capacity=10^8, kwargs...)
    return batched_link_col_sqerr(model, D; capacity=capacity, kwargs...)
end


#########################################################
# Ordinal noise model
#########################################################

mutable struct OrdinalNoise
    weight::AbstractVector
    ext_thresholds::AbstractVector{<:Number}
end

@functor OrdinalNoise
Flux.trainable(on::OrdinalNoise) = (ext_thresholds=on.ext_thresholds,)

function OrdinalNoise(weight::AbstractVector, n_values::Integer)
    thresholds = collect(1:(n_values - 1)) .- 0.5*n_values
    ext_thresholds = [[-Inf]; thresholds; [Inf]]
    return OrdinalNoise(weight, ext_thresholds)
end

function OrdinalNoise(N::Integer, ext_thresholds::Vector{<:Number})
    return OrdinalNoise(ones(N), ext_thresholds)
end

function OrdinalNoise(N::Integer, n_values::Integer)
    return OrdinalNoise(ones(N), n_values)
end

nanround(x) = isfinite(x) ? round(UInt8, x) : UInt8(1)

# link function

function link(on::OrdinalNoise, D)

    nan_idx = (!isfinite).(D)
    D_idx = nanround.(D)
    R_idx = D_idx .+ UInt8(1) 

    sort!(on.ext_thresholds)
    l_thresh = on.ext_thresholds[D_idx]
    r_thresh = on.ext_thresholds[R_idx]

    thresholds_cpu = cpu(on.ext_thresholds)
    T = eltype(on.ext_thresholds)
    th_max = T(thresholds_cpu[end-1] + 1)
    th_min = T(thresholds_cpu[2] - 1)
    map!( th -> (isfinite(th) ? th : th_max), r_thresh, r_thresh)
    map!( th -> (isfinite(th) ? th : th_min), l_thresh, l_thresh)
    centers = (r_thresh .+ l_thresh).*T(0.5)

    tonan!(centers, nan_idx)
    return centers
end


# inverse link function (**WARNING: costly!** we rarely ever use this.
# Whenever possible, it's better to call `invlinkloss` and directly compute loss
# without explicitly constructing the inverse-linked values.)
# Also, it returns a 3D array -- which makes it inconsistent with other
# noise models' inverse link functions.
function invlink(on::OrdinalNoise, A)

    K = length(on.ext_thresholds) - 1
    l_thresh = on.ext_thresholds[1:K]
    l_thresh = reshape(l_thresh, (1,1,K)) 
    r_thresh = on.ext_thresholds[2:end]
    r_thresh = reshape(r_thresh, (1,1,K)) 

    # Need to allocate a 3D array!
    sig = similar(A, (size(A)..., K))
    sig .= sigmoid(A .- l_thresh) .- sigmoid(A .- r_thresh)

    return sig  
end


function ChainRulesCore.rrule(::typeof(invlink), on::OrdinalNoise, A::AbstractMatrix)
    
    K = length(on.ext_thresholds)
    thresh = reshape(on.ext_thresholds, (1,1,K))

    # Have to allocate 3D arrays!
    sig = similar(A, (size(A)..., K))
    sig .= sigmoid(A .- thresh)

    function ordinal_invlink_pullback(result_bar)
        sig_var = sig .* (1 .- sig)
        sig_var_l = view(sig_var, :, :, 1:(K-1))
        sig_var_r = view(sig_var, :, :, 2:K)
       
        sig_var_diff = sig_var_l .- sig_var_r
        sig_var_diff .*= result_bar
        
        A_bar = sum(sig_var_diff, dims=3)[:,:,1]

        thresh_bar = zero(on.ext_thresholds)
        thresh_bar[2:K] .= vec(sum(result_bar.*sig_var_r, dims=(1,2)))
        thresh_bar[1:(K-1)] .-= vec(sum(result_bar.*sig_var_l, dims=(1,2)))
        
        on_bar = Tangent{OrdinalNoise}(ext_thresholds=thresh_bar)

        return NoTangent(), on_bar, A_bar
    end

    sig_l = view(sig, :, :, 1:(K-1))
    sig_r = view(sig, :, :, 2:K)
    result = sig_l .- sig_r

    return result, ordinal_invlink_pullback
end


function loss(on::OrdinalNoise, Z::AbstractArray, D::AbstractMatrix)
 
    K = length(on.ext_thresholds) - 1
 
    nan_idx = (!isfinite).(D)
    D_idx = nanround.(D)

    selected = similar(D)
    selected .= 0
    relevant_idx = similar(D, Bool)
    for k=1:K
        relevant_idx .= (D_idx .== k)
        selected .+= (Z[:,:,k] .* relevant_idx)
    end
    
    lss = -log.(selected)
    lss[nan_idx] .= 0 
    return lss 
end


function ChainRulesCore.rrule(::typeof(loss), on::OrdinalNoise, Z::AbstractArray, D::AbstractMatrix)

    K = length(on.ext_thresholds) - 1
    
    nan_idx = (!isfinite).(D)
    D_idx = nanround.(D)

    selected = similar(D)
    selected .= 0
    relevant_idx = similar(Z, Bool)
    Z_bar = zero(Z)
    for k=1:K
        relevant_idx[:,:,k] .= (D_idx .== k)
    end
    selected = sum(Z .* relevant_idx, dims=3)[:,:,1]
    selected[nan_idx] .= 1

    function ordinal_loss_pullback(loss_bar)
        Z_bar = ((1 ./ Z) .* relevant_idx)
        Z_bar .*= -(loss_bar .* (!).(nan_idx))
        return NoTangent(), NoTangent(), Z_bar, NoTangent()
    end

    return -log.(selected), ordinal_loss_pullback
end


function ordinal_loss_kernel(z::T, l_t::T, r_t::T) where T <: Number
    eps_t = T(1e-15)
    return T(-log(sigmoid_kernel(r_t - z) - sigmoid_kernel(l_t - z) + eps_t))
end


function ordinal_calibration(l_thresh::AbstractMatrix{T}, 
                             r_thresh::AbstractMatrix{T}, 
                             thresholds::AbstractVector{T}) where T <: Number
    thresholds_cpu = cpu(thresholds)
    th_max = T(thresholds_cpu[end-1] + 1e3)
    th_min = T(thresholds_cpu[2] - 1e3)
    map!( th -> (isfinite(th) ? th : th_max), r_thresh, r_thresh)
    map!( th -> (isfinite(th) ? th : th_min), l_thresh, l_thresh)
    centers = (r_thresh .+ l_thresh).*T(0.5)

    return map(ordinal_loss_kernel, centers, l_thresh, r_thresh)
end


function invlinkloss(on::OrdinalNoise, Z::AbstractMatrix{T}, D::AbstractMatrix; calibrate=false) where T <: Number

    nanvals = (!isfinite).(D)
    D_idx = nanround.(D)
    
    sort!(on.ext_thresholds)

    ext_thresholds = T.(on.ext_thresholds)
    l_thresh = ext_thresholds[D_idx]
    r_thresh = ext_thresholds[D_idx .+ UInt8(1)]

    l = map(ordinal_loss_kernel, Z, l_thresh, r_thresh)
    if calibrate
        l .-= ordinal_calibration(l_thresh, r_thresh, ext_thresholds)
    end
    l[nanvals] .= 0
    l .*= transpose(on.weight)

    return l
end


function ChainRulesCore.rrule(::typeof(invlinkloss), on::OrdinalNoise, Z::AbstractMatrix{T}, D; calibrate=false) where T <: Number

    D_idx = nanround.(D)
    R_idx = D_idx .+ UInt8(1) 
    nanvals = (!isfinite).(D)

    sort!(on.ext_thresholds)

    ext_thresholds = T.(on.ext_thresholds) 
    l_thresh = ext_thresholds[D_idx]
    r_thresh = ext_thresholds[R_idx]

    sig_r = sigmoid(r_thresh .- Z)
    toone!(sig_r, nanvals)

    sig_l = sigmoid(l_thresh .- Z)
    tozero!(sig_l, nanvals)
    
    sig_diff = sig_r .- sig_l
    toone!(sig_diff, nanvals) 

    function loss_ordinal_pullback(loss_bar)

        on_r_bar = loss_bar .* -sig_r .* (1 .- sig_r) ./ sig_diff
        on_r_bar .*= transpose(on.weight)
        on_l_bar = loss_bar .* sig_l .* (1 .- sig_l) ./ sig_diff
        on_l_bar .*= transpose(on.weight)

        N_bins = length(ext_thresholds)-1
        on_bar = zeros(N_bins + 1)
        relevant_idx = similar(on_r_bar, Bool)
        for i=1:N_bins
            relevant_idx .= (D_idx .== i) 
            on_bar[i] += sum(on_l_bar .* relevant_idx)
            on_bar[i+1] += sum(on_r_bar .* relevant_idx)
        end
        relevant_idx = nothing

        U = typeof(on.ext_thresholds) # Move to GPU if necessary

        on_bar = Tangent{OrdinalNoise}(ext_thresholds=U(on_bar))
        Z_bar = loss_bar .* (1 .- sig_r .- sig_l) 
        Z_bar .*= transpose(on.weight)
 
        return ChainRulesCore.NoTangent(), 
               on_bar, 
               Z_bar, 
               ChainRulesCore.NoTangent()
    end
    
    l = -log.(sig_diff .+ 1e-15)
    if calibrate
        l .-= ordinal_calibration(l_thresh, r_thresh, ext_thresholds)
    end
    l .*= transpose(on.weight)

    return sum(l), loss_ordinal_pullback
end

function Base.view(on::OrdinalNoise, idx)
    return OrdinalNoise(Base.view(on.weight, idx), on.ext_thresholds)
end

function Base.getindex(on::OrdinalNoise, idx)
    return OrdinalNoise(Base.getindex(on.weight, idx), copy(on.ext_thresholds))
end

function link_col_sqerr(ord::OrdinalNoise, model, D::AbstractMatrix; capacity=10^8, kwargs...)
    #n_levels = length(ord.ext_thresholds) - 1
    #M, N = size(D)
    #result = similar(D, N)
    #result .= (M*n_levels*n_levels)
    #return result
    return column_nonnan(D) .* (n_levels*n_levels)
end


#########################################################
# Ordinal squared hinge noise
#########################################################

mutable struct OrdinalSqHingeNoise
    weight::AbstractVector
    ext_thresholds::AbstractVector{<:Number}
end

@functor OrdinalSqHingeNoise
Flux.trainable(on::OrdinalSqHingeNoise) = (ext_thresholds=on.ext_thresholds,)

function OrdinalSqHingeNoise(weight::AbstractVector, n_values::Integer)
    thresholds = collect(1:(n_values - 1)) .- 0.5*n_values
    ext_thresholds = [[-Inf]; thresholds; [Inf]]
    return OrdinalSqHingeNoise(weight, ext_thresholds)
end

function OrdinalSqHingeNoise(N::Integer, ext_thresholds::Vector{<:Number})
    return OrdinalSqHingeNoise(ones(N), ext_thresholds)
end

function OrdinalSqHingeNoise(N::Integer, n_values::Integer)
    return OrdinalSqHingeNoise(ones(N), n_values)
end

# link function. We'll just map from
# ordinal values to "hidden" values implied by the 
# thresholds.
function link(osh::OrdinalSqHingeNoise, D::AbstractMatrix)

    nan_idx = (!isfinite).(D)

    temp_thresh = Flux.cpu(osh.ext_thresholds)
    min_thresh = temp_thresh[2] - 1
    max_thresh = temp_thresh[end-1] + 1

    L_idx = nanround.(D)
    l_thresh = osh.ext_thresholds[L_idx]
    l_thresh = map(th -> isfinite(th) ? th : min_thresh, l_thresh) 

    R_idx = L_idx .+ UInt8(1) 
    r_thresh = osh.ext_thresholds[R_idx]
    r_thresh = map(th -> isfinite(th) ? th : max_thresh, r_thresh) 

    midpoints = 0.5.*(l_thresh .+ r_thresh)

    midpoints[nan_idx] .= NaN

    return midpoints
end

# inverse link function. We'll just have it do nothing.
invlink(osh::OrdinalSqHingeNoise, A::AbstractMatrix) = A

# loss function
function loss(osh::OrdinalSqHingeNoise, A::AbstractMatrix, D::AbstractMatrix; kwargs...)

    nanvals = (!isfinite).(D)

    sort!(osh.ext_thresholds)

    L_idx = nanround.(D)
    l_thresh = osh.ext_thresholds[L_idx]
    l_diff = A .- l_thresh .- 1
    l_mask = (l_diff .> 0)
    loss = l_diff.*l_diff
    loss[l_mask] .= 0

    R_idx = L_idx .+ UInt8(1) 
    r_thresh = osh.ext_thresholds[R_idx]
    r_diff = A .- r_thresh .+ 1
    r_mask = (r_diff .< 0)
    r_loss = r_diff .* r_diff
    r_loss[r_mask] .= 0
    
    loss .+= r_loss

    loss[nanvals] .= 0
    loss .*= transpose(osh.weight)

    return 0.5.*loss
end


function ChainRulesCore.rrule(::typeof(loss), osh::OrdinalSqHingeNoise, A::AbstractMatrix, D::AbstractMatrix; kwargs...)

    nanvals = (!isfinite).(D)

    sort!(osh.ext_thresholds)

    L_idx = nanround.(D)
    l_thresh = osh.ext_thresholds[L_idx]
    l_diff = A .- l_thresh .- 1
    l_mask = (l_diff .> 0)
    l_diff[l_mask] .= 0
    l_diff[nanvals] .= 0

    R_idx = L_idx .+ UInt8(1) 
    r_thresh = osh.ext_thresholds[R_idx]
    r_diff = A .- r_thresh .+ 1
    r_mask = (r_diff .< 0)
    r_diff[r_mask] .= 0
    r_diff[nanvals] .= 0

    function ordinal_sq_hinge_pullback(loss_bar)
        w = loss_bar .* osh.weight
        A_bar = transpose(w) .* (l_diff .+ r_diff)
        thresh_bar = zeros(Float32, length(osh.ext_thresholds))
      
        for k=2:(length(thresh_bar)-1)
            rel_L_idx = (L_idx .== k)
            rel_R_idx = (R_idx .== k)

            rel_r_diff = rel_R_idx .* r_diff
            rel_l_diff = rel_L_idx .* l_diff

            thresh_bar[k] = -sum(transpose(w) .* (rel_r_diff .+ rel_l_diff))
        end 
        thresh_bar = Flux.gpu(thresh_bar) 
        return NoTangent(), Tangent{OrdinalSqHingeNoise}(ext_thresholds=thresh_bar),
               A_bar, NoTangent()
    end    

    loss = l_diff.*l_diff
    loss = l_diff.*l_diff .+ r_diff.*r_diff
    loss .*= transpose(osh.weight)

    return 0.5.*sum(loss), ordinal_sq_hinge_pullback
end


invlinkloss(osh::OrdinalSqHingeNoise, A::AbstractMatrix, D::AbstractMatrix; kwargs...) = sum(loss(osh, A, D; kwargs...)) 

function view(osh::OrdinalSqHingeNoise, idx)
    return OrdinalSqHingeNoise(view(osh.weight, idx), osh.ext_thresholds)
end

function link_col_sqerr(osh::OrdinalSqHingeNoise, model, D::AbstractMatrix; capacity=10^8, kwargs...)

    n_thresh = length(osh.ext_thresholds) - 2
    M, N = size(D)
    result = similar(D, N)
    #result .= (M*n_thresh*n_thresh)
    result .= M
    return result
end


#########################################################
# "Composite" noise model
#########################################################
# This struct assigns noise models
# to ranges of columns.

mutable struct CompositeNoise
    col_ranges::Tuple # UnitRanges
    noises::Tuple # NoiseModel objects
end

@functor CompositeNoise (noises,)


function CompositeNoise(noise_model_ids::Vector{String}; 
                        weight::Union{Nothing,<:AbstractVector}=nothing)
   
    if weight == nothing
        weight = ones(length(noise_model_ids))
    end

    unq_ids = unique(noise_model_ids)
    ranges = ids_to_ranges(noise_model_ids)
    weights = [view(weight, r) for r in ranges]

    noise_models = map(construct_noise, unq_ids, weights)

    return CompositeNoise(Tuple(ranges), Tuple(noise_models)) 
end


function Base.view(cn::CompositeNoise, idx::AbstractRange)

    new_ranges, r_min, r_max = subset_ranges(cn.col_ranges, idx)
    shifted_new_ranges = shift_range.(new_ranges, (1 - new_ranges[1].start))

    # Create views of the column functions
    newnoise_views = []
    for (n_rng, o_rng, noise) in zip(new_ranges, cn.col_ranges[r_min:r_max], cn.noises[r_min:r_max])
        sh_rng = shift_range(n_rng, (1 - o_rng.start))
        push!(newnoise_views, view(noise, sh_rng))
    end

    return CompositeNoise(Tuple(shifted_new_ranges), 
                          Tuple(newnoise_views))
end


function Base.view(cn::CompositeNoise, idx::Colon)
    idx = 1:cn.col_ranges[end].stop
    return view(cn, idx)
end


function Base.getindex(cn::CompositeNoise, idx)

    # Map from column index to column range index
    idx_to_oldrange = Dict(old_i => k for (k,r) in enumerate(cn.col_ranges) for old_i in r)
    oldranges = map(i -> idx_to_oldrange[i], idx)
    # Collect (a) the starts of new ranges, and
    # (b) the "irredundant" values of the old ranges 
    new_starts = [1]
    cur_r = oldranges[1]
    irredundant_r = [cur_r]
    for (new_i,r) in enumerate(oldranges)
        if r != cur_r
            push!(new_starts, new_i)
            push!(irredundant_r, r)
            cur_r = r
        end
    end
    new_stops = vcat(new_starts[2:end] .- 1, [length(idx)])
    new_ranges = map((strt,stp) -> strt:stp, new_starts, new_stops) 
  
    # Collect the indices for viewing into the old noise models. 
    view_idx = [] 
    for (old_r, rng) in zip(irredundant_r, new_ranges)
        old_range = cn.col_ranges[old_r]
        vi = idx[rng] .- old_range.start .+ 1
        push!(view_idx, vi)
    end 

    new_noises = map((ir, vi) -> cn.noises[ir][vi], irredundant_r, view_idx)

    return CompositeNoise(Tuple(new_ranges), 
                          Tuple(new_noises))
end


link(cn::CompositeNoise, D) = hcat(map((n,rng) -> link(n, view(D,:,rng)), cn.noises, cn.col_ranges)...)
#invlink(cn::CompositeNoise, A) = hcat(map((n,rng)->invlink(n, view(A,:,rng)), cn.noises, cn.col_ranges)...)
loss(cn::CompositeNoise, Z, D; kwargs...) = hcat(map((n,rng)->loss(n, view(Z,:,rng), view(D,:,rng); kwargs...), cn.noises, cn.col_ranges)...)
invlinkloss(cn::CompositeNoise, A, D; kwargs...) = sum(map((n,rng)->invlinkloss(n, view(A,:,rng), view(D,:,rng); kwargs...), cn.noises, cn.col_ranges))
#link_scale(cn::CompositeNoise, D; kwargs...) = vcat(map((n,rng) -> link_scale(n, view(D,:,rng); kwargs...), cn.noises, cn.col_ranges)...)
link_col_sqerr(cn::CompositeNoise, model, D; kwargs...) = vcat(map((n,rng) -> link_col_sqerr(n, view(model,:,rng), view(D,:,rng); kwargs...), cn.noises, cn.col_ranges)...)

function ChainRulesCore.rrule(::typeof(invlinkloss), cn::CompositeNoise, A, D; kwargs...)

    result = 0
    pullbacks = []
    for (noise, rng) in zip(cn.noises, cn.col_ranges)
        lss, pb = Zygote.pullback((n,a,d) -> invlinkloss(n,a,d; kwargs...), noise, view(A,:,rng), view(D, :, rng))
        result += lss
        push!(pullbacks, pb)
    end

    function invlinkloss_composite_pullback(result_bar)
        cn_bar = []
        A_bar = similar(A)
  
        for (rng, pb) in zip(cn.col_ranges, pullbacks)
            noise_bar, abar, _ = pb(result_bar)
            push!(cn_bar, noise_bar)
            A_bar[:,rng] .= abar
        end
        return ChainRulesCore.NoTangent(),
               Tangent{CompositeNoise}(noises=Tuple(cn_bar)),
               A_bar,
               ChainRulesCore.NoTangent()
    end

    return result, invlinkloss_composite_pullback
end


######################################################
# HELPER FUNCTIONS
######################################################

"""
Check that the values in `vec` occur in contiguous blocks.
I.e., the unique values are grouped together, with no intermingling.
I.e., for each unique value the set of indices mapping to that value
occur consecutively.
"""
function is_contiguous(vec::AbstractVector{T}) where T

    past_values = Set{T}()
    
    for i=1:(length(vec)-1)
        next_value = vec[i+1]
        if in(vec[i+1], past_values)
            return false
        end
        if vec[i+1] != vec[i]
            push!(past_values, vec[i])
        end
    end

    return true
end


function ids_to_ranges(id_vec)

    @assert is_contiguous(id_vec) "IDs in id_vec need to appear in contiguous chunks."

    unique_ids = unique(id_vec)
    start_idx = indexin(unique_ids, id_vec)
    end_idx = length(id_vec) .- indexin(unique_ids, reverse(id_vec)) .+ 1
    ranges = UnitRange[start:finish for (start,finish) in zip(start_idx, end_idx)]

    return ranges
end

function subset_ranges(ranges::Vector, rng::UnitRange) 
   
    r_min = rng.start
    r_max = rng.stop
    @assert r_min <= r_max

    @assert r_min >= ranges[1].start
    @assert r_max <= ranges[end].stop

    starts = [rr.start for rr in ranges]
    r_min_idx = searchsorted(starts, r_min).stop
    
    stops = [rr.stop for rr in ranges]
    r_max_idx = searchsorted(stops, r_max).start

    new_ranges = collect(ranges[r_min_idx:r_max_idx])
    new_ranges[1] = r_min:new_ranges[1].stop
    new_ranges[end] = new_ranges[end].start:r_max

    return new_ranges, r_min_idx, r_max_idx
end

function subset_ranges(ranges::Tuple, rng::UnitRange)
    new_ranges, r_min, r_max = subset_ranges(collect(ranges), rng)
    return Tuple(new_ranges), r_min, r_max
end

function shift_range(rng, delta)
    return (rng.start + delta):(rng.stop + delta) 
end


function construct_noise(string_id::String, weight::AbstractVector)

    if string_id == "normal"
        noise = NormalNoise(weight)
    elseif string_id == "bernoulli_sq_hinge"
        noise = SquaredHingeNoise(weight)
    elseif string_id == "bernoulli"
        noise = BernoulliNoise(weight)
    elseif string_id == "poisson"
        noise = PoissonNoise(weight)
    elseif startswith(string_id, "ordinal_sq_hinge")
        n_levels = parse(Int, string_id[17:end])
        noise = OrdinalSqHingeNoise(weight, n_levels)
    elseif startswith(string_id, "ordinal")
        n_levels = parse(Int, string_id[8:end])
        noise = OrdinalNoise(weight, n_levels)
    else
        throw(ArgumentError(string(string_id, " is not a valid noise model identifier")))
    end

    return noise
end


function set_weight!(cn::CompositeNoise, weight::AbstractVector)
    for (cr, n) in zip(cn.col_ranges, cn.noises)
        n.weight .= weight[cr]
    end
end

function set_weight!(nm, weight::AbstractVector)
    nm.weight .= weight
end


