

import Base: view


#########################################################
# Normal noise model
#########################################################
mutable struct NormalNoise end

# inverse link

function invlink(::NormalNoise, A::AbstractMatrix)
    return A
end


function loss(::NormalNoise, Z::AbstractMatrix, D::AbstractMatrix)
    diff = (Z .- D)
    return 0.5 .* diff .* diff
end


function ChainRulesCore.rrule(::typeof(loss), nn::NormalNoise, Z, D)

    nanvals = isnan.(D)
    diff = (Z .- D)
    diff[nanvals] .= 0
    loss = diff.*diff

    function loss_normal_pullback(loss_bar)
        Z_bar = loss_bar .* diff
        return ChainRulesCore.NoTangent(), 
               ChainRulesCore.NoTangent(),
               Z_bar, 
               ChainRulesCore.NoTangent()
    end


    return loss, loss_normal_pullback
end

invlinkloss(nn::NormalNoise, Z, D) = sum(loss(nn::NormalNoise, Z, D))

function view(nn::NormalNoise, idx)
    return nn
end


function simulate(nn::NormalNoise, Z, std)
    return Z .+ randn(size(Z)...).*transpose(std)
end


#########################################################
# Bernoulli (logistic) noise model
#########################################################

mutable struct BernoulliNoise end

# inverse link function
sigmoid(x) = 1 ./ (1 .+ exp.(-x))


function invlink(bn::BernoulliNoise, A::AbstractMatrix)
    return sigmoid(A) 
end


function ChainRulesCore.rrule(::typeof(invlink), bn::BernoulliNoise, A)

    Z = invlink(bn, A)

    function invlink_bernoulli_pullback(Z_bar)
        return ChainRulesCore.NoTangent(), 
               ChainRulesCore.NoTangent(),
               Z_bar .* Z .* (1 .- Z)
    end

    return Z, invlink_bernoulli_pullback
end


# Loss function

function loss(bn::BernoulliNoise, Z::AbstractMatrix, D::AbstractMatrix)
    return - D.*log.(Z .+ 1e-8) - (1 .- D).*log.(1 .- Z .+ 1e-8)
end


function ChainRulesCore.rrule(::typeof(loss), bn::BernoulliNoise, Z, D)
    
    nanvals = isnan.(D)
    result = loss(bn,Z,D)
    result[nanvals] .= 0

    function loss_bernoulli_pullback(loss_bar)
        Z_bar = loss_bar.*(-D./Z - (1 .- D) ./ (1 .- Z))
        Z_bar[nanvals] .= 0
        return ChainRulesCore.NoTangent(),
               ChainRulesCore.NoTangent(), 
               Z_bar,
               ChainRulesCore.NoTangent()
    end

    return result, loss_bernoulli_pullback 
end


invlinkloss(bn::BernoulliNoise, Z, D) = sum(loss(bn, invlink(bn, Z), D))


function ChainRulesCore.rrule(::typeof(invlinkloss), bn::BernoulliNoise, Z, D)

    nanvals = isnan.(D) # Handle missing values
    A = invlink(bn, Z)

    function invlinkloss_bernoulli_pullback(loss_bar)
        diff = A .- D
        diff[nanvals] .= 0

        return ChainRulesCore.NoTangent(),
               ChainRulesCore.NoTangent(), 
               loss_bar .* diff,
               ChainRulesCore.NoTangent()
    end
    
    res = loss(bn, A, D)
    res[nanvals] .= 0


    return sum(res), invlinkloss_bernoulli_pullback
end


function view(bn::BernoulliNoise, idx)
    return bn
end


function simulate(bn::BernoulliNoise, Z, p)

    signal = (rand(size(Z)...) .<= Z)
    noise = (rand(size(Z)...) .<= transpose(p))

    return xor.(signal, noise)
end


#########################################################
# Poisson noise model
#########################################################

mutable struct PoissonNoise end


# inverse link function

function invlink(pn::PoissonNoise, A::AbstractMatrix)
    return exp.(A)
end



# Loss function

function loss(pn::PoissonNoise, Z::AbstractMatrix, D::AbstractMatrix)
    return Z .- D.*log.(Z .+ 1e-8)
end


function ChainRulesCore.rrule(::typeof(loss), pn::PoissonNoise, Z, D)
    
    result = loss(pn, Z, D)
    nanvals = isnan.(D)
    result[nanvals] .= 0


    function loss_poisson_pullback(loss_bar)
        Z_bar = loss_bar.*( 1 .- D ./ Z)
        Z_bar[nanvals] .= 0
        return ChainRulesCore.NoTangent(),
               ChainRulesCore.NoTangent(), 
               Z_bar,
               ChainRulesCore.NoTangent()
    end

    return result, loss_poisson_pullback 
end


invlinkloss(pn::PoissonNoise, Z, D) = loss(pn, invlink(pn, Z), D)


function ChainRulesCore.rrule(::typeof(invlinkloss), pn::PoissonNoise, Z, D)

    nanvals = isnan.(D)
    A = invlink(pn, Z)

    function invlinkloss_poisson_pullback(result_bar)
        diff = A .- D
        diff[nanvals] .= 0
        return ChainRulesCore.NoTangent(),
               ChainRulesCore.NoTangent(), 
               result_bar .* diff,
               ChainRulesCore.NoTangent()
    end

    lss = loss(pn, A, D)
    lss[nanvals] .= 0
    result = sum(lss)

    return result, invlinkloss_poisson_pullback
end


function view(pn::PoissonNoise, idx)
    return pn
end


poisson_sample(z) = rand(Poisson(z))

function simulate(pn::PoissonNoise, Z, noise)
    return poisson_sample.(Z .+ transpose(noise))
end

#########################################################
# Ordinal noise model
#########################################################

# Inverse link function

mutable struct OrdinalNoise 
    ext_thresholds::AbstractVector{<:Number}
end

function OrdinalNoise(n_values::Integer)
    thresholds = collect(1:(n_values - 1)) .- 0.5*n_values
    ext_thresholds = [[-Inf]; thresholds; [Inf]]
    return OrdinalNoise(ext_thresholds)
end

@functor OrdinalNoise


function invlink(on::OrdinalNoise, A)
    return A
end


function loss(on::OrdinalNoise, Z::AbstractMatrix, D::AbstractMatrix)

    D_idx = round.(UInt8, D)
    l_thresh = on.ext_thresholds[D_idx]
    r_thresh = on.ext_thresholds[D_idx .+ UInt8(1)]
    loss = -log.(sigmoid(r_thresh .- Z) .- sigmoid(l_thresh .- Z) .+ 1e-8)

    return loss
end

function summary_str(A::AbstractArray, name::String)
    return string(name, ": ", size(A), " ", typeof(A))
end


nanround(x) = isnan(x) ? UInt8(1) : round(UInt8, x)

function ChainRulesCore.rrule(::typeof(loss), on::OrdinalNoise, Z, D)

    nanvals = isnan.(D)
    D_idx = nanround.(D)
    R_idx = D_idx .+ UInt8(1) 

    sort!(on.ext_thresholds)
    l_thresh = on.ext_thresholds[D_idx]
    r_thresh = on.ext_thresholds[R_idx]

    sig_r = sigmoid(r_thresh .- Z)
    sig_l = sigmoid(l_thresh .- Z)
    sig_r[nanvals] .= 1 # These settings ensure that the missing
    sig_l[nanvals] .= 0 # data don't contribute to loss or gradients

    sig_diff = sig_r .- sig_l

    nanvals .= (sig_diff .== 0)
    sig_diff[nanvals] .= 1

    function loss_ordinal_pullback(loss_bar)

        on_r_bar = loss_bar .* -sig_r .* (1 .- sig_r) ./ sig_diff
        on_l_bar = loss_bar .* sig_l .* (1 .- sig_l) ./ sig_diff

        N_bins = length(on.ext_thresholds)-1
        on_bar = zeros(N_bins + 1)
        for i=1:N_bins
            relevant_idx = (D_idx .== i) 
            on_bar[i] += sum(on_l_bar .* relevant_idx)
            on_bar[i+1] += sum(on_r_bar .* relevant_idx)
        end

        T = typeof(on.ext_thresholds) # Move to GPU if necessary

        on_bar = Tangent{OrdinalNoise}(ext_thresholds=T(on_bar))
        Z_bar = loss_bar .* (1 .- sig_r .- sig_l) 

        return ChainRulesCore.NoTangent(), 
               on_bar, 
               Z_bar, 
               ChainRulesCore.NoTangent()
    end
    
    loss = -log.(sig_diff .+ 1e-8)

    return loss, loss_ordinal_pullback
end

# Remember: inverse link == identity function
invlinkloss(on::OrdinalNoise, Z, D) = sum(loss(on, Z, D))


function view(on::OrdinalNoise, idx)
    return on
end


function simulate(on::OrdinalNoise, Z, noise)
   
    Z_noise = Z .+ randn(size(Z)...).*transpose(noise)
    n_bins = length(on.ext_thresholds) - 1

    data = zero(Z)
    for i=1:n_bins
        l_thresh = on.ext_thresholds[i]
        r_thresh = on.ext_thresholds[i+1]
        valid_idx = ((l_thresh .<= Z_noise) .& (Z_noise .<= r_thresh))
        
        data[valid_idx] .= i 
    end

    return data
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

@functor CompositeNoise

trainable(cn::CompositeNoise) = (noises=cn.noises,)


function CompositeNoise(noise_model_ids::Vector{String})
    
    unq_ids = unique(noise_model_ids)
    ranges = ids_to_ranges(noise_model_ids)

    noise_models = map(construct_noise, unq_ids)

    return CompositeNoise(Tuple(ranges), Tuple(noise_models)) 
end


function view(cn::CompositeNoise, idx::UnitRange)

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


function view(cn::CompositeNoise, idx::Colon)
    idx = 1:cn.col_ranges[end].stop
    return view(cn, idx)
end


invlink(cn::CompositeNoise, A) = hcat(map((n,rng)->invlink(n, A[:,rng]), cn.noises, cn.col_ranges)...)
loss(cn::CompositeNoise, Z, D) = hcat(map((n,rng)->loss(n, Z[:,rng], view(D,:,rng)), cn.noises, cn.col_ranges)...)
invlinkloss(cn::CompositeNoise, A, D) = sum(map((n,rng)->invlinkloss(n, A[:,rng], view(D,:,rng)), cn.noises, cn.col_ranges))

function ChainRulesCore.rrule(::typeof(invlinkloss), cn::CompositeNoise, A, D)

    result = 0
    pullbacks = []
    for (noise, rng) in zip(cn.noises, cn.col_ranges)
        lss, pb = Zygote.pullback(invlinkloss, noise, A[:,rng], view(D, :, rng))
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


function simulate(cn::CompositeNoise, Z, noise_param)

    result = zero(Z)

    for (rng, nm) in zip(cn.col_ranges, cn.noises)
        result[:,rng] .= simulate(nm, Z[:,rng], noise_param[rng])
    end

    return result
end

######################################################
# HELPER FUNCTIONS
######################################################


function construct_noise(string_id::String)

    if string_id == "normal"
        noise = NormalNoise()
    elseif string_id == "bernoulli"
        noise = BernoulliNoise()
    elseif string_id == "poisson"
        noise = PoissonNoise()
    elseif startswith(string_id, "ordinal")
        n_values = parse(Int, string_id[8:end])
        noise = OrdinalNoise(n_values)
    else
        throw(ArgumentError, string(string_id, " is not a valid noise model identifier"))
    end

    return noise
end





