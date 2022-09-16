

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

# inverse link

function invlink(::NormalNoise, A::AbstractMatrix)
    return A
end


function loss(nn::NormalNoise, Z::AbstractMatrix, D::AbstractMatrix)
    diff = (Z .- D)
    diff .*= diff
    diff .*= 0.5
    return diff .* transpose(nn.weight)
end


function ChainRulesCore.rrule(::typeof(loss), nn::NormalNoise, Z, D)

    nanvals = isnan.(D)
    diff = (Z .- D)
    tozero!(diff, nanvals)
    loss = diff.*diff
    loss .*= 0.5
    loss .*= transpose(nn.weight)

    function loss_normal_pullback(loss_bar)
        Z_bar = loss_bar .* diff .* transpose(nn.weight)
        return ChainRulesCore.NoTangent(), 
               ChainRulesCore.NoTangent(),
               Z_bar, 
               ChainRulesCore.NoTangent()
    end


    return loss, loss_normal_pullback
end

invlinkloss(nn::NormalNoise, Z, D) = sum(loss(nn::NormalNoise, Z, D))

function view(nn::NormalNoise, idx)
    return NormalNoise(view(nn.weight, idx))
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

# inverse link function
function sigmoid(x)
    y = -x
    y .= exp.(y)
    y .+= 1
    y .= (1 ./ y)
    return y
end

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

function loss(bn::BernoulliNoise, Z::AbstractMatrix, D::AbstractMatrix)
    Z2 = (1 .- D)
    Z2 .*= log.(1 .- Z .+ 1e-8)
    
    Z3 = D.*log.(Z .+ 1e-8)

    Z2 .+= Z3
    Z2 .*= -1
    Z2 .*= transpose(bn.weight)
    return Z2
end


function ChainRulesCore.rrule(::typeof(loss), bn::BernoulliNoise, Z, D)
    
    result = loss(bn,Z,D)
    nanvals = isnan.(result)
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


invlinkloss(bn::BernoulliNoise, Z, D) = sum(loss(bn, invlink(bn, Z), D))


function ChainRulesCore.rrule(::typeof(invlinkloss), bn::BernoulliNoise, Z, D)

    nanvals = isnan.(D) # Handle missing values
    A = sigmoid(Z)
    diff = A .- D
    diff .*= transpose(bn.weight)
    tozero!(diff, nanvals)

    function invlinkloss_bernoulli_pullback(loss_bar)

        return ChainRulesCore.NoTangent(),
               ChainRulesCore.NoTangent(), 
               loss_bar .* diff,
               ChainRulesCore.NoTangent()
    end
    
    A .= loss(bn, A, D)
    tozero!(A, nanvals)

    return sum(A), invlinkloss_bernoulli_pullback
end


function view(bn::BernoulliNoise, idx)
    return BernoulliNoise(view(bn.weight, idx))
end


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

# inverse link function

function invlink(pn::PoissonNoise, A::AbstractMatrix)
    return exp.(A)
end



# Loss function

function loss(pn::PoissonNoise, Z::AbstractMatrix, D::AbstractMatrix)
    return (Z .- D.*log.(Z .+ 1e-8)) .* transpose(pn.weight)
end


function ChainRulesCore.rrule(::typeof(loss), pn::PoissonNoise, Z, D)
    
    result = loss(pn, Z, D)
    nanvals = isnan.(D)
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


invlinkloss(pn::PoissonNoise, Z, D) = loss(pn, invlink(pn, Z), D)


function ChainRulesCore.rrule(::typeof(invlinkloss), pn::PoissonNoise, Z, D)

    nanvals = isnan.(D)
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

    A .= loss(pn, A, D)
    tozero!(A, nanvals)
    result = sum(A)

    return result, invlinkloss_poisson_pullback
end


function view(pn::PoissonNoise, idx)
    return PoissonNoise(view(pn.weight, idx))
end


poisson_sample(z) = rand(Poisson(z))


#########################################################
# Ordinal noise model
#########################################################

# Inverse link function

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

function invlink(on::OrdinalNoise, A)
    return A
end


function loss(on::OrdinalNoise, Z::AbstractMatrix, D::AbstractMatrix)

    D_idx = round.(UInt8, D)
    l_thresh = on.ext_thresholds[D_idx]
    r_thresh = on.ext_thresholds[D_idx .+ UInt8(1)]
    loss = -log.(sigmoid(r_thresh .- Z) .- sigmoid(l_thresh .- Z) .+ 1e-8)
    loss .*= transpose(on.weight)

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
    toone!(sig_r, nanvals)  # These settings ensure that the missing
    tozero!(sig_l, nanvals) # data don't contribute to loss or gradients

    sig_diff = sig_r .- sig_l

    nanvals .= (sig_diff .== 0)
    toone!(sig_diff, nanvals) 

    function loss_ordinal_pullback(loss_bar)

        on_r_bar = loss_bar .* -sig_r .* (1 .- sig_r) ./ sig_diff
        on_r_bar .*= transpose(on.weight)
        on_l_bar = loss_bar .* sig_l .* (1 .- sig_l) ./ sig_diff
        on_l_bar .*= transpose(on.weight)

        N_bins = length(on.ext_thresholds)-1
        on_bar = zeros(N_bins + 1)
        relevant_idx = similar(on_r_bar, Bool)
        for i=1:N_bins
            relevant_idx .= (D_idx .== i) 
            on_bar[i] += sum(on_l_bar .* relevant_idx)
            on_bar[i+1] += sum(on_r_bar .* relevant_idx)
        end
        relevant_idx = nothing

        T = typeof(on.ext_thresholds) # Move to GPU if necessary

        on_bar = Tangent{OrdinalNoise}(ext_thresholds=T(on_bar))
        Z_bar = loss_bar .* (1 .- sig_r .- sig_l) 
        Z_bar .*= transpose(on.weight)

        return ChainRulesCore.NoTangent(), 
               on_bar, 
               Z_bar, 
               ChainRulesCore.NoTangent()
    end
    
    loss = -log.(sig_diff .+ 1e-8)
    loss .*= transpose(on.weight)

    return loss, loss_ordinal_pullback
end

# Remember: inverse link == identity function
invlinkloss(on::OrdinalNoise, Z, D) = sum(loss(on, Z, D))


function view(on::OrdinalNoise, idx)
    return OrdinalNoise(view(on.weight, idx), on.ext_thresholds)
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


invlink(cn::CompositeNoise, A) = hcat(map((n,rng)->invlink(n, view(A,:,rng)), cn.noises, cn.col_ranges)...)
loss(cn::CompositeNoise, Z, D) = hcat(map((n,rng)->loss(n, view(Z,:,rng), view(D,:,rng)), cn.noises, cn.col_ranges)...)
invlinkloss(cn::CompositeNoise, A, D) = sum(map((n,rng)->invlinkloss(n, view(A,:,rng), view(D,:,rng)), cn.noises, cn.col_ranges))

function ChainRulesCore.rrule(::typeof(invlinkloss), cn::CompositeNoise, A, D)

    result = 0
    pullbacks = []
    for (noise, rng) in zip(cn.noises, cn.col_ranges)
        lss, pb = Zygote.pullback(invlinkloss, noise, view(A,:,rng), view(D, :, rng))
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
    elseif string_id == "bernoulli"
        noise = BernoulliNoise(weight)
    elseif string_id == "poisson"
        noise = PoissonNoise(weight)
    elseif startswith(string_id, "ordinal")
        n_values = parse(Int, string_id[8:end])
        noise = OrdinalNoise(weight, n_values)
    else
        throw(ArgumentError, string(string_id, " is not a valid noise model identifier"))
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


