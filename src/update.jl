

#################################################
# Extensions to Flux's update! function
#################################################


import Flux.Optimise: update!


function update!(opt, params::Tuple, grads::Tuple)
    for (p, g) in zip(params, grads)
        if g != nothing
            update!(opt, p, g)
        end
    end
end


function update!(opt, params::Any, grads::NamedTuple)
    for pname in propertynames(grads)
        g = getproperty(grads,pname)
        if g != nothing
            update!(opt, getproperty(params, pname), g) 
        end
    end
end


#function update!(opt, params, grads::Vector{Vector})
#    for (p,g) in zip(params, grads)
#        if g != nothing
#            update!(opt, p, g)
#        end
#    end
#end



