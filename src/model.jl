
import Base: view


mutable struct BatchMatFacModel
    mp::MatProd
    cscale::ColScale
    cshift::ColShift
    bscale::BatchScale
    bshift::BatchShift
    inv_link::ColMap
end

@functor BatchMatFacModel


function (bm::BatchMatFacModel)()
    return bm.inv_link(
                bm.bshift(
                      bm.bscale(
                             bm.cshift(
                                   bm.cscale(
                                         bm.mp()
                                            )
                                      )
                                )
                         )
           )
end


function view(bm::BatchMatFacModel, idx1, idx2)
    return BatchMatFacModel(view(bm.mp, idx1, idx2),
                            view(bm.cscale, idx2),
                            view(bm.cshift, idx2),
                            view(bm.bscale, idx1, idx2),
                            view(bm.bshift, idx1, idx2),
                            view(bm.inv_link, idx2)
                           )

end



