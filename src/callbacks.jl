
"""
    HistoryCallback

    A simple callback that records a history of the loss.
    Serves as an example/template callback.
    For the purposes of MatFac.jl, a "callback" must be callable
    and receive the following arguments:

    `model`: a MatFacModel struct
    `iter`: the just-completed epoch
    `data_loss`: the model's total loss on the data
    `X_reg_loss`: the model's X-regularizer loss
    `Y_reg_loss`: the model's Y-regularizer loss
    `row_layer_reg_loss`: the model's row layer regularizer loss
    `col_layer_reg_loss`: the model's col layer regularizer loss

    The callback is called at the end of each epoch.
    It may be a function or a callable struct.
    It may _mutate_ the model.
"""
mutable struct HistoryCallback
    history::Vector{<:Dict}
end

function HistoryCallback()
    return HistoryCallback(Dict[])
end

function (hcb::HistoryCallback)(model::MatFacModel, iter::Number, data_loss::Number,
                                X_reg_loss::Number, Y_reg_loss::Number, 
                                row_layer_reg_loss::Number, col_layer_reg_loss::Number)
    push!(hcb.history, Dict("data_loss" => data_loss,
                            "X_reg_loss" => X_reg_loss,
                            "Y_reg_loss" => Y_reg_loss,
                            "row_layer_reg_loss" => row_layer_reg_loss,
                            "col_layer_reg_loss" => col_layer_reg_loss))

end


