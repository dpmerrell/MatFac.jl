
function forward(model::BatchMatFacModel)

    A = (transpose(model.X)*model.Y).* transpose(model.sigma) .+ transpose(model.mu)

    A = bq_add( bq_mult(A, model.delta), model.theta)

    A = model.feature_link_map(A)

    return A
end
