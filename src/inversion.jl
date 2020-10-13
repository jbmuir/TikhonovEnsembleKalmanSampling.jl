function teki(u, y₀, Γ, γ, C₀, G; parallel=false, h₀=1, iδ=100, λ=1)
    J = size(u, 2)
    if parallel
        g = reduce(hcat, pmap(G, eachcol(u)))
    else
        g = reduce(hcat, map(G, eachcol(u)))
    end
    y = reduce(hcat, [y₀ .+ γ*randn(length(y₀)) for i = 1:J])
    ḡ = mean(g, dims=2)
    ū = mean(u, dims=2)
    Gū = G(view(ū,:))
    D = (g .- y)' * (Γ \ (g .- ḡ))
    ∇R = λ * (u' * (C₀ \ (u .- ū)))
    E = D .+ ∇R
    hₙ = h₀ /  (norm(E) + 1/iδ)
    return (u  .- (hₙ / J) .* (E * (u .- ū)')', ū, Gū)
end

function whteki(ξ, θ, y₀, Γ, γ, GT, ∇pθ_fun; parallel=false, h₀=1, iδ=100, λ=1)
    # ∇Rθ_fun = gradient of -ve log prior for hyperparameters
    @assert size(ξ, 2) == size(θ, 2)
    J = size(ξ, 2)
    if parallel
        g = reduce(hcat, pmap(GT, zip(eachcol(ξ), eachcol(θ))))
        ∇pθ = reduce(hcat, pmap(∇pθ_fun,  eachcol(θ)))
    else
        g = reduce(hcat, map(GT, zip(eachcol(ξ), eachcol(θ))))
        ∇pθ = reduce(hcat, map(∇pθ_fun,  eachcol(θ)))
    end
    y = reduce(hcat, [y₀ .+ γ*randn(length(y₀)) for i = 1:J])
    ḡ = mean(g, dims=2)
    ξ̄ = mean(ξ, dims=2)
    θ̄ = mean(θ, dims=2)
    Gū = GT((view(ξ̄,:), view(θ̄,:)))
    D = (g .- y)' * (Γ \ (g .- ḡ))
    Rξ = λ * (ξ' * (ξ .- ξ̄))
    Rθ = λ * (∇pθ' * (θ .- θ̄))
    Eξ = D .+ Rξ 
    Eθ = D .+ Rθ
    nE = max(norm(Eξ), norm(Eθ))
    hₙ = h₀ /  (nE + 1/iδ)
    return (ξ  .- (hₙ / J) .* (Eξ * (ξ .- ξ̄)')', 
            θ  .- (hₙ / J) .* (Eθ * (θ .- θ̄)')',
            ξ̄, θ̄, Gū)

end