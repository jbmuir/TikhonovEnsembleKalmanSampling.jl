function teks(u::Array{T,2}, y, Γ⁻¹, C₀⁻¹, G, n_steps; 
              parallel=false, 
              h₀=1, 
              iδ=100, 
              λ=1, 
              ϵ=convert(T, 1e-10), 
              savechain=false) where T
    if savechain
        uchain = Array{T,2}[]
    end
    hₙchain = T[]
    J = size(u, 2)
    for i = 1:n_steps
        #calculate i-step sample covariance in parameter space
        ū = mean(u, dims=2)
        Cu = Symmetric((u .- ū)*(u .- ū)')/J
        Cuc = cholesky(Cu + ϵ*I) 
        # forward map and differencing operator
        if parallel
            g = reduce(hcat, pmap(G, eachcol(u)))
        else
            g = reduce(hcat, map(G, eachcol(u)))
        end
        ḡ = mean(g, dims=2)
        D = (g .- y)' * (Γ⁻¹ * (g .- ḡ))
        #Regularization/prior, adaptive stepsize, gradient step
        ∇R = Cu*C₀⁻¹
        hₙ = h₀ / (norm(D) + 1/iδ)
        Du = u .- ((hₙ/J)*D*u')'
        #Implicit solve of partial step
        I∇R = cholesky(I + hₙ*∇R)
        u = I∇R \ Du
        # perturb parameters according to i-step sample covariance
        Wu = randn(T, size(u))
        u .+= sqrt(2*hₙ)*Cuc.U*Wu
        if savechain
            push!(uchain, u)
        end
        push!(hₙchain, hₙ)
    end
    if savechain
        return (uchain, hₙchain)
    else
        return (u, hₙchain)
    end
end

function implicit_hyper_solve(θΔθ, Cθ, hₙ, ∇nlpθ_fun; solver=NelderMead())
    θ, Δθ = θΔθ
    # ∇nlpθ_fun = gradient of negative log prior in respect to θ
    res = optimize(θ̂ -> sum((θ̂ .+ hₙ*Cθ*∇nlpθ_fun(θ̂) - Δθ).^2), θ[:], solver)
    return Optim.minimizer(res)
end

function whteks(ξ::Array{T,2}, θ::Array{T,2}, y, Γ⁻¹, GT, ∇nlpθ_fun, n_steps; 
                parallel=false, 
                h₀=1, 
                iδ=100, 
                λ=1, 
                ϵ=convert(T, 1e-10), 
                savechain=false,
                hyper_solver=NelderMead()) where T
    
    if savechain
        ξchain = Array{T,2}[]
        θchain = Array{T,2}[]
    end
    hₙchain = T[]
    J = size(ξ, 2)
    for i = 1:n_steps
        #calculate i-step sample covariance in parameter space & update parameters
        ξ̄ = mean(ξ, dims=2)
        θ̄ = mean(θ, dims=2)
        Cξ = Symmetric((ξ .- ξ̄)*(ξ .- ξ̄)')/J
        Cθ = Symmetric((θ .- θ̄)*(θ .- θ̄)')/J
        Cξc = cholesky(Cξ + ϵ*I)
        Cθc = cholesky(Cθ + ϵ*I) 
        # forward map and differencing operator
        if parallel
            g = reduce(hcat, pmap(GT, zip(eachcol(ξ), eachcol(θ))))
        else
            g = reduce(hcat, map(GT, zip(eachcol(ξ), eachcol(θ))))
        end
        ḡ = mean(g, dims=2)
        D = (g .- y)' * (Γ⁻¹*(g .- ḡ))
        #Regularization / prior and adaptive stepsize, gradient step
        ∇Rξ = Cξ
        hₙ = h₀ / (norm(D) + 1/iδ)
        Dξ = ξ .- ((hₙ/J)*D*ξ')'
        Dθ = θ .- ((hₙ/J)*D*θ')'
        #Implicit solve of partial step
        I∇Rξ = cholesky(I + hₙ*∇Rξ)
        ξ = I∇Rξ \ Dξ
        if parallel
            θ = reduce(hcat, 
                       pmap(θDθ->implicit_hyper_solve(θDθ, Cθ, hₙ, ∇nlpθ_fun, solver=hyper_solver),  
                       zip(eachcol(θ), eachcol(Dθ))))    
        else
            θ = reduce(hcat, 
                       map(θDθ->implicit_hyper_solve(θDθ, Cθ, hₙ, ∇nlpθ_fun, solver=hyper_solver),  
                       zip(eachcol(θ), eachcol(Dθ))))        
        end        
        # perturb parameters according to i-step sample covariance
        Wξ = randn(T, size(ξ))
        Wθ = randn(T, size(θ))
        ξ .+= sqrt(2*hₙ)*Cξc.U*Wξ
        θ .+= sqrt(2*hₙ)*Cθc.U*Wθ
        if savechain
            push!(ξchain, ξ)
            push!(θchain, θ)
        end
        push!(hₙchain, hₙ)
    end
    if savechain 
        return (ξchain, θchain, hₙchain)
    else
        return (ξ, θ, hₙchain)
    end

end