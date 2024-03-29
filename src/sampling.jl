function add_noise!(x, W, C, hₙ, ϵ)
    Cc = cholesky(C + ϵ*I)
    x.+= sqrt(2*hₙ)*Cc.L*W
end

function teks_update(u::Array{T,2}, y, Γ, C₀, G; 
    parallel=false, 
    h₀=1, 
    iδ=100, 
    λ=1, 
    ϵ=convert(T, 1e-10), adapt_h=true) where T

    J = size(u, 2)
    #calculate i-step sample covariance in parameter space
    ū = mean(u, dims=2)
    Cu = Symmetric((u .- ū)*(u .- ū)')/(J-1)
    # forward map and differencing operator
    if parallel
        g = reduce(hcat, pmap(G, eachcol(u)))
    else
        g = reduce(hcat, map(G, eachcol(u)))
    end
    ḡ = mean(g, dims=2)
    D = (g .- y)' * (Γ \ (g .- ḡ))
    #Regularization/prior, adaptive stepsize, gradient step
    ∇R = (C₀\ Cu)' # because both are symmetric you can do this...
    if adapt_h
        hₙ = h₀ / (norm(D) + 1/iδ)
    else
        hₙ = h₀ 
    end
    Du = u .- ((hₙ/J)*D*u')'
    #Implicit solve of partial step
    I∇R = I + hₙ*∇R
    u = I∇R \ Du
    # perturb parameters according to i-step sample covariance
    Wu = randn(T, size(u))
    # u .+= sqrt(2*hₙ)*Cuc.L*Wu
    add_noise!(u, Wu, Cu, hₙ, ϵ)
    return (u, hₙ, g)
end

function teks(u::Array{T,2}, y, Γ, C₀, G, n_steps; 
              parallel=false, 
              h₀=1, 
              iδ=100, 
              λ=1, 
              ϵ=convert(T, 1e-10), 
              savechain=false, adapt_h=true) where T
    if savechain
        uchain = Array{T,2}[]
    end
    hₙchain = T[]
    for i = 1:n_steps
        u, hₙ, _ = teks_update(u, y, Γ, C₀, G; 
                            parallel=parallel, 
                            h₀=h₀, 
                            iδ=iδ, 
                            λ=λ, 
                            ϵ=ϵ, adapt_h)
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

function whteks_update(ξ::Array{T,2}, θ::Array{T,2}, y, Γ, GT, ∇nlpθ_fun; 
                        parallel=false, 
                        h₀=1, 
                        iδ=100, 
                        λ=1, 
                        ϵ=convert(T, 1e-10), 
                        hyper_solver=NelderMead(), adapt_h=true) where T
    J = size(ξ, 2)
    #calculate i-step sample covariance in parameter space & update parameters
    ξ̄ = mean(ξ, dims=2)
    θ̄ = mean(θ, dims=2)
    Cξ = Symmetric((ξ .- ξ̄)*(ξ .- ξ̄)')/(J-1)
    Cθ = Symmetric((θ .- θ̄)*(θ .- θ̄)')/(J-1)
    Cξc = cholesky(Cξ + ϵ*I)
    Cθc = cholesky(Cθ + ϵ*I) 
    # forward map and differencing operator
    if parallel
        g = reduce(hcat, pmap(GT, zip(eachcol(ξ), eachcol(θ))))
    else
        g = reduce(hcat, map(GT, zip(eachcol(ξ), eachcol(θ))))
    end
    ḡ = mean(g, dims=2)
    D = (g .- y)' * (Γ \ (g .- ḡ))
    #Regularization / prior and adaptive stepsize, gradient step
    ∇Rξ = Cξ
    if adapt_h
        hₙ = h₀ / (norm(D) + 1/iδ)
    else
        hₙ = h₀ 
    end
    Dξ = ξ .- ((hₙ/J)*D*ξ')'
    Dθ = θ .- ((hₙ/J)*D*θ')'
    #Implicit solve of partial step
    # I∇Rξ = cholesky(Symmetric(I + hₙ*∇Rξ))
    I∇Rξ = I + hₙ*∇Rξ
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
    ξ .+= sqrt(2*hₙ)*Cξc.L*Wξ
    θ .+= sqrt(2*hₙ)*Cθc.L*Wθ
    return (ξ, θ, hₙ, g)
end

function whteks(ξ::Array{T,2}, θ::Array{T,2}, y, Γ, GT, ∇nlpθ_fun, n_steps; 
                parallel=false, 
                h₀=1, 
                iδ=100, 
                λ=1, 
                ϵ=convert(T, 1e-10), 
                savechain=false,
                hyper_solver=NelderMead(), adapt_h=true) where T
    
    if savechain
        ξchain = Array{T,2}[]
        θchain = Array{T,2}[]
    end
    hₙchain = T[]
    J = size(ξ, 2)
    for i = 1:n_steps
        ξ, θ, hₙ, _ = whteks_update(ξ, θ, y, Γ, GT, ∇nlpθ_fun; 
                                parallel=parallel, 
                                h₀=h₀, 
                                iδ=iδ, 
                                λ=λ, 
                                ϵ=ϵ, 
                                hyper_solver=hyper_solver, adapt_h=adapt_h)
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