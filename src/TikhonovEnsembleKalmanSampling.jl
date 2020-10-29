module TikhonovEnsembleKalmanSampling

using Distributed
using Statistics
using LinearAlgebra
using Optim

include("sampling.jl")

export teks_update, whteks_update
export teks, whteks

end # module
