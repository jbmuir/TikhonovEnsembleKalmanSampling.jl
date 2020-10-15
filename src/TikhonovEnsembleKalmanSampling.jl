module TikhonovEnsembleKalmanSampling

using Distributed
using Statistics
using LinearAlgebra
using Optim

include("sampling.jl")

export teks, whteks

end # module
