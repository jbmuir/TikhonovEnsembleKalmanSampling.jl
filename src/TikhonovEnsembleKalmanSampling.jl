module TikhonovEnsembleKalmanSampling

using Distributed
using Statistics
using LinearAlgebra

include("inversion.jl")
include("sampling.jl")

export teki, whteki

end # module
