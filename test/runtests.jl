module TestStringAnalysis

using Test
using Logging
global_logger(ConsoleLogger(stdout, Logging.Error))  # supress test warnings
using SparseArrays
using LinearAlgebra
using StringAnalysis
using Languages
using DataStructures

println("Running tests:")
include("autohash.jl")
include("tokenizer.jl")
include("ngramizer.jl")
include("document.jl")
include("metadata.jl")
include("corpus.jl")
include("preprocessing.jl")
include("dtm.jl")
include("coom.jl")
include("stemmer.jl")
include("stats.jl")
include("lda.jl")
include("lsa.jl")
include("rp.jl")
include("show.jl")

end
