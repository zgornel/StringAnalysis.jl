module TestStringAnalysis

using SparseArrays
using Test
using Languages
using StringAnalysis

println("Running tests:")
include("tokenizer.jl")
include("ngramizer.jl")
include("document.jl")
include("metadata.jl")
include("corpus.jl")
include("preprocessing.jl")
include("dtm.jl")
include("stemmer.jl")
include("stats.jl")
include("lda.jl")

end
