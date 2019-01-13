# Co-occurence matrix
"""
Basic Co-occurrence Matrix (COOM) type.

# Fields
  * `coomm::SparseMatriCSC{T,Int}` the actual COOM; elements represent
co-occurrences of two terms within a given window
  * `terms::Vector{String}` a list of terms that represent the lexicon of
the document or corpus
  * `column_indices::Dict{String, Int}` a map between the `terms` and the
columns of the co-occurrence matrix
"""
mutable struct CooMatrix{T}
    coom::SparseMatrixCSC{T, Int}
    terms::Vector{String}
    column_indices::Dict{String, Int}
end

"""
    CooMatrix{T}(crps::Corpus [,terms])

Auxiliary constructor(s) of the `CooMatrix` type. The type `T` has to be
a subtype of `Real`. The constructor(s) requires a corpus `crps` and
a `terms` structure representing the lexicon of the corpus. The latter
can be a `Vector{String}`, an `AbstractDict` where the keys are the lexicon or can
be missing, in which case the `lexicon` field of the corpus is used.
"""
function CooMatrix{T}(crps::Corpus, terms::Vector{String}) where T<:Real
    #TODO: Implement
end

CooMatrix(crps::Corpus, terms::Vector{String}) =
    CooMatrix{DEFAULT_DTM_TYPE}(crps, terms)

CooMatrix{T}(crps::Corpus, lex::AbstractDict) where T<:Real =
    CooMatrix{T}(crps, sort(collect(keys(lex))))

CooMatrix(crps::Corpus, lex::AbstractDict) =
    CooMatrix{DEFAULT_DTM_TYPE}(crps, sort(collect(keys(lex))))

CooMatrix{T}(crps::Corpus) where T<:Real = begin
    isempty(lexicon(crps)) && update_lexicon!(crps)
    CooMatrix{T}(crps, lexicon(crps))
end

CooMatrix(crps::Corpus) = begin
    isempty(lexicon(crps)) && update_lexicon!(crps)
    CooMatrix{DEFAULT_DTM_TYPE}(crps, lexicon(crps))
end

CooMatrix(dtm::SparseMatrixCSC{T, Int}, terms::Vector{String}) where T<:Real =
    CooMatrix(dtm, terms, columnindices(terms))


"""
    coom(c::CooMatrix)

Access the co-occurrence matrix field `coom` of a `CooMatrix` `c`.
"""
coom(c::CooMatrix) = c.coom

"""
    coom(crps::Corpus, eltype::Type{T}=DEFAULT_DTM_TYPE)

Access the co-occurrence matrix of the `CooMatrix` associated
with the corpus `crps`. The `CooMatrix{T}` will first have to
be created in order for the actual matrix to be accessed.
"""
coom(crps::Corpus, eltype::Type{T}=DEFAULT_DTM_TYPE) where T<:Real =
    coom(CooMatrix{T}(crps))
