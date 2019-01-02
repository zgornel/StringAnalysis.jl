# LSA
mutable struct LSAModel{S<:AbstractString, T<:Real, A<:AbstractMatrix{T}, H<:Integer}
    vocab::Vector{S}
    vocab_hash::Dict{S,H}
    U::A
    Σ::A
    V::A
end

function LSAModel(vocab::AbstractArray{S,1},
                  X::AbstractArray{T,2};
                  k::Int=size(X, 1)
                 ) where {S<:AbstractString, T<:Real}
    length(vocab) == size(X, 2) ||
        throw(DimensionMismatch("Dimension of vocab and dtm are inconsistent."))
    k > size(X, 1) &&
        @warn "k can be at most $(size(X, 1)); using k=$(size(X, 1))"
    vocab_hash = Dict{S, Int}()
    for (i, word) in enumerate(vocab)
        vocab_hash[word] = i
    end
    U, σ, V = svd(X)
    Σ = diagm(0 => T.(σ[1:k]))
    LSAModel(vocab, vocab_hash, T.(U[:,1:k]), Σ, T.(V[:,1:k]))
end

function LSAModel(dtm::DocumentTermMatrix{T};
                  k::Int=size(dtm.dtm, 1)
                 ) where T<:Real
    length(dtm.terms) == size(dtm.dtm, 2) ||
        throw(DimensionMismatch("Dimensions inside dtm are inconsistent."))
    k > size(dtm.dtm, 1) &&
        @warn "k can be at most $(size(dtm.dtm, 1)); using k=$(size(dtm.dtm, 1))"
    U, σ, V = svd(Matrix(dtm.dtm))
    Σ = diagm(0 => T.(σ[1:k]))
    LSAModel(dtm.terms, dtm.column_indices, T.(U[:,1:k]), Σ, T.(V[:,1:k]))
end


function Base.show(io::IO, lm::LSAModel{S,T,A,H}) where {S,T,A,H}
    num_docs, len_vecs = size(lm.U)
    num_terms = length(lm.vocab)
    print(io, "LSA Model $(num_docs) documents, $(num_terms) terms, " *
          "$(len_vecs)-element $(T) vectors")
end


"""
    lsa(X [;k=3])

Perform [Latent Semantic Analysis](https://en.wikipedia.org/wiki/Latent_semantic_analysis).
The input `X` can be a `Corpus`, `DocumentTermMatrix` or `AbstractArray`.
"""
lsa(X::DocumentTermMatrix; k::Int=3) = LSAModel(X, k=k)

lsa(crps::Corpus; k::Int=3) = lsa(DocumentTermMatrix{Float32}(crps), k=k)


"""
    vocabulary(lm)

Return the vocabulary as a vector of words of the LSA model `lm`.
"""
vocabulary(lm::LSAModel) = lm.vocab


"""
    in_vocabulary(lm, word)

Return `true` if `word` is part of the vocabulary of the LSA model `lm` and
`false` otherwise.
"""
in_vocabulary(lm::LSAModel, word::AbstractString) = word in lm.vocab


"""
    size(lm)

Return a tuple containing the number of terms, the number of documents and
the vector representation dimensionality of the LSA model `lm`.
"""
size(lm::LSAModel) = length(lm.vocab), size(U,1), size(Σ,1)


"""
    index(lm, word)

Return the index of `word` from the LSA model `lm`.
"""
index(lm::LSAModel, word) = lm.vocab_hash[word]


"""
    get_vector(lm, word)

Returns the vector representation of `word` from the LSA model `lm`.
"""
function get_vector(lm::LSAModel{S,T,A,H}, word) where {S,T,A,H}
    default = zeros(T, size(lm.Σ,1))
    idx = get(lm.vocab_hash, word, 0)
    if idx == 0
        return default
    else
        return lm.V[idx,:]
    end
end


"""
    embed_document(lm, doc)

Return the vector representation of a document `doc` using the LSA model `lm`.
"""
embed_document(lm::LSAModel{S,T,A,H}, doc::AbstractDocument) where {S,T,A,H} =
    # Hijack vocabulary hash to use as lexicon (only the keys needed)
    embed_document(lm, dtv(doc, lm.vocab_hash, T))

embed_document(lm::LSAModel{S,T,A,H}, doc::AbstractString) where {S,T,A,H} =
    embed_document(lm, NGramDocument{S}(doc))

# Actual embedding function: takes as input the LSA model `lm` and a document
# term vector `dtv`. Returns the representation of `dtv` in the embedding space.
function embed_document(lm::LSAModel{S,T,A,H}, dtv::Vector{T}) where {S,T,A,H}
    # d̂ⱼ= Σ⁻¹⋅Vᵀ⋅dⱼ
    d̂ = inv(lm.Σ) * lm.V' * dtv
    return d̂
end


"""
    embed_word(lm, word)

Return the vector representation of `word` using the LSA model `lm`.
"""
function embed_word(lm::LSAModel, word)
    @error """Word embedding is not supported as it would require storing
              all documents in the model in order to determine the counts
              of `word` across the corpus."""
end


"""
    cosine(lm, doc, n=10)

Return the position of `n` (by default `n = 10`) neighbors of document `doc`
and their cosine similarities.
"""
function cosine(lm::LSAModel, doc, n=10)
    metrics = lm.U * embed_document(lm, doc)
    n = min(n, length(metrics))
    topn_positions = sortperm(metrics[:], rev = true)[1:n]
    topn_metrics = metrics[topn_positions]
    return topn_positions, topn_metrics
end


"""
    similarity(lm, doc1, doc2)

Return the cosine similarity value between two documents `doc1` and `doc2`.
"""
function similarity(lm::LSAModel, doc1, doc2)
    return embed_document(lm, doc1)' * embed_document(lm, doc2)
end
