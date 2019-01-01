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
    Σ = diagm(0 => T.(σ))
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
    Σ = diagm(0 => T.(σ))
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
    get_vector(lm, word [; vector_type=:document])

Returns the vector representation of `word` from the LSA model `lm`
using .
"""
get_vector(lm::LSAModel, word;) = begin
    # TODO(Corneliu) Implement this
end


"""
    cosine(lm, word, n=10)

Return the position of `n` (by default `n = 10`) neighbors of `word` and their
cosine similarities.
"""
function cosine(lm::LSAModel, word, n=10)
    metrics = lm.vectors'*get_vector(lm, word)
    topn_positions = sortperm(metrics[:], rev = true)[1:n]
    topn_metrics = metrics[topn_positions]
    return topn_positions, topn_metrics
end


"""
    similarity(lm, word1, word2)

Return the cosine similarity value between two words `word1` and `word2`.
"""
function similarity(lm::LSAModel, word1, word2)
    return get_vector(lm, word1)'*get_vector(lm, word2)
end


"""
    cosine_similar_words(lm, word, n=10)

Return the top `n` (by default `n = 10`) most similar words to `word`
from the LSAModel `lm`.
"""
function cosine_similar_words(lm::LSAModel, word, n=10)
    indx, metr = cosine(lm, word, n)
    return vocabulary(lm)[indx]
end
