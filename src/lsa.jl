# LSA
mutable struct LSAModel{S<:AbstractString, T<:Real, A<:AbstractMatrix{T}, H<:Integer}
    vocab::Vector{S}
    vocab_hash::Dict{S,H}
    U::A
    Σ::A
    V::A
end

function LSAModel(vocab::AbstractArray{S,1},
                  A::AbstractArray{T,2};
                  k::Int=size(A, 1)
                 ) where {S<:AbstractString, T<:Real}
    length(vocab) == size(A, 2) ||
        throw(DimensionMismatch("Dimension of vocab and dtm are inconsistent."))
    k > size(A, 1) &&
        throw(DimensionMismatch("k has to be at most $(size(A,1))"))
    vocab_hash = Dict{S, Int}()
    for (i, word) in enumerate(vocab)
        vocab_hash[word] = i
    end
    U, σ, V = svd(A)
    Σ = diagm(0 => T.(σ))
    LSAModel(vocab, vocab_hash, T.(U[:,1:k]), Σ, T.(V[:,1:k]))
end

function LSAModel(dtm::DocumentTermMatrix{T};
                  k::Int=size(dtm.dtm, 1)
                 ) where T<:Real
    length(dtm.terms) == size(dtm.dtm, 2) ||
        throw(DimensionMismatch("Dimensions inside dtm are inconsistent."))
    k > size(dtm.dtm, 1) &&
        throw(DimensionMismatch("k has to be at most $(size(dtm.dtm,1))"))
    U, σ, V = svd(Matrix(dtm.dtm))
    Σ = diagm(0 => T.(σ))
    LSAModel(dtm.terms, dtm.column_indices, T.(U[:,1:k]), Σ, T.(V[:,1:k]))
end


"""
Perform [Latent Semantic Analysis](https://en.wikipedia.org/wiki/Latent_semantic_analysis).
"""
lsa(dtm::DocumentTermMatrix) = svd(Matrix(tf_idf(dtm)))
lsa(crps::Corpus) = svd(Matrix(tf_idf(DocumentTermMatrix(crps))))
