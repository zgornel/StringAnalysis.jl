"""
    LSAModel{S<:AbstractString, T<:AbstractFloat, A<:AbstractMatrix{T}, H<:Integer}

LSA (latent semantic analysis) model. It constructs from a document term matrix (dtm)
a model that can be used to embed documents in a latent semantic space pertaining to
the data. The model requires that the document term matrix be a
`DocumentTermMatrix{T<:AbstractFloat}` because the matrices resulted from the SVD operation
will be forced to contain elements of type `T`.

# Fields
  * `vocab::Vector{S}` a vector with all the words in the corpus
  * `vocab_hash::Dict{S,H}` a word to index in word embeddings matrix mapping
  * `U::A` the document embeddings matrix
  * `Σinv::A` inverse of the singular value matrix
  * `Vᵀ::A` transpose of the word embedding matrix
  * `stats::Symbol` the statistical measure to use for word importances in documents
                    available values are:
                    `:count` (term counts)
                    `:tf` (term frequency)
                    `:tfidf` (default, term frequency - inverse document frequency)
                    `:bm25` (Okapi BM25)
  * `idf::Vector{T}` inverse document frequencies for the words in the vocabulary
  * `nwords::T` averge number of words in a document
  * `κ::Int` the `κ` parameter of the BM25 statistic
  * `β::Float64` the `β` parameter of the BM25 statistic

# `U`, `Σinv` and `Vᵀ`:
  If `X` is a `m`×`n` document-term-matrix with `m` documents and `n` words so that
`X[i,j]` represents a statistical indicator of the importance of term `j` in document `i`
then:
  * `U, Σ, V = svd(X)`
  * `Σinv = inv(Σ)`
  * `Vᵀ = V'`

# Examples
```
julia> using StringAnalysis

       doc1 = StringDocument("This is a text about an apple. There are many texts about apples.")
       doc2 = StringDocument("Pears and apples are good but not exotic. An apple a day keeps the doctor away.")
       doc3 = StringDocument("Fruits are good for you.")
       doc4 = StringDocument("This phrase has nothing to do with the others..")
       doc5 = StringDocument("Simple text, little fruit inside")

       crps = Corpus(AbstractDocument[doc1, doc2, doc3, doc4, doc5])
       prepare!(crps, strip_punctuation)
       update_lexicon!(crps)
       dtm = DocumentTermMatrix{Float32}(crps, sort(collect(keys(crps.lexicon))))

       ### Build LSA Model ###
       lsa_model = LSAModel(dtm, k=3, stats=:count)

       query = StringDocument("Apples and an exotic fruit.")
       idxs, corrs = cosine(lsa_model, query)

       println("Query: \"\$(query.text)\"")
       for (idx, corr) in zip(idxs, corrs)
           println("\$corr -> \"\$(crps[idx].text)\"")
       end

Query: "Apples and an exotic fruit."
0.10621497 -> "Pears and apples are good but not exotic  An apple a day keeps the doctor away "
0.044571217 -> "This is a text about an apple  There are many texts about apples "
0.020167388 -> "Fruits are good for you "
0.001160352 -> "Simple text  little fruit inside"
-0.016810883 -> "This phrase has nothing to do with the others "
```

# References:
  * [The LSA wiki page](https://en.wikipedia.org/wiki/Latent_semantic_analysis)
  * [Deerwester et al. 1990](http://lsa.colorado.edu/papers/JASIS.lsi.90.pdf)

"""
mutable struct LSAModel{S<:AbstractString, T<:AbstractFloat, A<:AbstractMatrix{T}, H<:Integer}
    vocab::Vector{S}        # vocabulary
    vocab_hash::Dict{S,H}   # term to column index in V
    U::A                    # document vectors
    Σinv::A                 # inverse of Σ
    Vᵀ::A                   # word vectors (transpose of V)
    stats::Symbol           # term/document importance
    idf::Vector{T}          # inverse document frequencies
    nwords::T               # average words/document in corpus
    κ::Int                  # κ parameter for Okapi BM25 (used if stats==:bm25)
    β::Float64              # β parameter for Okapi BM25 (used if stats==:bm25)
end

function LSAModel(dtm::DocumentTermMatrix{T}; kwargs...) where T<:Integer
    throw(ErrorException(
        """A LSA model requires a that the document term matrix
        be a DocumentTermMatrix{<:AbstractFloat}!"""))
end

function LSAModel(dtm::DocumentTermMatrix{T};
                  k::Int=size(dtm.dtm, 1),
                  stats::Symbol=:tfidf,
                  κ::Int=2,
                  β::Float64=0.75
                 ) where T<:AbstractFloat
    n, p = size(dtm.dtm)
    # Checks
    length(dtm.terms) == p ||
        throw(DimensionMismatch("Dimensions inside dtm are inconsistent."))
    k > n &&
        @warn "k can be at most $n; using k=$n"
    if !(stats in [:count, :tf, :tfidf, :bm25])
        @warn "stats has to be either :tf, :tfidf or :bm25; defaulting to :tfidf"
        stats = :tfidf
    end
    # Calculate inverse document frequency, mean document size
    documents_containing_term = vec(sum(dtm.dtm .> 0, dims=1)) .+ one(T)
    idf = log.(n ./ documents_containing_term) .+ one(T)
    nwords = mean(sum(dtm.dtm, dims=2))
    # Get X
    if stats == :count
        X = dtm.dtm
    elseif stats == :tf
        X = tf(dtm.dtm)
    elseif stats == :tfidf
        X = tf_idf(dtm.dtm)
    elseif stats == :bm25
        X = bm_25(dtm.dtm, κ=κ, β=β)
    end
    # Get the model
    U, Σ, V = svd(Matrix(X))
    Σinv = diagm(0 => T.(1 ./ Σ[1:k]))
    U = U[:,1:k]  # trim components
    U = T.(U ./ sqrt.(sum(U.^2, dims=2)))  # Normalize + type convert
    # Return the model
    return LSAModel(dtm.terms, dtm.column_indices,
                    T.(U[:,1:k]), Σinv, T.(V[:,1:k]'),
                    stats, idf, nwords, κ, β)
end


function Base.show(io::IO, lm::LSAModel{S,T,A,H}) where {S,T,A,H}
    num_docs, len_vecs = size(lm.U)
    num_terms = length(lm.vocab)
    print(io, "LSA Model ($(lm.stats)) $(num_docs) documents, " *
          "$(num_terms) terms, $(len_vecs)-element $(T) vectors")
end


"""
    lsa(X [;k=3, stats=:tfidf, κ=2, β=0.75])

Constructs an LSA model. The input `X` can be a `Corpus` or a `DocumentTermMatrix`.
Use `?LSAModel` for more details.
"""
function lsa(dtm::DocumentTermMatrix{T};
             k::Int=size(dtm.dtm, 1),
             stats::Symbol=:tfidf,
             κ::Int=2,
             β::Float64=0.75) where T<:AbstractFloat
    LSAModel(dtm, k=k, stats=stats, κ=κ, β=β)
end

function lsa(crps::Corpus;
             k::Int=length(crps),
             stats::Symbol=:tfidf,
             κ::Int=2,
             β::Float64=0.75) where T<:AbstractFloat
    if isempty(crps.lexicon)
        update_lexicon!(crps)
    end
    lsa(DocumentTermMatrix{Float32}(crps, lexicon(crps)), k=k, stats=stats, κ=κ, β=β)
end


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
size(lm::LSAModel) = length(lm.vocab), size(lm.U,1), size(lm.Σinv,1)


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
    default = zeros(T, size(lm.Σinv,1))
    idx = get(lm.vocab_hash, word, 0)
    if idx == 0
        return default
    else
        return lm.Vᵀ[:, idx]
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
    words_in_document = sum(dtv)
    # Calculate document vector
    if lm.stats == :count
        v = dtv
    elseif lm.stats == :tf
        tf = sqrt.(dtv ./ max(words_in_document, one(T)))
        v = tf
    elseif lm.stats == :tfidf
        tf = sqrt.(dtv ./ max(words_in_document, one(T)))
        v = tf .* lm.idf
    elseif lm.stats == :bm25
        k = T(lm.κ)
        b = T(lm.β)
        oneval = one(T)
        tf = sqrt.(dtv ./ max(words_in_document, oneval))
        v = lm.idf .* ((k + 1) .* tf) ./
                       (k * (oneval - b + b * sum(dtv)/lm.nwords) .+ tf)
    end
    # Embed
    d̂ = lm.Σinv * lm.Vᵀ * v  # d̂ⱼ= Σ⁻¹⋅Vᵀ⋅dⱼ
    return d̂ ./ norm(d̂,2)
end


"""
    embed_word(lm, word)

Return the vector representation of `word` using the LSA model `lm`.
"""
function embed_word(lm::LSAModel, word)
    throw(ErrorException(
        """Word embedding is not supported as it would require storing
        all documents in the model in order to determine the counts
        of `word` across the corpus."""))
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
