"""
    RPModel{S<:AbstractString, T<:AbstractFloat, A<:AbstractMatrix{T}, H<:Integer}

Random projection model. It constructs from a document term matrix (DTM)
a model that can be used to embed documents in a random sub-space. The model requires
that the document term matrix be a `DocumentTermMatrix{T<:AbstractFloat}` because
the elements of the matrices resulted projection operation are floating point
numbers and these have to match or be convertible to type `T`. The approach is
based on the effects of the
[Johnson-Lindenstrauss lemma](https://en.wikipedia.org/wiki/Johnson%E2%80%93Lindenstrauss_lemma).

# Fields
  * `vocab::Vector{S}` a vector with all the words in the corpus
  * `vocab_hash::OrderedDict{S,H}` a word to index in the random projection maatrix mapping
  * `R::A` the random projection matrix
  * `stats::Symbol` the statistical measure to use for word importances in documents. Available values are: `:count` (term count), `:tf` (term frequency), `:tfidf` (default, term frequency-inverse document frequency) and `:bm25` (Okapi BM25)
  * `idf::Vector{T}` inverse document frequencies for the words in the vocabulary
  * `nwords::T` averge number of words in a document
  * `ngram_complexity::Int` ngram complexity
  * `κ::Int` the `κ` parameter of the BM25 statistic
  * `β::Float64` the `β` parameter of the BM25 statistic
  * `project::Bool` specifies whether the model actually performs the projection or not; it is false if the number of dimensions provided is zero or negative

# References:
  * [Kaski 1998](http://www.academia.edu/371863/Dimensionality_Reduction_by_Random_Mapping_Fast_Similarity_Computation_for_Clustering)
  * [Achlioptas 2001](https://users.soe.ucsc.edu/~optas/papers/jl.pdf)
  * [Li et al. 2006](http://web.stanford.edu/~hastie/Papers/Ping/KDD06_rp.pdf)
"""
struct RPModel{S<:AbstractString, T<:AbstractFloat, A<:AbstractMatrix{T}, H<:Integer}
    vocab::Vector{S}        # vocabulary
    vocab_hash::OrderedDict{S,H}   # term to column index in V
    R::A                    # projection matrix
    stats::Symbol           # term/document importance
    idf::Vector{T}          # inverse document frequencies
    nwords::T               # average words/document in corpus
    ngram_complexity::Int   # ngram complexity
    κ::Int                  # κ parameter for Okapi BM25 (used if stats==:bm25)
    β::Float64              # β parameter for Okapi BM25 (used if stats==:bm25)
    project::Bool
end

function RPModel(dtm::DocumentTermMatrix{T};
                 k::Int=size(dtm.dtm, 1),
				 density::Float64=1/sqrt(k),
                 stats::Symbol=:tfidf,
                 ngram_complexity::Int=DEFAULT_NGRAM_COMPLEXITY,
                 κ::Int=BM25_KAPPA,
                 β::Float64=BM25_BETA
                ) where T<:AbstractFloat
    m, n = size(dtm.dtm)
    zeroval = zero(T)
    # Checks
    length(dtm.terms) != m &&
        throw(DimensionMismatch("Dimensions inside dtm are inconsistent."))
    if !(stats in [:count, :tf, :tfidf, :bm25])
        @warn "stats has to be either :tf, :tfidf or :bm25; defaulting to :tfidf"
        stats = :tfidf
    end
    # Calculate inverse document frequency, mean document size
    documents_containing_term = vec(sum(dtm.dtm .> 0, dims=2)) .+ one(T)
    idf = log.(n ./ documents_containing_term) .+ one(T)
    nwords = mean(sum(dtm.dtm, dims=1))
    R = random_projection_matrix(k, m, T, density)
    project = ifelse(k > 0, true, false)
    # Return the model
    return RPModel(dtm.terms, dtm.row_indices, R, stats, idf,
                   nwords, ngram_complexity, κ, β, project)
end

function RPModel(dtm::DocumentTermMatrix{T}; kwargs...) where T<:Integer
    throw(ErrorException(
        """A random projection model requires a that the document term matrix
        be a DocumentTermMatrix{<:AbstractFloat}!"""))
end


"""
    random_projection_matrix(k::Int, m::Int, eltype::Type{T<:AbstractFloat}, density::Float64)

Builds a `k`×`m` sparse random projection matrix with elements of type `T` and
a non-zero element frequency of `density`. `k` and `m` are the output and input
dimensionalities.

# Matrix Probabilities
If we note `s = 1 / density`, the components of the random matrix are drawn from:
- `-sqrt(s) / sqrt(k)` with probability `1/2s`
- `0` with probability `1 - 1/s`
- `+sqrt(s) / sqrt(k)`   with probability `1/2s`

# No projection hack
If `k<=0` no projection is performed and the function returns an identity matrix
sized `m`×`m` with elements of type `T`. This is useful if one does not want to
embed documents but rather calculate term frequencies, BM25 and other statistical
indicators (similar to `dtv`).
"""
function random_projection_matrix(k::Int, m::Int, eltype::Type{T}, density::Float64
                                 ) where T<:AbstractFloat
    if k <= 0
        # No projection, return the identity matrix
        R = spdiagm(0 => ones(T, m))
        return R
    else
        R = zeros(T, k, m)
        s = 1/density
        is_pos = 0.0
        is_neg = 0.0
        v = sqrt(s/k)
        pmin = 1/(2*s)
        @inbounds for j in 1:m
            for i in 1:k
                p = rand()
                if p < pmin
                    R[i,j] = v
                elseif p > 1-pmin
                    R[i,j] = -v
                end
            end
        end
        return sparse(R)
    end
end


function Base.show(io::IO, rpm::RPModel{S,T,A,H}) where {S,T,A,H}
    len_vecs, num_terms = size(rpm.R)
    str_proj = ifelse(rpm.project, "Random Projection model",
                      "Identity Projection")
    print(io, "$str_proj ($(rpm.stats)), " *
          "$(num_terms) terms, dimensionality $(len_vecs), $(T) vectors")
end


"""
    rp(X [;k=m, density=1/sqrt(k), stats=:tfidf, ngram_complexity=DEFAULT_NGRAM_COMPLEXITY, κ=2, β=0.75])

Constructs a random projection model. The input `X` can be a `Corpus` or a `DocumentTermMatrix`
with `m` words in the lexicon. The model does not store the corpus or DTM document embeddings,
just the projection matrix. Use `?RPModel` for more details.
"""
function rp(dtm::DocumentTermMatrix{T};
            k::Int=size(dtm.dtm, 1),
            density::Float64=1/sqrt(k),
            stats::Symbol=:tfidf,
            ngram_complexity::Int=DEFAULT_NGRAM_COMPLEXITY,
            κ::Int=BM25_KAPPA,
            β::Float64=BM25_BETA
           ) where T<:AbstractFloat
    RPModel(dtm, k=k, density=density, stats=stats,
            ngram_complexity=ngram_complexity, κ=κ, β=β)
end

function rp(crps::Corpus,
            ::Type{T} = DEFAULT_FLOAT_TYPE;
            k::Int=length(crps.lexicon),
            density::Float64=1/sqrt(k),
            stats::Symbol=:tfidf,
            ngram_complexity::Int=DEFAULT_NGRAM_COMPLEXITY,
            κ::Int=BM25_KAPPA,
            β::Float64=BM25_BETA
           ) where T<:AbstractFloat
    lex = create_lexicon(crps, ngram_complexity)
    rp(DocumentTermMatrix{T}(crps, lex, ngram_complexity=ngram_complexity),
       k=k, density=density, stats=stats, ngram_complexity=ngram_complexity, κ=κ, β=β)
end


"""
    vocabulary(rpm)

Return the vocabulary as a vector of words of the random projection model `rpm`.
"""
vocabulary(rpm::RPModel) = rpm.vocab


"""
    in_vocabulary(rpm, word)

Return `true` if `word` is part of the vocabulary of the random projection
model `rpm` and `false` otherwise.
"""
in_vocabulary(rpm::RPModel, word::AbstractString) = word in rpm.vocab


"""
    size(rpm)

Return a tuple containing the input data and projection sub-space
dimensionalities of the random projection model `rpm`.
"""
size(rpm::RPModel) = size(rpm.R, 2), size(rpm.R, 1)


"""
    index(rpm, word)

Return the index of `word` from the random projection model `rpm`.
"""
index(rpm::RPModel, word) = rpm.vocab_hash[word]


"""
    get_vector(rpm, word)

Returns the random projection vector corresponding to `word` in the
random projection model `rpm`.
"""
function get_vector(rpm::RPModel{S,T,A,H}, word) where {S,T,A,H}
    default = zeros(T, size(rpm.R, 1))
    idx = get(rpm.vocab_hash, word, 0)
    if idx == 0
        return default
    else
        return rpm.R[:, idx]
    end
end


"""
    embed_document(rpm, doc)

Return the vector representation of `doc`, obtained using the
random projection model `rpm`. `doc` can be an `AbstractDocument`,
`Corpus` or DTV or DTM.
"""
embed_document(rpm::RPModel{S,T,A,H}, doc::AbstractDocument) where {S,T,A,H} =
    # Hijack vocabulary hash to use as lexicon (only the keys needed)
    embed_document(rpm, dtv(doc, rpm.vocab_hash, T,
                            ngram_complexity=rpm.ngram_complexity,
                            lex_is_row_indices=true))

embed_document(rpm::RPModel{S,T,A,H}, doc::AbstractString) where {S,T,A,H} =
    embed_document(rpm, NGramDocument{S}(doc, DocumentMetadata(), rpm.ngram_complexity))

embed_document(rpm::RPModel{S,T,A,H}, doc::AbstractVector{S2}) where {S,T,A,H,S2<:AbstractString} =
    embed_document(rpm, TokenDocument{S}(doc))

# Actual embedding function: takes as input the random projection model `rpm` and a document
# term vector `dtv`. Returns the representation of `dtv` in the embedding space.
function embed_document(rpm::RPModel{S,T,A,H}, dtv::AbstractVector{T}) where {S,T,A,H}
    words_in_document = sum(dtv)
    # Calculate document vector
    if rpm.stats == :count
        v = dtv
    elseif rpm.stats == :tf
        v = sqrt.(dtv ./ max(words_in_document, one(T)))
    elseif rpm.stats == :tfidf
        v = sqrt.(dtv ./ max(words_in_document, one(T))) .* rpm.idf
    elseif rpm.stats == :bm25
        k = T(rpm.κ)
        b = T(rpm.β)
        tf = sqrt.(dtv ./ max(words_in_document, one(T)))
        v = rpm.idf .* (k + 1) .* tf ./
                       (k * (one(T) - b + b * words_in_document/rpm.nwords) .+ tf)
    end
    # Embed
    local d̂
    if rpm.project
        d̂ = rpm.R * v  # embed
    else
        d̂ = v
    end
    d̂ = d̂ ./ (norm(d̂,2) .+ eps(T))  # normalize
    return d̂
end

function embed_document(rpm::RPModel{S,T,A,H}, dtm::DocumentTermMatrix{T}) where {S,T,A,H}
    n = size(dtm.dtm,1)
    k = size(rpm.R, 1)
    if rpm.stats == :count
        X = dtm.dtm
    elseif rpm.stats == :tf
        X = tf(dtm)
    elseif rpm.stats == :tfidf
        X = tf_idf(dtm)
    elseif rpm.stats == :bm25
        X = bm_25(dtm, κ=rpm.κ, β=rpm.β)
    end
    local U
    if rpm.project
        U = rpm.R * X
    else
        U = X
    end
    U ./= (sqrt.(sum(U.^2, dims=1)) .+ eps(T))
    return U
end

function embed_document(rpm::RPModel{S,T,A,H}, crps::Corpus) where {S,T,A,H}
    lex = create_lexicon(crps, rpm.ngram_complexity)
    embed_document(rpm, DocumentTermMatrix{T}(crps, lex, ngram_complexity=rpm.ngram_complexity))
end


"""
    save_rp_model(rpm, filename)

Saves an random projection model `rpm` to disc in file `filename`.
"""
function save_rp_model(rpm::RPModel{S,T,A,H}, filename::AbstractString) where {S,T,A,H}
    k, nwords = size(rpm.R)
    open(filename, "w") do fid
        println(fid, "Random Projection Model saved at $(Dates.now())")
        println(fid, "$nwords $k")  # number of words, k
        println(fid, rpm.project)
        println(fid, rpm.stats)
        writedlm(fid, rpm.idf', " ")
        println(fid, rpm.nwords)
        println(fid, rpm.ngram_complexity)
        println(fid, rpm.κ)
        println(fid, rpm.β)
        # Vocabulary
        mv = Matrix{String}(undef, 1, nwords)
        mv[1,:] = rpm.vocab
        writedlm(fid, mv, " ")
        # Random projection matrix
        writedlm(fid, rpm.R, " ")
    end
end


"""
    load_rp_model(filename, eltype; [sparse=true])

Loads an random projection model from `filename` into an random projection model object.
The projection matrix element type is specified by `eltype` (default `DEFAULT_FLOAT_TYPE`)
while the keyword argument `sparse` specifies whether the matrix should be sparse or not.
"""
function load_rp_model(filename::AbstractString, ::Type{T}=DEFAULT_FLOAT_TYPE;
                       sparse::Bool=true) where T<: AbstractFloat
    # Matrix type for random projection model
    A = ifelse(sparse, SparseMatrixCSC{T, Int}, Matrix{T})
    # Define parsed variables local to outer scope of do statement
    local vocab, vocab_hash, R, stats, idf, nwords, ngram_complexity, κ, β, project
    open(filename, "r") do fid
        readline(fid)  # first line, header
        line = readline(fid)
        vocab_size, k = map(x -> parse(Int, x), split(line, ' '))
        # Preallocate
        if sparse
            R = SparseMatrixCSC{T, Int}(UniformScaling(0), k, vocab_size)
        else
            R = zeros(T, k, vocab_size)
        end
        # Start parsing the rest of the file
        project = parse(Bool, strip(readline(fid)))
        stats = Symbol(strip(readline(fid)))
        idf = map(x->parse(T, x), split(readline(fid), ' '))
        nwords = parse(T, readline(fid))
        ngram_complexity = parse(Int, readline(fid))
        κ = parse(Int, readline(fid))
        β = parse(Float64, readline(fid))
        vocab = Vector{String}(split(readline(fid),' '))
        vocab_hash = OrderedDict((v,i) for (i,v) in enumerate(vocab))
        for i in 1:k
            R[i,:] = map(x->parse(T,x), split(readline(fid), ' '))
        end
    end
    return RPModel{String,T,A,Int}(vocab, vocab_hash, R, stats, idf,
                                      nwords, ngram_complexity, κ, β, project)
end
