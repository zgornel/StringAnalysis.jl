"""
Basic Document-Term-Matrix (DTM) type.

# Fields
  * `dtm::SparseMatriCSC{T,Int}` the actual DTM; rows represent terms
and columns represent documents
  * `terms::Vector{String}` a list of terms that represent the lexicon of
the corpus associated with the DTM
  * `row_indices::Dict{String, Int}` a map between the `terms` and the
rows of the `dtm`
"""
mutable struct DocumentTermMatrix{T}
    dtm::SparseMatrixCSC{T, Int}
    terms::Vector{String}
    row_indices::Dict{String, Int}
end

"""
    rowindices(terms)

Returns a dictionary that maps each term from the vector `terms`
to a integer idex.
"""
function rowindices(terms::Vector{String})
    row_indices = Dict{String, Int}()
    for (i, term) in enumerate(terms)
        row_indices[term] = i
    end
    return row_indices
end

"""
    columnindices(terms)

Identical to `rowindices`. Returns a dictionary that maps
each term from the vector `terms` to a integer idex.
"""
columnindices = rowindices


"""
    DocumentTermMatrix{T}(crps::Corpus [,terms])

Auxiliary constructor(s) of the `DocumentTermMatrix` type. The type `T` has to be
a subtype of `Real`. The constructor(s) requires a corpus `crps` and
a `terms` structure representing the lexicon of the corpus. The latter
can be a `Vector{String}`, an `AbstractDict` where the keys are the lexicon, or can
be missing, in which case the `lexicon` field of the corpus is used.
"""
function DocumentTermMatrix{T}(crps::Corpus, terms::Vector{String}) where T<:Real
    row_indices = rowindices(terms)
    m = length(terms)
    n = length(crps)
    rows = Vector{Int}(undef, 0)
    columns = Vector{Int}(undef, 0)
    values = Vector{T}(undef, 0)
    for (i, doc) in enumerate(crps)
        ngs = ngrams(doc)
        for ngram in keys(ngs)
            j = get(row_indices, ngram, 0)
            v = ngs[ngram]
            if j != 0
                push!(columns, i)
                push!(rows, j)
                push!(values, v)
            end
        end
    end
    if length(rows) > 0
        dtm = sparse(rows, columns, values, m, n)
    else
        dtm = spzeros(T, m, n)
    end
    return DocumentTermMatrix(dtm, terms, row_indices)
end

DocumentTermMatrix(crps::Corpus, terms::Vector{String}) =
    DocumentTermMatrix{DEFAULT_DTM_TYPE}(crps, terms)

DocumentTermMatrix{T}(crps::Corpus, lex::AbstractDict) where T<:Real =
    DocumentTermMatrix{T}(crps, sort(collect(keys(lex))))

DocumentTermMatrix(crps::Corpus, lex::AbstractDict) =
    DocumentTermMatrix{DEFAULT_DTM_TYPE}(crps, lex)

DocumentTermMatrix{T}(crps::Corpus) where T<:Real = begin
    isempty(lexicon(crps)) && update_lexicon!(crps)
    DocumentTermMatrix{T}(crps, lexicon(crps))
end

DocumentTermMatrix(crps::Corpus) = begin
    isempty(lexicon(crps)) && update_lexicon!(crps)
    DocumentTermMatrix{DEFAULT_DTM_TYPE}(crps, lexicon(crps))
end

DocumentTermMatrix(dtm::SparseMatrixCSC{T, Int},
                   terms::Vector{String}) where T<:Real =
    DocumentTermMatrix(dtm, terms, rowindices(terms))


"""
    dtm(d::DocumentTermMatrix)

Access the matrix of a `DocumentTermMatrix` `d`.
"""
dtm(d::DocumentTermMatrix) = d.dtm

"""
    dtm(crps::Corpus, eltype::Type{T}=DEFAULT_DTM_TYPE)

Access the matrix of the DTM associated with the corpus `crps`. The
`DocumentTermMatrix{T}` will first have to be created in order for
the actual matrix to be accessed.
"""
dtm(crps::Corpus, eltype::Type{T}=DEFAULT_DTM_TYPE) where T<:Real =
    dtm(DocumentTermMatrix{T}(crps))


# Produce the signature of a DTM entry for a document
function dtm_entries(d::AbstractDocument,
                     lex::Dict{String, Int},
                     eltype::Type{T}=DEFAULT_DTM_TYPE) where T<:Real
    ngs = ngrams(d)
    indices = Vector{Int}(undef, 0)
    values = Vector{T}(undef, 0)
    terms = sort(collect(keys(lex)))
    row_indices = rowindices(terms)
    for ngram in keys(ngs)
        j = get(row_indices, ngram, 0)
        v = ngs[ngram]
        if j != 0
            push!(indices, j)
            push!(values, v)
        end
    end
    return (indices, values)
end


"""
    dtv(d::AbstractDocument, lex::Dict{String,Int}, eltype::Type{T}=DEFAULT_DTM_TYPE)

Creates a document-term-vector with elements of type `T` for document `d`
using the lexicon `lex`.
"""
function dtv(d::AbstractDocument,
             lex::Dict{String, Int},
             eltype::Type{T}=DEFAULT_DTM_TYPE) where T<:Real
    p = length(keys(lex))
    column = zeros(T, p)
    indices, values = dtm_entries(d, lex)
    column[indices] = values
    return column
end

"""
    dtv(crps::Corpus, idx::Int, eltype::Type{T}=DEFAULT_DTM_TYPE)

Creates a document-term-vector with elements of type `T` for document `idx`
of the corpus `crps`.
"""
function dtv(crps::Corpus,
             idx::Int,
             eltype::Type{T}=DEFAULT_DTM_TYPE) where T<:Real
    if isempty(crps.lexicon)
        error("Cannot construct a DTV without a pre-existing lexicon")
    elseif idx >= length(crps.documents) || idx < 1
        error("DTV requires the document index in [1,$(length(crps.documents))]")
    else
        return dtv(crps.documents[idx], crps.lexicon, eltype)
    end
end

function dtv(d::AbstractDocument)
    error("Cannot construct a DTV without a pre-existing lexicon")
end


"""
    hash_dtv(d::AbstractDocument, h::TextHashFunction, eltype::Type{T}=DEFAULT_DTM_TYPE)

Creates a hashed document-term-vector with elements of type `T` for document `d`
using the hashing function `h`.
"""
function hash_dtv(d::AbstractDocument,
                  h::TextHashFunction,
                  eltype::Type{T}=DEFAULT_DTM_TYPE) where T<:Real
    p = cardinality(h)
    res = zeros(T, p)
    ngs = ngrams(d)
    for ng in keys(ngs)
        res[index_hash(ng, h)] += ngs[ng]
    end
    return res
end

hash_dtv(d::AbstractDocument;
         cardinality::Int=DEFAULT_CARDINALITY,
         eltype::Type{T}=DEFAULT_DTM_TYPE) where T<:Real =
    hash_dtv(d, TextHashFunction(cardinality), eltype)


"""
    hash_dtm(crps::Corpus [,h::TextHashFunction], eltype::Type{T}=DEFAULT_DTM_TYPE)

Creates a hashed DTM with elements of type `T` for corpus `crps` using the
the hashing function `h`. If `h` is missing, the hash function of the `Corpus`
is used.
"""
function hash_dtm(crps::Corpus,
                  h::TextHashFunction,
                  eltype::Type{T}=DEFAULT_DTM_TYPE) where T<:Real
    n, p = length(crps), cardinality(h)
    res = zeros(T, p, n)
    for (i, doc) in enumerate(crps)
        res[:, i] = hash_dtv(doc, h, eltype)
    end
    return res
end

hash_dtm(crps::Corpus, eltype::Type{T}=DEFAULT_DTM_TYPE) where T<:Real =
    hash_dtm(crps, hash_function(crps), eltype)


# Produce entries for on-line analysis when DTM would not fit in memory
mutable struct EachDTV{U, S<:AbstractString, T<:AbstractDocument}
    corpus::Corpus{S,T}
    function EachDTV{U,S,T}(corpus::Corpus{S,T}) where
            {U, S<:AbstractString, T<:AbstractDocument}
        isempty(lexicon(corpus)) && update_lexicon!(corpus)
        new(corpus)
    end
end

EachDTV{U}(crps::Corpus{S,T}) where {U,S,T} = EachDTV{U,S,T}(crps)

Base.iterate(edt::EachDTV, state=1) = begin
    if state > length(edt.corpus)
        return nothing
    else
        return next(edt, state)
    end
end

next(edt::EachDTV{U,S,T}, state::Int) where {U,S,T} =
    (dtv(edt.corpus.documents[state], lexicon(edt.corpus), U), state + 1)

"""
    each_dtv(crps::Corpus [; eltype::Type{U}=DEFAULT_DTM_TYPE])

Iterates through the columns of the DTM of the corpus `crps` without
constructing it. Useful when the DTM would not fit in memory.
`eltype` specifies the element type of the generated vectors.
"""
each_dtv(crps::Corpus; eltype::Type{U}=DEFAULT_DTM_TYPE) where U<:Real =
    EachDTV{U}(crps)

Base.eltype(::Type{EachDTV{U,S,T}}) where {U,S,T} = Vector{U}

Base.length(edt::EachDTV) = length(edt.corpus)

Base.size(edt::EachDTV) = (length(edt.corpus), edt.corpus.h.cardinality)

Base.show(io::IO, edt::EachDTV{U,S,T}) where {U,S,T} =
    print(io, "DTV iterator, $(length(edt)) elements of type $(eltype(edt)).")


mutable struct EachHashDTV{U, S<:AbstractString, T<:AbstractDocument}
    corpus::Corpus{S,T}
    function EachHashDTV{U,S,T}(corpus::Corpus{S,T}) where
            {U, S<:AbstractString, T<:AbstractDocument}
        isempty(lexicon(corpus)) && update_lexicon!(corpus)
        new(corpus)
    end
end

EachHashDTV{U}(crps::Corpus{S,T}) where {U,S,T} = EachHashDTV{U,S,T}(crps)

Base.iterate(edt::EachHashDTV, state=1) = begin
    if state > length(edt.corpus)
        return nothing
    else
        return next(edt, state)
    end
end

next(edt::EachHashDTV{U,S,T}, state::Int) where {U,S,T} =
    (hash_dtv(edt.corpus.documents[state], edt.corpus.h, U), state + 1)

"""
    each_hash_dtv(crps::Corpus [; eltype::Type{U}=DEFAULT_DTM_TYPE])

Iterates through the columns of the hashed DTM of the corpus `crps` without
constructing it. Useful when the DTM would not fit in memory.
`eltype` specifies the element type of the generated vectors.
"""
each_hash_dtv(crps::Corpus; eltype::Type{U}=DEFAULT_DTM_TYPE) where U<:Real =
    EachHashDTV{U}(crps)

Base.eltype(::Type{EachHashDTV{U,S,T}}) where {U,S,T} = Vector{U}

Base.length(edt::EachHashDTV) = length(edt.corpus)

Base.size(edt::EachHashDTV) = (length(edt.corpus), edt.corpus.h.cardinality)

Base.show(io::IO, edt::EachHashDTV{U,S,T}) where {U,S,T} =
    print(io, "Hash-DTV iterator, $(length(edt)) elements of type $(eltype(edt)).")


## getindex() methods
Base.getindex(dtm::DocumentTermMatrix, k::AbstractString) = dtm.dtm[dtm.row_indices[k], :]

Base.getindex(dtm::DocumentTermMatrix, i) = dtm.dtm[i]

Base.getindex(dtm::DocumentTermMatrix, i, j) = dtm.dtm[i, j]
