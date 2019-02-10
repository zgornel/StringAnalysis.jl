"""
Basic Document-Term-Matrix (DTM) type.

# Fields
  * `dtm::SparseMatriCSC{T,Int}` the actual DTM; rows represent terms
and columns represent documents
  * `terms::Vector{String}` a list of terms that represent the lexicon of
the corpus associated with the DTM
  * `row_indices::OrderedDict{String, Int}` a map between the `terms` and the
rows of the `dtm`
"""
mutable struct DocumentTermMatrix{T}
    dtm::SparseMatrixCSC{T, Int}
    terms::Vector{String}
    row_indices::OrderedDict{String, Int}
end

"""
    rowindices(terms)

Returns a dictionary that maps each term from the vector `terms`
to a integer idex.
"""
function rowindices(terms::Vector{String})
    row_indices = OrderedDict{String, Int}(
                    term => i for (i,term) in enumerate(terms))
    return row_indices
end

"""
    columnindices(terms)

Identical to `rowindices`. Returns a dictionary that maps
each term from the vector `terms` to a integer idex.
"""
columnindices = rowindices


"""
    DocumentTermMatrix{T}(crps::Corpus [,terms] [; tokenizer=DEFAULT_TOKENIZER])

Auxiliary constructor(s) of the `DocumentTermMatrix` type. The type `T` has to be
a subtype of `Real`. The constructor(s) requires a corpus `crps` and
a `terms` structure representing the lexicon of the corpus. The latter
can be a `Vector{String}`, an `AbstractDict` where the keys are the lexicon, or can
be missing, in which case the `lexicon` field of the corpus is used.
"""
function DocumentTermMatrix{T}(crps::Corpus,
                               terms::Vector{String};
                               tokenizer::Symbol=DEFAULT_TOKENIZER) where T<:Real
    row_indices = rowindices(terms)
    m = length(terms)
    n = length(crps)
    rows = Vector{Int}(undef, 0)
    columns = Vector{Int}(undef, 0)
    values = Vector{T}(undef, 0)
    for (i, doc) in enumerate(crps)
        ngs = ngrams(doc, tokenizer=tokenizer)
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

DocumentTermMatrix(crps::Corpus, terms::Vector{String}; tokenizer::Symbol=DEFAULT_TOKENIZER) =
    DocumentTermMatrix{DEFAULT_DTM_TYPE}(crps, terms, tokenizer=tokenizer)

DocumentTermMatrix{T}(crps::Corpus, lex::AbstractDict; tokenizer::Symbol=DEFAULT_TOKENIZER
                     ) where T<:Real =
    DocumentTermMatrix{T}(crps, collect(keys(lex)), tokenizer=tokenizer)

DocumentTermMatrix(crps::Corpus, lex::AbstractDict; tokenizer::Symbol=DEFAULT_TOKENIZER) =
    DocumentTermMatrix{DEFAULT_DTM_TYPE}(crps, lex, tokenizer=tokenizer)

DocumentTermMatrix{T}(crps::Corpus; tokenizer::Symbol=DEFAULT_TOKENIZER) where T<:Real = begin
    isempty(lexicon(crps)) && update_lexicon!(crps)
    DocumentTermMatrix{T}(crps, lexicon(crps), tokenizer=tokenizer)
end

DocumentTermMatrix(crps::Corpus; tokenizer::Symbol=DEFAULT_TOKENIZER) = begin
    isempty(lexicon(crps)) && update_lexicon!(crps)
    DocumentTermMatrix{DEFAULT_DTM_TYPE}(crps, lexicon(crps), tokenizer=tokenizer)
end

DocumentTermMatrix(dtm::SparseMatrixCSC{T, Int}, terms::Vector{String}) where T<:Real =
    DocumentTermMatrix(dtm, terms, rowindices(terms))


"""
    dtm(d::DocumentTermMatrix)

Access the matrix of a `DocumentTermMatrix` `d`.
"""
dtm(d::DocumentTermMatrix) = d.dtm

"""
    dtm(crps::Corpus, eltype::Type{T}=DEFAULT_DTM_TYPE [; tokenizer=DEFAULT_TOKENIZER])

Access the matrix of the DTM associated with the corpus `crps`. The
`DocumentTermMatrix{T}` will first have to be created in order for
the actual matrix to be accessed.
"""
dtm(crps::Corpus, eltype::Type{T}=DEFAULT_DTM_TYPE;
        tokenizer::Symbol=DEFAULT_TOKENIZER) where T<:Real =
    dtm(DocumentTermMatrix{T}(crps, tokenizer=tokenizer))


# Produce the signature of a DTM entry for a document
function dtm_entries(d, lex::OrderedDict{String, Int},
                     eltype::Type{T}=DEFAULT_DTM_TYPE;
                     tokenizer::Symbol=DEFAULT_TOKENIZER,
                     lex_is_row_indices::Bool=false) where T<:Real
    ngs = ngrams(d, tokenizer=tokenizer)
    indices = Vector{Int}(undef, 0)
    values = Vector{T}(undef, 0)
    local row_indices
    if lex_is_row_indices
        row_indices = lex
    else
        terms = collect(keys(lex))
        row_indices = rowindices(terms)
    end
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
    dtv(d, lex::OrderedDict{String,Int}, eltype::Type{T}=DEFAULT_DTM_TYPE [; tokenizer=DEFAULT_TOKENIZER])

Creates a document-term-vector with elements of type `T` for document `d`
using the lexicon `lex`. `d` can be an `AbstractString` or an `AbstractDocument`.
"""
function dtv(d, lex::OrderedDict{String, Int},
             eltype::Type{T}=DEFAULT_DTM_TYPE;
             tokenizer::Symbol=DEFAULT_TOKENIZER,
             lex_is_row_indices::Bool=false) where T<:Real
    p = length(keys(lex))
    column = spzeros(T, p)
    indices, values = dtm_entries(d, lex, eltype, tokenizer=tokenizer,
                                  lex_is_row_indices=lex_is_row_indices)
    column[indices] = values
    return column
end

"""
    dtv(crps::Corpus, idx::Int, eltype::Type{T}=DEFAULT_DTM_TYPE [; tokenizer=DEFAULT_TOKENIZER])

Creates a document-term-vector with elements of type `T` for document `idx`
of the corpus `crps`.
"""
function dtv(crps::Corpus, idx::Int,
             eltype::Type{T}=DEFAULT_DTM_TYPE;
             tokenizer::Symbol=DEFAULT_TOKENIZER,
             lex_is_row_indices::Bool=false) where T<:Real
    if isempty(crps.lexicon)
        error("Cannot construct a DTV without a pre-existing lexicon")
    elseif idx >= length(crps.documents) || idx < 1
        error("DTV requires the document index in [1,$(length(crps.documents))]")
    else
        return dtv(crps.documents[idx], crps.lexicon, eltype, tokenizer=tokenizer,
                   lex_is_row_indices=lex_is_row_indices)
    end
end

function dtv(d; tokenizer::Symbol=DEFAULT_TOKENIZER, lex_is_row_indices::Bool=false)
    throw(ErrorException("Cannot construct a DTV without a pre-existing lexicon"))
end


# Document is a list of regular expressions in text form
function dtm_regex_entries(d, lex::OrderedDict{String, Int},
                           eltype::Type{T}=DEFAULT_DTM_TYPE;
                           tokenizer::Symbol=DEFAULT_TOKENIZER,
                           lex_is_row_indices::Bool=false) where T<:Real
    ngs = ngrams(d, tokenizer=tokenizer)
    patterns = Regex.(keys(ngs))
    indices = Vector{Int}(undef, 0)
    terms = collect(keys(lex))
    local row_indices
    if lex_is_row_indices
        row_indices = lex
    else
        row_indices = rowindices(terms)
    end
    for pattern in patterns
        for term in terms
            if occursin(pattern, term)
                j = row_indices[term]
                push!(indices, j)
            end
        end
    end
    values = ones(T, length(indices))
    return (indices, values)
end


"""
    dtv_regex(d, lex::OrderedDict{String,Int}, eltype::Type{T}=DEFAULT_DTM_TYPE [; tokenizer=DEFAULT_TOKENIZER])

Creates a document-term-vector with elements of type `T` for document `d`
using the lexicon `lex`. The tokens of document `d` are assumed to be regular
expressions in text format. `d` can be an `AbstractString` or an `AbstractDocument`.

# Examples
```
julia> dtv_regex(NGramDocument("a..b"), OrderedDict("aaa"=>1, "aaab"=>2, "accb"=>3, "bbb"=>4), Float32)
4-element Array{Float32,1}:
 0.0
 1.0
 1.0
 0.0
```
"""
function dtv_regex(d, lex::OrderedDict{String, Int},
                   eltype::Type{T}=DEFAULT_DTM_TYPE;
                   tokenizer::Symbol=DEFAULT_TOKENIZER,
                   lex_is_row_indices::Bool=false) where T<:Real
    p = length(keys(lex))
    column = spzeros(T, p)
    indices, values = dtm_regex_entries(d, lex, eltype, tokenizer=tokenizer,
                                        lex_is_row_indices=lex_is_row_indices)
    column[indices] = values
    return column
end


"""
    hash_dtv(d, h::TextHashFunction, eltype::Type{T}=DEFAULT_DTM_TYPE [; tokenizer=DEFAULT_TOKENIZER])

Creates a hashed document-term-vector with elements of type `T` for document `d`
using the hashing function `h`. `d` can be an `AbstractString` or an `AbstractDocument`.
"""
function hash_dtv(d, h::TextHashFunction, eltype::Type{T}=DEFAULT_DTM_TYPE;
                 tokenizer::Symbol=DEFAULT_TOKENIZER) where T<:Real
    p = cardinality(h)
    res = spzeros(T, p)
    ngs = ngrams(d, tokenizer=tokenizer)
    for ng in keys(ngs)
        res[index_hash(ng, h)] += ngs[ng]
    end
    return res
end

function hash_dtv(d; cardinality::Int=DEFAULT_CARDINALITY, eltype::Type{T}=DEFAULT_DTM_TYPE,
                 tokenizer::Symbol=DEFAULT_TOKENIZER) where T<:Real
    hash_dtv(d, TextHashFunction(cardinality), eltype, tokenizer=tokenizer)
end


"""
    hash_dtm(crps::Corpus [,h::TextHashFunction], eltype::Type{T}=DEFAULT_DTM_TYPE [; tokenizer=DEFAULT_TOKENIZER])

Creates a hashed DTM with elements of type `T` for corpus `crps` using the
the hashing function `h`. If `h` is missing, the hash function of the `Corpus`
is used.
"""
function hash_dtm(crps::Corpus,
                  h::TextHashFunction,
                  eltype::Type{T}=DEFAULT_DTM_TYPE;
                  tokenizer::Symbol=DEFAULT_TOKENIZER) where T<:Real
    n, p = length(crps), cardinality(h)
    res = spzeros(T, p, n)
    for (i, doc) in enumerate(crps)
        res[:, i] = hash_dtv(doc, h, eltype, tokenizer=tokenizer)
    end
    return res
end

hash_dtm(crps::Corpus, eltype::Type{T}=DEFAULT_DTM_TYPE;
         tokenizer::Symbol=DEFAULT_TOKENIZER) where T<:Real =
    hash_dtm(crps, hash_function(crps), eltype, tokenizer=tokenizer)


# Produce entries for on-line analysis when DTM would not fit in memory
mutable struct EachDTV{U, S<:AbstractString, T<:AbstractDocument}
    corpus::Corpus{S,T}
    row_indices::OrderedDict{String, Int}
    tokenizer::Symbol
    function EachDTV{U,S,T}(corpus::Corpus{S,T},
                            row_indices::OrderedDict{String, Int},
                            tokenizer::Symbol=DEFAULT_TOKENIZER) where
            {U, S<:AbstractString, T<:AbstractDocument}
        isempty(lexicon(corpus)) && update_lexicon!(corpus)
        @assert tokenizer in [:slow, :fast] "Tokenizer has to be either :slow or :fast"
        new(corpus, row_indices, tokenizer)
    end
end

EachDTV{U}(crps::Corpus{S,T}; tokenizer::Symbol=DEFAULT_TOKENIZER) where {U,S,T} = begin
    isempty(lexicon(crps)) && update_lexicon!(crps)
    row_indices = rowindices(collect(keys(lexicon(crps))))
    EachDTV{U,S,T}(crps, row_indices, tokenizer)
end

Base.iterate(edt::EachDTV, state=1) = begin
    if state > length(edt.corpus)
        return nothing
    else
        return next(edt, state)
    end
end

next(edt::EachDTV{U,S,T}, state::Int) where {U,S,T} =
    (dtv(edt.corpus.documents[state], edt.row_indices, U,
         tokenizer=edt.tokenizer, lex_is_row_indices=true), state + 1)

"""
    each_dtv(crps::Corpus [; eltype::Type{U}=DEFAULT_DTM_TYPE, tokenizer=DEFAULT_TOKENIZER])

Iterates through the columns of the DTM of the corpus `crps` without
constructing it. Useful when the DTM would not fit in memory.
`eltype` specifies the element type of the generated vectors.
"""
each_dtv(crps::Corpus;
         eltype::Type{U}=DEFAULT_DTM_TYPE,
         tokenizer::Symbol=DEFAULT_TOKENIZER) where U<:Real =
    EachDTV{U}(crps, tokenizer=tokenizer)

Base.eltype(::Type{EachDTV{U,S,T}}) where {U,S,T} = Vector{U}

Base.length(edt::EachDTV) = length(edt.corpus)

Base.size(edt::EachDTV) = (length(edt.corpus), edt.corpus.h.cardinality)

Base.show(io::IO, edt::EachDTV{U,S,T}) where {U,S,T} =
    print(io, "DTV iterator, tokenizer is $(edt.tokenizer), "*
          "$(length(edt)) elements of type $(eltype(edt)).")


mutable struct EachHashDTV{U, S<:AbstractString, T<:AbstractDocument}
    corpus::Corpus{S,T}
    tokenizer::Symbol
    function EachHashDTV{U,S,T}(corpus::Corpus{S,T}, tokenizer::Symbol=DEFAULT_TOKENIZER) where
            {U, S<:AbstractString, T<:AbstractDocument}
        @assert tokenizer in [:slow, :fast] "Tokenizer has to be either :slow or :fast"
        new(corpus, tokenizer)
    end
end

EachHashDTV{U}(crps::Corpus{S,T}; tokenizer::Symbol=DEFAULT_TOKENIZER) where {U,S,T} =
    EachHashDTV{U,S,T}(crps)

Base.iterate(edt::EachHashDTV, state=1) = begin
    if state > length(edt.corpus)
        return nothing
    else
        return next(edt, state)
    end
end

next(edt::EachHashDTV{U,S,T}, state::Int) where {U,S,T} =
    (hash_dtv(edt.corpus.documents[state], edt.corpus.h, U, tokenizer=edt.tokenizer), state + 1)

"""
    each_hash_dtv(crps::Corpus [; eltype::Type{U}=DEFAULT_DTM_TYPE, tokenizer=DEFAULT_TOKENIZER])

Iterates through the columns of the hashed DTM of the corpus `crps` without
constructing it. Useful when the DTM would not fit in memory.
`eltype` specifies the element type of the generated vectors.
"""
each_hash_dtv(crps::Corpus;
              eltype::Type{U}=DEFAULT_DTM_TYPE,
              tokenizer::Symbol=DEFAULT_TOKENIZER) where U<:Real =
    EachHashDTV{U}(crps, tokenizer=tokenizer)

Base.eltype(::Type{EachHashDTV{U,S,T}}) where {U,S,T} = Vector{U}

Base.length(edt::EachHashDTV) = length(edt.corpus)

Base.size(edt::EachHashDTV) = (length(edt.corpus), edt.corpus.h.cardinality)

Base.show(io::IO, edt::EachHashDTV{U,S,T}) where {U,S,T} =
    print(io, "Hash-DTV iterator, tokenizer is $(edt.tokenizer) "*
          "$(length(edt)) elements of type $(eltype(edt)).")


## getindex() methods
Base.getindex(dtm::DocumentTermMatrix, k::AbstractString) = dtm.dtm[dtm.row_indices[k], :]

Base.getindex(dtm::DocumentTermMatrix, i) = dtm.dtm[i]

Base.getindex(dtm::DocumentTermMatrix, i, j) = dtm.dtm[i, j]
