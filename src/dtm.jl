# Basic DocumentTermMatrix type
mutable struct DocumentTermMatrix{T}
    dtm::SparseMatrixCSC{T, Int}
    terms::Vector{String}
    column_indices::Dict{String, Int}
end

# Construct a DocumentTermMatrix from a Corpus
# create col index lookup dictionary from a (sorted?) vector of terms
function columnindices(terms::Vector{String})
    column_indices = Dict{String, Int}()
    for (i, term) in enumerate(terms)
        column_indices[term] = i
    end
    return column_indices
end

function DocumentTermMatrix{T}(crps::Corpus,
                               terms::Vector{String}) where T<:Real
    column_indices = columnindices(terms)
    m = length(crps)
    n = length(terms)
    rows = Vector{Int}(undef, 0)
    columns = Vector{Int}(undef, 0)
    values = Vector{T}(undef, 0)
    for (i, doc) in enumerate(crps)
        ngs = ngrams(doc)
        for ngram in keys(ngs)
            j = get(column_indices, ngram, 0)
            v = ngs[ngram]
            if j != 0
                push!(rows, i)
                push!(columns, j)
                push!(values, v)
            end
        end
    end
    if length(rows) > 0
        dtm = sparse(rows, columns, values, m, n)
    else
        dtm = spzeros(T, m, n)
    end
    return DocumentTermMatrix(dtm, terms, column_indices)
end

DocumentTermMatrix(crps::Corpus, terms::Vector{String}) =
    DocumentTermMatrix{DEFAULT_DTM_TYPE}(crps, terms)

DocumentTermMatrix(crps::Corpus, lex::AbstractDict) =
    DocumentTermMatrix(crps, sort(collect(keys(lex))))

DocumentTermMatrix(crps::Corpus) = begin
    isempty(lexicon(crps)) && update_lexicon!(crps)
    DocumentTermMatrix(crps, lexicon(crps))
end

DocumentTermMatrix(dtm::SparseMatrixCSC{T, Int},
                   terms::Vector{String}) where T<:Real =
    DocumentTermMatrix(dtm, terms, columnindices(terms))


# Access the DTM of a DocumentTermMatrix
dtm(d::DocumentTermMatrix) = d.dtm

dtm(crps::Corpus) = dtm(DocumentTermMatrix(crps))


# Term-document matrix
tdm(crps::DocumentTermMatrix) = dtm(crps)' #'

tdm(crps::Corpus) = dtm(crps)' #'


# Produce the signature of a DTM entry for a document
function dtm_entries(d::AbstractDocument,
                     lex::Dict{String, Int},
                     eltype::Type{T}=DEFAULT_DTM_TYPE) where T<:Real
    ngs = ngrams(d)
    indices = Vector{Int}(undef, 0)
    values = Vector{T}(undef, 0)
    terms = sort(collect(keys(lex)))
    column_indices = columnindices(terms)
    for ngram in keys(ngs)
        j = get(column_indices, ngram, 0)
        v = ngs[ngram]
        if j != 0
            push!(indices, j)
            push!(values, v)
        end
    end
    return (indices, values)
end

function dtv(d::AbstractDocument,
             lex::Dict{String, Int},
             eltype::Type{T}=DEFAULT_DTM_TYPE) where T<:Real
    p = length(keys(lex))
    row = zeros(T, p)
    indices, values = dtm_entries(d, lex)
    row[indices] = values
    return row
end

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


# The hash trick: use a hash function instead of a lexicon to determine the
# columns of a DocumentTermMatrix-like encoding of the data
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

function hash_dtm(crps::Corpus,
                  h::TextHashFunction,
                  eltype::Type{T}=DEFAULT_DTM_TYPE) where T<:Real
    n, p = length(crps), cardinality(h)
    res = zeros(T, n, p)
    for (i, doc) in enumerate(crps)
        res[i, :] = hash_dtv(doc, h, eltype)
    end
    return res
end


hash_dtm(crps::Corpus, eltype::Type{T}=DEFAULT_DTM_TYPE) where T<:Real =
    hash_dtm(crps, hash_function(crps), eltype)

hash_tdm(crps::Corpus, eltype::Type{T}=DEFAULT_DTM_TYPE) where T<:Real =
    hash_dtm(crps, eltype)' #'



# Produce entries for on-line analysis when DTM would not fit in memory
mutable struct EachDTV{U, S<:AbstractString, T<:AbstractDocument}
    corpus::Corpus{S,T}
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

each_dtv(crps::Corpus; eltype::Type{U}=DEFAULT_DTM_TYPE) where U<:Real =
    EachDTV{U}(crps)

Base.eltype(::Type{EachDTV{U,S,T}}) where {U,S,T} = Vector{U}

Base.length(edt::EachDTV) = length(edt.corpus)

Base.size(edt::EachDTV) = (length(edt.corpus), edt.corpus.h.cardinality)

Base.show(io::IO, edt::EachDTV{U,S,T}) where {U,S,T} =
    print(io, "DTV iterator, $(length(edt)) elements of type $(eltype(edt)).")


mutable struct EachHashDTV{U, S<:AbstractString, T<:AbstractDocument}
    corpus::Corpus{S,T}
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

each_hash_dtv(crps::Corpus; eltype::Type{U}=DEFAULT_DTM_TYPE) where U<:Real =
    EachHashDTV{U}(crps)

Base.eltype(::Type{EachHashDTV{U,S,T}}) where {U,S,T} = Vector{U}

Base.length(edt::EachHashDTV) = length(edt.corpus)

Base.size(edt::EachHashDTV) = (length(edt.corpus), edt.corpus.h.cardinality)

Base.show(io::IO, edt::EachHashDTV{U,S,T}) where {U,S,T} =
    print(io, "Hash-DTV iterator, $(length(edt)) elements of type $(eltype(edt)).")


## getindex() methods
Base.getindex(dtm::DocumentTermMatrix, k::AbstractString) = dtm.dtm[:, dtm.column_indices[k]]

Base.getindex(dtm::DocumentTermMatrix, i) = dtm.dtm[i]

Base.getindex(dtm::DocumentTermMatrix, i, j) = dtm.dtm[i, j]
