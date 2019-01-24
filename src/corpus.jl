# Basic Corpus type
mutable struct Corpus{S, T<:AbstractDocument{S}}
    documents::Vector{T}
    total_terms::Int
    lexicon::OrderedDict{S, Int}
    inverse_index::OrderedDict{S, Vector{Int}}
    h::TextHashFunction
end

# Constructors
Corpus(docs::Vector{T};
       hash_function::Function = DEFAULT_HASH_FUNCTION,
       cardinality::Int=DEFAULT_CARDINALITY
      ) where T<:AbstractDocument{S} where S<:AbstractString =
    Corpus(
        docs,
        0,
        OrderedDict{S, Int}(),
        OrderedDict{S, Vector{Int}}(),
        TextHashFunction(hash_function, cardinality)
    )

Corpus(docs::Vector{AbstractDocument{S}};
       hash_function::Function = DEFAULT_HASH_FUNCTION,
       cardinality::Int=DEFAULT_CARDINALITY) where S<:AbstractString =
    Corpus(Vector{GenericDocument{S}}(docs),
           hash_function=hash_function,
           cardinality=cardinality)

Corpus(docs::Vector{AbstractDocument};
       hash_function::Function = DEFAULT_HASH_FUNCTION,
       cardinality::Int=DEFAULT_CARDINALITY) where S<:AbstractString = begin
    # Force-convert to String ;)
    n = length(docs)
    T = String
    new_docs = Vector{GenericDocument{T}}(undef, n)
    for i in 1:n
        docs[i] isa FileDocument && (new_docs[i] = convert(FileDocument{T}, docs[i]))
        docs[i] isa StringDocument && (new_docs[i] = convert(StringDocument{T}, docs[i]))
        docs[i] isa TokenDocument && (new_docs[i] = convert(TokenDocument{T}, docs[i]))
        docs[i] isa NGramDocument && (new_docs[i] = convert(NGramDocument{T}, docs[i]))
    end
    return Corpus(new_docs; hash_function=hash_function, cardinality=cardinality)
end



# Construct a Corpus from a directory of text files
function DirectoryCorpus(dirname::AbstractString;
                         hash_function::Function = DEFAULT_HASH_FUNCTION,
                         cardinality::Int=DEFAULT_CARDINALITY)
    # Recursive descent of directory
    # Add all non-hidden files to Corpus
    docs = GenericDocument{String}[]

    function add_files(dirname::AbstractString)
        if !isdir(dirname)
            error("DirectoryCorpus() can only be called on directories")
        end
        starting_dir = pwd()
        cd(dirname)
        for filename in readdir(".")
            if isfile(filename) && !occursin(r"^\.", filename)
                push!(docs, FileDocument(abspath(filename)))
            end
            if isdir(filename) && !islink(filename)
                add_files(filename)
            end
        end
        cd(starting_dir)
    end

    add_files(dirname)
    return Corpus(docs, hash_function=hash_function, cardinality=cardinality)
end


# Basic Corpus properties
documents(c::Corpus) = c.documents


# Treat a Corpus as an iterable
function Base.iterate(crps::Corpus, ind=1)
    ind > length(crps.documents) && return nothing
    crps.documents[ind], ind+1
end

Base.eltype(::Type{Corpus{T}}) where T = T

Base.length(crps::Corpus) = length(crps.documents)

Base.size(crps, i) = size(crps.documents, i)



# Treat a Corpus as a container
Base.push!(crps::Corpus, d::AbstractDocument) = push!(crps.documents, d)

Base.pop!(crps::Corpus) = pop!(crps.documents)

Base.pushfirst!(crps::Corpus, d::AbstractDocument) = pushfirst!(crps.documents, d)

Base.popfirst!(crps::Corpus) = popfirst!(crps.documents)

function Base.insert!(crps::Corpus, index::Int, d::AbstractDocument)
    insert!(crps.documents, index, d)
end

Base.deleteat!(crps::Corpus, index::Integer) = deleteat!(crps.documents, index)


# Indexing into a Corpus
#
# (a) Numeric indexing just provides the n-th document
# (b) String indexing is effectively a trivial search engine
Base.getindex(crps::Corpus, ind::Integer) = crps.documents[ind]
Base.getindex(crps::Corpus, inds::Vector{T}) where {T <: Integer} = crps.documents[inds]
Base.getindex(crps::Corpus, r::AbstractRange) = crps.documents[r]
Base.getindex(crps::Corpus, term::AbstractString) = begin
    isempty(crps.inverse_index) && @warn "Inverse index is empty."
    get(crps.inverse_index, term, Int[])
end
# Assignment into a Corpus
function Base.setindex!(crps::Corpus, d::AbstractDocument, ind::Real)
    crps.documents[ind] = d
    return d
end



# Lexicon and inverse index
# TODO: Offer progressive update that only changes based on current document
lexicon(crps::Corpus) = crps.lexicon

function update_lexicon!(crps::Corpus, doc::AbstractDocument)
    ngs = ngrams(doc)
    for (ngram, counts) in ngs
        crps.total_terms += counts
        crps.lexicon[ngram] = get(crps.lexicon, ngram, 0) + counts
    end
end

function update_lexicon!(crps::Corpus)
    crps.total_terms = 0
    crps.lexicon = OrderedDict{String,Int}()
    for doc in crps
        update_lexicon!(crps, doc)
    end
end

lexicon_size(crps::Corpus) = length(keys(crps.lexicon))

lexical_frequency(crps::Corpus, term::AbstractString) =
    (get(crps.lexicon, term, 0) / crps.total_terms)



# Work with the Corpus's inverse index
# TODO: offer progressive update that only changes based on current document
inverse_index(crps::Corpus) = crps.inverse_index

function update_inverse_index!(crps::Corpus)
    idx = OrderedDict{String, Vector{Int}}()
    for i in 1:length(crps)
        doc = crps.documents[i]
        ngram_arr = isa(doc, NGramDocument) ? collect(keys(ngrams(doc))) : tokens(doc)
        ngram_arr = convert(Vector{String}, ngram_arr)
        for ngram in ngram_arr
            if haskey(idx, ngram)
                push!(idx[ngram], i)
            else
                idx[ngram] = [i]
            end
        end
    end
    for key in keys(idx)
        idx[key] = unique(idx[key])
    end
    crps.inverse_index = idx
    nothing
end

index_size(crps::Corpus) = length(crps.inverse_index)


# Every Corpus prespecifies a hash function for hash trick analysis
hash_function(crps::Corpus) = crps.h

hash_function!(crps::Corpus, f::TextHashFunction) = (crps.h = f; nothing)


# Standardize the documents in a Corpus to a common type
function standardize!(crps::Corpus, ::Type{T}) where T<:AbstractDocument
    for i in 1:length(crps)
        crps.documents[i] = convert(T, crps.documents[i])
    end
end
