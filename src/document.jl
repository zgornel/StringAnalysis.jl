# All Document types share a common metadata profile using DocumentMetadata
mutable struct DocumentMetadata
    language::Languages.Language
    name::String
    author::String
    timestamp::String
    id::String
    publisher::String
    edition_year::String
    published_year::String
    documenttype::String
    note::String
end

DocumentMetadata() = DocumentMetadata(
    DEFAULT_LANGUAGE,
    "Unnamed Document",
    "Unknown Author",
    "Unknown Time",
    "Unknown ID",
    "Unknown Publisher",
    "Unknown Edition Year",
    "Unknown Publishing Year",
    "Unknown Type",
    ""
)


# The abstract Document type
abstract type AbstractDocument{T<:AbstractString}; end

# FileDocument type and constructors
mutable struct FileDocument{T} <: AbstractDocument{T}
    filename::T
    metadata::DocumentMetadata
end

FileDocument{T}(f::AbstractString) where T<:AbstractString = begin
    d = FileDocument(T(f), DocumentMetadata())
    d.metadata.name = f
    return d
end

FileDocument(f::AbstractString) = FileDocument{String}(f)


# StringDocument type and constructors
mutable struct StringDocument{T<:AbstractString} <: AbstractDocument{T}
    text::T
    metadata::DocumentMetadata
end

StringDocument{T}(txt::AbstractString) where T<:AbstractString =
    StringDocument(T(txt), DocumentMetadata())

StringDocument(txt::T) where T<:AbstractString =
    StringDocument{T}(txt, DocumentMetadata())


# TokenDocument type and constructors
mutable struct TokenDocument{T<:AbstractString} <: AbstractDocument{T}
    tokens::Vector{T}
    metadata::DocumentMetadata
end

TokenDocument(tkns::Vector{T}) where T <: AbstractString =
    TokenDocument{T}(tkns, DocumentMetadata())

TokenDocument{T}(tkns::Vector{S}) where {T<:AbstractString, S<:AbstractString} =
    TokenDocument{T}(T.(tkns), DocumentMetadata())

TokenDocument{T}(txt::AbstractString,
                 dm::DocumentMetadata=DocumentMetadata();
                 method::Symbol=DEFAULT_TOKENIZER
                ) where T<:AbstractString =
    TokenDocument{T}(T.(tokenize(txt, method=method)), dm)

TokenDocument(txt::AbstractString,
              dm::DocumentMetadata=DocumentMetadata();
              method::Symbol=DEFAULT_TOKENIZER) =
    TokenDocument(tokenize(txt, method=method), dm)


# NGramDocument type and constructors
mutable struct NGramDocument{T<:AbstractString} <: AbstractDocument{T}
    ngrams::Dict{T,Int}
    n::Int
    metadata::DocumentMetadata
end

NGramDocument(ng::Dict{T, Int}, n::Int=DEFAULT_NGRAM_COMPLEXITY
             ) where T <: AbstractString =
    NGramDocument{T}(ng, n, DocumentMetadata())

NGramDocument{T}(txt::AbstractString,
                 dm::DocumentMetadata=DocumentMetadata(),
                 n::Int=DEFAULT_NGRAM_COMPLEXITY;
                 tokenizer::Symbol=DEFAULT_TOKENIZER
                ) where T<:AbstractString =
    NGramDocument(ngramize(dm.language, T.(tokenize(txt, method=tokenizer)), n), n, dm)

NGramDocument(txt::AbstractString,
              dm::DocumentMetadata=DocumentMetadata(),
              n::Int=DEFAULT_NGRAM_COMPLEXITY;
              tokenizer::Symbol=DEFAULT_TOKENIZER)=
    NGramDocument(ngramize(dm.language, tokenize(txt, method=tokenizer), n), n, dm)


# Union type that refers to a generic, non-abstract document type
const GenericDocument{T} = Union{
                                 FileDocument{T},
                                 StringDocument{T},
                                 TokenDocument{T},
                                 NGramDocument{T}
                                } where T<:AbstractString


# Easier Document() constructor that decides types based on inputs
Document(str::AbstractString) = isfile(str) ? FileDocument(str) : StringDocument(str)

Document(tkns::Vector{T}) where {T <: AbstractString} = TokenDocument(tkns)

Document(ng::Dict{String, Int}) = NGramDocument(ng)


"""
    text(d)

Access the text of document `d` if possible.
"""
text(d::AbstractString) = d

text(fd::FileDocument) = begin
    !isfile(fd.filename) && error("Can't find file: $(fd.filename)")
    read(fd.filename, String)
end

text(sd::StringDocument) = sd.text

text(td::TokenDocument) = begin
    @warn("TokenDocument's can only approximate the original text")
    join(td.tokens, " ")
end

text(ngd::NGramDocument) =
    error("The text of an NGramDocument cannot be reconstructed")

"""
    text!(d, new_text)

Replace the original text of document `d` with `new_text`.
"""
text!(sd::StringDocument{T}, new_text::T) where T<:AbstractString =
    (sd.text = new_text)

text!(d::AbstractDocument, new_text::AbstractString) =
    error("The text of a $(typeof(d)) cannot be edited")


"""
    tokens(d [; method=DEFAULT_TOKENIZER])

Access the tokens of document `d` as a token array. The `method` keyword
argument specifies the type of tokenization to perform. Available
options are `:slow` and `:fast`.
"""
tokens(d::AbstractString; method::Symbol=DEFAULT_TOKENIZER) =
    tokenize(d, method=method)

tokens(d::(Union{FileDocument, StringDocument}); method::Symbol=DEFAULT_TOKENIZER) =
    tokens(text(d), method=method)

tokens(d::TokenDocument; method::Symbol=DEFAULT_TOKENIZER) = d.tokens

tokens(d::NGramDocument; method::Symbol=DEFAULT_TOKENIZER) =
    error("The tokens of an NGramDocument cannot be reconstructed")


"""
    tokens!(d, new_tokens)

Replace the original tokens of document `d` with `new_tokens`.
"""
tokens!(d::TokenDocument{T}, new_tokens::Vector{T}) where T<:AbstractString =
    (d.tokens = new_tokens)

tokens!(d::AbstractDocument, new_tokens::Vector{T}) where T<:AbstractString =
    error("The tokens of a $(typeof(d)) cannot be directly edited")


"""
    ngrams(d, n=DEFAULT_GRAM_COMPLEXITY [; tokenizer=DEFAULT_TOKENIZER])

Access the document text of `d` as n-gram counts. The ngrams contain
at most `n` tokens which are obtained using `tokenizer`.
"""
ngrams(d::AbstractString,
       n::Int=DEFAULT_NGRAM_COMPLEXITY;
       tokenizer::Symbol=DEFAULT_TOKENIZER) =
    ngramize(DEFAULT_LANGUAGE, tokens(d, method=tokenizer), n)

ngrams(d::NGramDocument; tokenizer::Symbol=DEFAULT_TOKENIZER) = d.ngrams

ngrams(d::AbstractDocument,
       n::Int=DEFAULT_NGRAM_COMPLEXITY;
       tokenizer::Symbol=DEFAULT_TOKENIZER) =
    ngramize(language(d), tokens(d, method=tokenizer), n)

ngrams(d::Vector{S},
       n::Int=DEFAULT_NGRAM_COMPLEXITY;
       tokenizer::Symbol=DEFAULT_TOKENIZER) where S<:AbstractString =
    ngramize(DEFAULT_LANGUAGE, d, n)



"""
    ngrams!(d, new_ngrams)

Replace the original n-grams of document `d` with `new_ngrams`.
"""
ngrams!(d::NGramDocument{T}, new_ngrams::Dict{T, Int}) where T<:AbstractString =
    (d.ngrams = new_ngrams)

ngrams!(d::AbstractDocument, new_ngrams::Dict) =
    error("The n-grams of $(typeof(d)) cannot be directly edited")


# Length describes length of document in characters
Base.length(d::NGramDocument) =
    error("NGramDocument's do not have a well-defined length")

Base.length(d::AbstractDocument) = length(text(d))


# Length describes length of document in characters
ngram_complexity(ngd::NGramDocument) = ngd.n

ngram_complexity(d::AbstractDocument) =
    error("$(typeof(d))'s have no n-gram complexity")


# Conversion rules
Base.convert(::Type{FileDocument{T}}, d::FileDocument
            ) where T<:AbstractString =
    FileDocument(T.(d.filename), d.metadata)

Base.convert(::Type{StringDocument{T}}, d::Union{FileDocument, StringDocument}
            ) where T<:AbstractString =
    StringDocument(T.(text(d)), d.metadata)

Base.convert(::Type{TokenDocument{T}}, d::Union{TokenDocument, FileDocument, StringDocument}
            ) where T<:AbstractString =
    TokenDocument(T.(tokens(d)), d.metadata)

Base.convert(::Type{NGramDocument{T}}, d::Union{TokenDocument, FileDocument, StringDocument}
            ) where T<:AbstractString =
    NGramDocument(ngramize(language(d), T.(tokens(d))), DEFAULT_NGRAM_COMPLEXITY, d.metadata)

Base.convert(::Type{NGramDocument{T}}, d::NGramDocument{T2}
            ) where {T<:AbstractString, T2<:AbstractString} =
    NGramDocument(Dict{T, Int}(ngrams(d)), d.n, d.metadata)

# getindex() methods: StringDocument("This is text and that is not")["is"]
Base.getindex(d::AbstractDocument, term::AbstractString) = get(ngrams(d), term, 0)
