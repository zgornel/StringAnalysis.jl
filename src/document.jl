# All Document types share a common metadata profile using DocumentMetadata
mutable struct DocumentMetadata
    language
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

FileDocument(f::AbstractString) = begin
    d = FileDocument(String(f), DocumentMetadata())
    d.metadata.name = f
    return d
end


# StringDocument type and constructors
mutable struct StringDocument{T<:AbstractString} <: AbstractDocument{T}
    text::T
    metadata::DocumentMetadata
end

StringDocument(txt::AbstractString) = StringDocument(txt, DocumentMetadata())


# TokenDocument type and constructors
mutable struct TokenDocument{T<:AbstractString} <: AbstractDocument{T}
    tokens::Vector{T}
    metadata::DocumentMetadata
end

TokenDocument(txt::AbstractString, dm::DocumentMetadata) =
    TokenDocument(tokenize(String(txt)), dm)

TokenDocument(tkns::Vector{T}) where T <: AbstractString =
    TokenDocument(tkns, DocumentMetadata())

TokenDocument(txt::AbstractString) = TokenDocument(String(txt), DocumentMetadata())


# NGramDocument type and constructors
mutable struct NGramDocument{T<:AbstractString} <: AbstractDocument{T}
    ngrams::Dict{T,Int}
    n::Int
    metadata::DocumentMetadata
end

NGramDocument(txt::AbstractString, dm::DocumentMetadata, n::Integer=1) =
    NGramDocument(ngramize(dm.language, tokenize(String(txt)), n), n, dm)

NGramDocument(txt::AbstractString, n::Integer=1) =
    NGramDocument(txt, DocumentMetadata(), n)

NGramDocument(ng::Dict{T, Int}, n::Integer=1) where T <: AbstractString =
    NGramDocument(ng, n, DocumentMetadata())


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



# text() / text!(): Access to document text as a string
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

text!(sd::StringDocument, new_text::AbstractString) = (sd.text = new_text)

text!(d::AbstractDocument, new_text::AbstractString) =
    error("The text of a $(typeof(d)) cannot be edited")


# tokens() / tokens!(): Access to document text as a token array
tokens(d::(Union{FileDocument, StringDocument})) = tokenize(text(d))

tokens(d::TokenDocument) = d.tokens

tokens(d::NGramDocument) =
    error("The tokens of an NGramDocument cannot be reconstructed")

tokens!(d::TokenDocument, new_tokens::Vector{T}) where {T <: AbstractString} = (d.tokens = new_tokens)

tokens!(d::AbstractDocument, new_tokens::Vector{T}) where T <: AbstractString =
    error("The tokens of a $(typeof(d)) cannot be directly edited")


# ngrams() / ngrams!(): Access to document text as n-gram counts
ngrams(d::NGramDocument, n::Integer) =
    error("The n-gram complexity of an NGramDocument cannot be increased")

ngrams(d::AbstractDocument, n::Integer) = ngramize(language(d), tokens(d), n)

ngrams(d::NGramDocument) = d.ngrams

ngrams(d::AbstractDocument) = ngrams(d, 1)

ngrams!(d::NGramDocument, new_ngrams::Dict{AbstractString, Int}) = (d.ngrams = new_ngrams)

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
Base.convert(::Type{StringDocument{T}}, d::FileDocument{T}
            ) where T<:AbstractString =
    StringDocument(text(d), d.metadata)

Base.convert(::Type{TokenDocument{T}}, d::(Union{FileDocument{T}, StringDocument{T}})
            ) where T<:AbstractString =
    TokenDocument(T.(tokens(d)), d.metadata)

Base.convert(::Type{NGramDocument{T}},
             d::(Union{FileDocument{T}, StringDocument{T}, TokenDocument{T}})
            ) where T<:AbstractString=
    NGramDocument(T.(ngrams(d)), 1, d.metadata)


# getindex() methods: StringDocument("This is text and that is not")["is"]
Base.getindex(d::AbstractDocument, term::AbstractString) = ngrams(d)[term]
