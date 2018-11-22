# Split string into tokens on whitespace
tokenize(s::T) where T<:AbstractString = WordTokenizers.tokenize(s)
tokenize(lang::S, s::T) where {S<:Language, T<:AbstractString} = WordTokenizers.tokenize(s)

sentence_tokenize(s::T) where T<:AbstractString = WordTokenizers.split_sentences(s)
sentence_tokenize(lang::S, s::T) where {S<:Language, T<:AbstractString} = WordTokenizers.split_sentences(s)



"""
    tokenize_fast(text [;splitter])

Function that quickly tokenizes the `text` based on the splitting
pattern specified by `splitter::RegEx`. Supported types for`text`
are: `AbstractString`, `Vector{AbstractString}`, `StringDocument`
and `NGramDocument`.
"""
tokenize_fast(doc::Vector{S}; splitter::Regex=DEFAULT_TOKENIZATION_REGEX
             ) where S<:AbstractString =
    return vcat((tokenize_fast(words) for words in doc)...)

tokenize_fast(doc::S; splitter::Regex=DEFAULT_TOKENIZATION_REGEX
             ) where S<:AbstractString = begin
    # First, split
    tokens = String.(strip.(split(doc, splitter)))
    # Filter out empty strings
    return filter!(!isempty, tokens)
end

tokenize_fast(doc::NGramDocument; splitter::Regex=DEFAULT_TOKENIZATION_REGEX) =
    tokenize_fast(collect(keys(doc.ngrams)))

tokenize_fast(doc::StringDocument; splitter::Regex=DEFAULT_TOKENIZATION_REGEX) =
    tokenize_fast(doc.text, splitter=splitter)
