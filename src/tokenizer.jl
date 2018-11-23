"""
    sentence_tokenize([lang,] s)

Splits string `s` into sentences using `WordTokenizers.split_sentences`
function to perform the tokenization. If a language `lang` is provided,
it ignores it ;)
"""
sentence_tokenize(s::T) where T<:AbstractString =
    WordTokenizers.split_sentences(s)
sentence_tokenize(lang::S, s::T) where {S<:Language, T<:AbstractString} =
    WordTokenizers.split_sentences(s)


"""
    tokenize_slow([lang,] s)

Splits string `s` into tokens on whitespace using `WordTokenizers.tokenize`
function to perform the tokenization. If a language `lang` is provided,
it ignores it ;)
"""
tokenize_slow(s::T) where T<:AbstractString = WordTokenizers.tokenize(s)
tokenize_slow(lang::S, s::T) where {S<:Language, T<:AbstractString} = WordTokenizers.tokenize(s)


"""
    tokenize_fast(doc [;splitter])

Function that quickly tokenizes `doc` based on the splitting
pattern specified by `splitter::RegEx`.
Supported types for `doc` are: `AbstractString`, `Vector{AbstractString}`,
`StringDocument` and `NGramDocument`.
"""
tokenize_fast(doc::Vector{S}; splitter::Regex=DEFAULT_TOKENIZATION_REGEX
             ) where S<:AbstractString =
    return vcat((tokenize_fast(words) for words in doc)...)

tokenize_fast(doc::S; splitter::Regex=DEFAULT_TOKENIZATION_REGEX
             ) where S<:AbstractString = begin
    # First, split
    tokens = strip.(split(doc, splitter))
    # Filter out empty strings
    return filter!(!isempty, tokens)
end

tokenize_fast(doc::NGramDocument; splitter::Regex=DEFAULT_TOKENIZATION_REGEX) =
    tokenize_fast(collect(keys(doc.ngrams)))

tokenize_fast(doc::StringDocument; splitter::Regex=DEFAULT_TOKENIZATION_REGEX) =
    tokenize_fast(doc.text, splitter=splitter)

"""
    tokenize(s [;method])

Tokenizes based on either the `tokenize_slow` or `tokenize_fast`
functions.
"""
function tokenize(doc; method::Symbol=:slow)
    if method == :slow
        return tokenize_slow(doc)
    else
        return tokenize_fast(doc)
    end
end
