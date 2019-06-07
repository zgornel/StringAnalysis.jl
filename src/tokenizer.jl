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
    tokenize_default([lang,] s)

Splits string `s` into tokens on whitespace using `WordTokenizers.tokenize`
function to perform the tokenization. If a language `lang` is provided,
it ignores it ;)
"""
tokenize_default(s::T) where T<:AbstractString = DEFAULT_WT_TOKENIZER(s)
tokenize_default(lang::S, s::T) where {S<:Language, T<:AbstractString} = DEFAULT_WT_TOKENIZER(s)


"""
    tokenize_stringanalysis(doc [;splitter])

Function that quickly tokenizes `doc` based on the splitting
pattern specified by `splitter::RegEx`.
Supported types for `doc` are: `AbstractString`, `Vector{AbstractString}`,
`StringDocument` and `NGramDocument`.
"""
tokenize_stringanalysis(doc::S; splitter::Regex=DEFAULT_TOKENIZATION_REGEX
                       ) where S<:AbstractString =
    strip.(split(doc, splitter, keepempty=false))

tokenize_stringanalysis(doc::Vector{S}; splitter::Regex=DEFAULT_TOKENIZATION_REGEX
                       ) where S<:AbstractString =
    return vcat((tokenize_stringanalysis(words, splitter=splitter) for words in doc)...)

tokenize_stringanalysis(doc::NGramDocument; splitter::Regex=DEFAULT_TOKENIZATION_REGEX) =
    unique!(tokenize_stringanalysis(collect(keys(doc.ngrams)), splitter=splitter))

tokenize_stringanalysis(doc::StringDocument; splitter::Regex=DEFAULT_TOKENIZATION_REGEX) =
    tokenize_stringanalysis(doc.text, splitter=splitter)


""""
    tokenize(doc [;method, splitter])

Tokenizes the document `doc` based on the `mehtod` (default `:default`, i.e.
a `WordTokenizers.jl` tokenizer) and the `splitter`, which is a `Regex` used
if `method=:stringanalysis`.
"""
function tokenize(doc;
                  method::Symbol=DEFAULT_TOKENIZER,
                  splitter::Regex=DEFAULT_TOKENIZATION_REGEX)
    if method == :stringanalysis
        return tokenize_stringanalysis(doc, splitter=splitter)
    else
        return tokenize_default(doc)
    end
end
