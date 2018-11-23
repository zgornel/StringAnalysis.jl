const strip_patterns            = UInt32(0)
# Flags that activate function-based processors
const strip_corrupt_utf8        = UInt32(0x1) << 0
const strip_case                = UInt32(0x1) << 1
const strip_accents             = UInt32(0x1) << 2
const strip_punctuation         = UInt32(0x1) << 3
# Flags that activate function-based processors (external to this file)
const stem_words                = UInt32(0x1) << 7
# Flags that activate Regex based processors
const strip_whitespace          = UInt32(0x1) << 10
const strip_numbers             = UInt32(0x1) << 11
const strip_non_ascii           = UInt32(0x1) << 12
const strip_single_chars        = UInt32(0x1) << 13
const strip_html_tags           = UInt32(0x1) << 14
# Word list based
const strip_indefinite_articles = UInt32(0x1) << 20
const strip_definite_articles   = UInt32(0x1) << 21
const strip_prepositions        = UInt32(0x1) << 22
const strip_pronouns            = UInt32(0x1) << 23
const strip_stopwords           = UInt32(0x1) << 24
const strip_articles            = (strip_indefinite_articles |
                                   strip_definite_articles)
const strip_sparse_terms        = UInt32(0x1) << 25
const strip_frequent_terms      = UInt32(0x1) << 26

# RegEx Expressions for various stripping flags
# Format: flag => (match=>replacement)
const REGEX_CACHE = Dict{UInt32,Regex}(
    strip_whitespace => r"\s+",
    strip_numbers => r"\d+",
    strip_non_ascii => r"[^a-zA-Z\s]",
    strip_single_chars => r"[\s]+[A-Za-z]{1}[\s]+",
    strip_html_tags => r"(<script\b[^>]*>([\s\S]*?)</script>|<[^>]*>)"
)


# Basic string processing functions
# Remove corrupt UTF8 characters
remove_corrupt_utf8(s::AbstractString) = begin
    return map(x->isvalid(x) ? x : ' ', s)
end

# Conversion to lowercase
remove_case(s::T) where T<:AbstractString = lowercase(s)

# Removing accents
remove_accents(s::T) where T<:AbstractString =
    Unicode.normalize(s, stripmark=true)

# Remove punctuation
remove_punctuation(s::T) where T<:AbstractString =
    filter(x->!ispunct(x), s)

# Generate automatically functions for various Document types and Corpus
# Note: One has to add a simple method for `AbstractString` and the name
#       of the function in the `for` container to generate all needed
#       methods
# Note2: There is no distinction made for TokenDocument, NGramDocument
#        in the case of `remove_punctuation` because, depending on the
#        tokenization function, punctuation may creep up into the document
#        i.e. ngrams/tokens containing tags, punctuation.
for fname in [:remove_corrupt_utf8, :remove_case, :remove_accents, :remove_punctuation]
    # File document
    # TODO(Corneliu): Make these work on file documents
    #                 i.e. load file, process, re-write
    definition = """
        $(fname)!(d::FileDocument) = error("FileDocument cannot be modified.")
        """
    eval(Meta.parse(definition))
    # String Document
    definition = """
        function $(fname)!(d::StringDocument)
            d.text = $(fname)(d.text)
            return nothing
        end
        """
    eval(Meta.parse(definition))
    # Token Document
    definition = """
        function $(fname)!(d::TokenDocument)
            @inbounds for i in 1:length(d.tokens)
                d.tokens[i] = $(fname)(d.tokens[i])
            end
        end
        """
    eval(Meta.parse(definition))
    # NGramDocument
    definition = """
        function $(fname)!(d::NGramDocument{S}) where S
            _ngrams = Dict{S, Int}()
            for token in keys(d.ngrams)
                _token = $(fname)(token)
                _ngrams[_token] = get(_ngrams, _token, 0) + 1
            end
            d.ngrams = _ngrams
            return nothing
        end
        """
    eval(Meta.parse(definition))
    # Corpus
    definition = """
        function $(fname)!(crps::Corpus)
            for doc in crps
                $(fname)!(doc)
            end
        end
        """
    eval(Meta.parse(definition))
end

# The `remove_patterns` methods cannot be generated in the loop, different signature
remove_patterns(s::AbstractString, rex::Regex, rep="") =
    replace(s, rex => rep)

remove_patterns!(d::FileDocument, rex::Regex, rep="") = error("FileDocument cannot be modified.")

remove_patterns!(d::StringDocument, rex::Regex, rep="") = begin
    d.text = remove_patterns(d.text, rex, rep)
    nothing
end

remove_patterns!(d::TokenDocument, rex::Regex, rep="") = begin
    for i in 1:length(d.tokens)
        d.tokens[i] = remove_patterns(d.tokens[i], rex, rep)
    end
end

remove_patterns!(d::NGramDocument{S}, rex::Regex, rep="") where S = begin
    _ngrams = Dict{S, Int}()
    for token in keys(d.ngrams)
        _token = remove_patterns(token, rex, rep)
        _ngrams[_token] = get(_ngrams, _token, 0) + 1
    end
    d.ngrams = _ngrams
    return nothing
end

function remove_patterns!(crps::Corpus, rex::Regex, rep="")
    for doc in crps
        remove_patterns!(doc, rex, rep)
    end
end


# Remove specified words
function remove_words!(entity, words::Vector{T}) where T<: AbstractString
    skipwords = Set{T}()
    union!(skipwords, words)
    prepare!(entity, strip_patterns, skip_words = skipwords)
end



const alpha_sparse = 0.05
const alpha_frequent = 0.95
# TODO(Corneliu) get sparse terms for Document types, AbstractString

# Drop terms based on frequency
function sparse_terms(crps::Corpus, alpha = alpha_sparse)
    res = Vector{String}(undef, 0)
    ndocs = length(crps.documents)
    for term in keys(crps.lexicon)
        f = length(crps.inverse_index[term]) / ndocs
        if f <= alpha
            push!(res, String(term))
        end
    end
    return res
end

function frequent_terms(crps::Corpus, alpha = alpha_frequent)
    res = Vector{String}(undef, 0)
    ndocs = length(crps.documents)
    for term in keys(crps.lexicon)
        f = length(crps.inverse_index[term]) / ndocs
        if f >= alpha
            push!(res, String(term))
        end
    end
    return res
end

_build_words_pattern(words::Set{T}) where T<:AbstractString =
    Regex(ifelse(isempty(words), "", "\\b("* join(words,"|","|") *")\\b"))

_build_regex_pattern(regexes::Set{T}) where T<:Regex = begin
    l = length(regexes)
    if l == 0
        return r""
    elseif l == 1
        return pop!(regexes)
    else
        iob = IOBuffer()
        write(iob, "($(pop!(regexes).pattern))")
        for re in regexes
            write(iob, "|($(re.pattern))")
        end
        return Regex(String(take!(iob)))
    end
end



### # Remove parts from document based on flags or regular expressions
### function prepare!(crps::Corpus, flags::UInt32; skip_patterns = Set{AbstractString}(), skip_words = Set{AbstractString}())
###     ((flags & strip_sparse_terms) > 0) && union!(skip_words, sparse_terms(crps))
###     ((flags & strip_frequent_terms) > 0) && union!(skip_words, frequent_terms(crps))
### 
###     ((flags & strip_corrupt_utf8) > 0) && remove_corrupt_utf8!(crps)
###     ((flags & strip_case) > 0) && remove_case!(crps)
###     ((flags & strip_accents) > 0) && remove_accents!(crps)
###     ((flags & strip_punctuation) > 0) && remove_punctuation!(d)
### 
###     lang = language(crps.documents[1])   # assuming all documents are of the same language - practically true
###     r = _build_regex(lang, flags, skip_patterns, skip_words)
###     !isempty(r.pattern) && remove_patterns!(crps, r)
### 
###     ((flags & stem_words) > 0) && stem!(crps)
###     nothing
### end

function prepare!(entity, flags::UInt32; skip_patterns = Set{AbstractString}(), skip_words = Set{AbstractString}())
    ((flags & strip_sparse_terms) > 0) && union!(skip_words, sparse_terms(entity))
    ((flags & strip_frequent_terms) > 0) && union!(skip_words, frequent_terms(entity))

    ((flags & strip_corrupt_utf8) > 0) && remove_corrupt_utf8!(entity)
    ((flags & strip_case) > 0) && remove_case!(entity)
    ((flags & strip_accents) > 0) && remove_accents!(entity)
    ((flags & strip_punctuation) > 0) && remove_punctuation!(entity)
    ###r = _build_regex(language(d), flags, skip_patterns, skip_words)
    ###!isempty(r.pattern) && remove_patterns!(d, r)
    ((flags & stem_words) > 0) && stem!(d)
    nothing
end

function prepare(s::AbstractString,
                 flags::UInt32;
                 lang::Language = DEFAULT_LANGUAGE,
                 skip_patterns = Set{Regex}(),
                 skip_words = Set{AbstractString}())
    os = s  # Initialize output string
    # function-based stripping
    ((flags & strip_corrupt_utf8) > 0) && (os = remove_corrupt_utf8(os))
    ((flags & strip_case) > 0)         && (os = remove_case(os))
    ((flags & strip_accents) > 0)      && (os = remove_accents(os))
    ((flags & strip_punctuation) > 0)  && (os = remove_punctuation(os))
    # regex-based stripping
    rpatterns = Set{Regex}()  # patterns to remove
    ((flags & strip_whitespace) > 0) && push!(rpatterns, REGEX_CACHE[strip_whitespace])
    if (flags & strip_non_ascii) > 0
        push!(rpatterms, REGEX_CACHE[strip_non_ascii])
    else
        ((flags & strip_numbers) > 0) && push!(rpatterns, REGEX_CACHE[strip_numbers])
        ((flags & strip_single_chars) > 0) && push!(rpatterns, REGEX_CACHE[strip_single_chars])
    end
    # known word-based stripping
    if (flags & strip_articles) > 0
        union!(skip_words, articles(lang))
    else
        ((flags & strip_indefinite_articles) > 0) && union!(skip_words, indefinite_articles(lang))
        ((flags & strip_definite_articles) > 0) && union!(skip_words, definite_articles(lang))
    end
    ((flags & strip_prepositions) > 0) && union!(skip_words, prepositions(lang))
    ((flags & strip_pronouns) > 0) && union!(skip_words, pronouns(lang))
    ((flags & strip_stopwords) > 0) && union!(skip_words, stopwords(lang))
    if !isempty(skip_words)
        push!(rpatterns, _build_words_pattern(skip_words))
    end
    # custom regex-ping
    if !isempty(skip_patterns)
        push!(rpatterns, _build_regex_pattern(skip_patterns))
    end
    # Do filterning
    r = _build_regex_pattern(rpatterns)
    os = remove_patterns(os, r)
    # Stemming
    ((flags & stem_words) > 0) && (os = stem(os))
    return os
end
