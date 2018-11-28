const strip_patterns            = UInt32(0)
# Flags that activate function-based processors
const strip_corrupt_utf8        = UInt32(0x1) << 0
const strip_case                = UInt32(0x1) << 1
const strip_accents             = UInt32(0x1) << 2
const strip_html_tags           = UInt32(0x1) << 3
# Flags that activate function-based processors (external to this file)
const stem_words                = UInt32(0x1) << 7
# Flags that activate Regex based processors
const strip_punctuation         = UInt32(0x1) << 9
const strip_whitespace          = UInt32(0x1) << 10
const strip_numbers             = UInt32(0x1) << 11
const strip_non_ascii           = UInt32(0x1) << 12
const strip_single_chars        = UInt32(0x1) << 13
# Word list based
const strip_indefinite_articles = UInt32(0x1) << 20
const strip_definite_articles   = UInt32(0x1) << 21
const strip_prepositions        = UInt32(0x1) << 22
const strip_pronouns            = UInt32(0x1) << 23
const strip_stopwords           = UInt32(0x1) << 24
const strip_sparse_terms        = UInt32(0x1) << 25
const strip_frequent_terms      = UInt32(0x1) << 26

# Generate custom flag combinations (bit-shift)
function flag_generate(bs::Vector{<:Integer})
    b = UInt32(1)
    n = UInt32(0)
    for s in bs
        n|= (b << s)
    end
    return n
end

# Generate custom flag combinations (bit-toogle)
flag_generate(flags::UInt32...) = reduce(|, flags)

# Compound stripping flags
const strip_articles = flag_generate(
                        strip_indefinite_articles,
                        strip_definite_articles)

const strip_everything = flag_generate(
                            [0,1,2,3,
                            9,10,11,12,13,
                            20,21,22,23,24,25,26])
const strip_everything_stem = flag_generate(
                                [0,1,2,3,
                                 7,
                                 9,10,11,12,13,
                                 20,21,22,23,24,25,26])

# RegEx Expressions for various stripping flags
# Format: flag => (match=>replacement)
const strip2regex = Dict{UInt32,Regex}(
    strip_whitespace => r"[\s]+",
    strip_numbers => r"\d+",
    strip_non_ascii => r"[^a-zA-Z\s]",
    strip_single_chars => r"(\s|\b)[\w]{1}(\b|\s)",
    #strip_html_tags => r"(<script\b[^>]*>([\s\S]*?)</script>|<[^>]*>)",
    #strip_punctuation =>r"[^\d\w\s\b]+"
    strip_punctuation => r"[!\"#$%&\'()*+,-./:;<=>?@\[\\\]^_`\{\|\}~]+"
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

# Remove HTML tags
remove_html_tags(s::T) where T<:AbstractString =
    replace(s, r"(<script\b[^>]*>([\s\S]*?)</script>|<[^>]*>)" => ' ')

# Generate automatically functions for various Document types and Corpus
# Note: One has to add a simple method for `AbstractString` and the name
#       of the function in the `for` container to generate all needed
#       methods
for fname in [:remove_corrupt_utf8, :remove_case, :remove_accents, :remove_html_tags]
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
            filter!(t->(!isempty(t) && isvalid(t[1])), d.tokens)
        end
        """
    eval(Meta.parse(definition))
    # NGramDocument
    definition = """
        function $(fname)!(d::NGramDocument{S}) where S
            _ngrams = Dict{S, Int}()
            for token in keys(d.ngrams)
                _token = $(fname)(token)
                if !isempty(_token)
                    _ngrams[_token] = get(_ngrams, _token, 0) + 1
                end
            end
            filter!(p->isvalid(p.first[1]), _ngrams)
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



"""
    remove_patterns(s, rex)

Removes from the string `s` the text matching the pattern described
by the regular expression `rex`.
"""
function remove_patterns(s::AbstractString, rex::Regex)
    iob = IOBuffer()
    ibegin = 1
    v=codeunits(s)
    for m in eachmatch(rex, s)
        len = m.match.offset-ibegin+1
        if len > 0
            Base.write_sub(iob, v, ibegin, len)
            write(iob, ' ')
        end
        ibegin = nextind(s, lastindex(m.match)+m.match.offset)
    end
    len = length(v) - ibegin + 1
    (len > 0) && Base.write_sub(iob, v, ibegin, len)
    String(take!(iob))
end

function remove_patterns(s::SubString{T}, rex::Regex) where T <: String
    iob = IOBuffer()
    ioffset = s.offset
    data = codeunits(s.string)
    ibegin = 1
    for m in eachmatch(rex, s)
        len = m.match.offset-ibegin+1
        if len > 0
            Base.write_sub(iob, data, ibegin+ioffset, len)
            write(iob, ' ')
        end
        ibegin = nextind(s, lastindex(m.match)+m.match.offset)
    end
    len = lastindex(s) - ibegin + 1
    (len > 0) && Base.write_sub(iob, data, ibegin+ioffset, len)
    String(take!(iob))
end

"""
    remove_patterns!(d, rex)

Removes from the document or corpus `d` the text matching the pattern described
by the regular expression `rex`.
"""
remove_patterns!(d::FileDocument, rex::Regex) = error("FileDocument cannot be modified.")

remove_patterns!(d::StringDocument, rex::Regex) = begin
    d.text = remove_patterns(d.text, rex)
    nothing
end

remove_patterns!(d::TokenDocument, rex::Regex) = begin
    @inbounds for i in 1:length(d.tokens)
        d.tokens[i] = remove_patterns(d.tokens[i], rex)
    end
    filter!(t->(!isempty(t) && isvalid(t[1])), d.tokens)
end

remove_patterns!(d::NGramDocument{S}, rex::Regex) where S = begin
    _ngrams = Dict{S, Int}()
    for token in keys(d.ngrams)
        _token = remove_patterns(token, rex)
        if !isempty(_token)
            _ngrams[_token] = get(_ngrams, _token, 0) + 1
        end
    end
    filter!(p->isvalid(p.first[1]), _ngrams)
    d.ngrams = _ngrams
    return nothing
end

function remove_patterns!(crps::Corpus, rex::Regex)
    for doc in crps
        remove_patterns!(doc, rex)
    end
end


# Remove specified words
function remove_words!(entity, words::Vector{T}) where T<: AbstractString
    skipwords = Vector{T}()
    union!(skipwords, words)
    prepare!(entity, strip_patterns, skip_words = skipwords)
end



const alpha_sparse = 0.05
const alpha_frequent = 0.95

# Returns a vector with rare terms among all documents
function sparse_terms(crps::Corpus, alpha = alpha_sparse)
    #isempty(crps.inverse_index) && @warn "Inverse index is empty."
    #isempty(crps.lexicon) && @warn "Lexicon is empty."
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

# Returns a vector with frequent terms among all documents
function frequent_terms(crps::Corpus, alpha = alpha_frequent)
    #isempty(crps.inverse_index) && @warn "Inverse index is empty."
    #isempty(crps.lexicon) && @warn "Lexicon is empty."
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

# TODO(Corneliu): Implement for Documents
function sparse_terms(doc::AbstractDocument, alpha = alpha_sparse)
    String[]
end

function frequent_terms(doc::AbstractDocument, alpha = alpha_frequent)
    String[]
end


# Function that builds a regex out of a set of strings
_build_words_pattern(words::Vector{T}) where T<:AbstractString =
    Regex(ifelse(isempty(words), "", "\\b("* join(words,"|","|") *")\\b"))

# Function that builds a big regex out of a set of regexes
_build_regex_pattern(regexes::Vector{T}) where T<:Regex = begin
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

# Get the language
_language(doc::AbstractDocument) = doc.metadata.language
_language(crps::Corpus) = begin
    length(crps) < 1 && return DEFAULT_LANGUAGE
    return _language(crps[1])
end

function prepare!(entity,  # can be an AbstractDocument or Corpus
                  flags::UInt32;
                  skip_patterns = Vector{Regex}(),
                  skip_words = Vector{AbstractString}())
    # Do function-based stripping
    ((flags & strip_corrupt_utf8) > 0) && remove_corrupt_utf8!(entity)
    ((flags & strip_html_tags) > 0) && remove_html_tags!(entity)
    ((flags & strip_case) > 0) && remove_case!(entity)
    ((flags & strip_accents) > 0) && remove_accents!(entity)
    # regex
    rpatterns = Vector{Regex}(undef, 0)  # patterns to remove
    ((flags & strip_whitespace) > 0) && push!(rpatterns, strip2regex[strip_whitespace])
    if (flags & strip_non_ascii) > 0
        push!(rpatterns, strip2regex[strip_non_ascii])
    else
        ((flags & strip_numbers) > 0) && push!(rpatterns, strip2regex[strip_numbers])
        ((flags & strip_punctuation) > 0) && push!(rpatterns, strip2regex[strip_punctuation])
        ((flags & strip_single_chars) > 0) && push!(rpatterns, strip2regex[strip_single_chars])
    end
    # known words
    lang = _language(entity)
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
    # sparse, frequent terms
    ((flags & strip_sparse_terms) > 0) && union!(skip_words, sparse_terms(entity))
    ((flags & strip_frequent_terms) > 0) && union!(skip_words, frequent_terms(entity))
    # custom regex
    if !isempty(skip_patterns)
        push!(rpatterns, _build_regex_pattern(skip_patterns))
    end
    # Do regex-based stripping
    if !isempty(rpatterns)
        r = _build_regex_pattern(rpatterns)
        remove_patterns!(entity, r)
    end
    # Stemming
    ((flags & stem_words) > 0) && stem!(entity)
    nothing
end

function prepare(s::AbstractString,
                 flags::UInt32;
                 lang::Language = DEFAULT_LANGUAGE,
                 skip_patterns = Vector{Regex}(),
                 skip_words = Vector{AbstractString}())
    os = s  # Initialize output string
    # Do function-based stripping
    ((flags & strip_corrupt_utf8) > 0) && (os = remove_corrupt_utf8(os))
    ((flags & strip_html_tags) > 0)    && (os = remove_html_tags(os))
    ((flags & strip_case) > 0)         && (os = remove_case(os))
    ((flags & strip_accents) > 0)      && (os = remove_accents(os))
    # regex
    rpatterns = Vector{Regex}(undef, 0)  # patterns to remove
    ((flags & strip_whitespace) > 0) && push!(rpatterns, strip2regex[strip_whitespace])
    if (flags & strip_non_ascii) > 0
        push!(rpatterns, strip2regex[strip_non_ascii])
    else
        ((flags & strip_numbers) > 0) && push!(rpatterns, strip2regex[strip_numbers])
        ((flags & strip_punctuation) > 0) && push!(rpatterns, strip2regex[strip_punctuation])
        ((flags & strip_single_chars) > 0) && push!(rpatterns, strip2regex[strip_single_chars])
    end
    # known words
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
    # sparse, frequent terms
    #((flags & strip_sparse_terms) > 0) && union!(skip_words, sparse_terms(os))
    #((flags & strip_frequent_terms) > 0) && union!(skip_words, frequent_terms(os))
    # custom regex
    if !isempty(skip_patterns)
        push!(rpatterns, _build_regex_pattern(skip_patterns))
    end
    # Do regex-based stripping
    if !isempty(rpatterns)
        r = _build_regex_pattern(rpatterns)
        os = remove_patterns(os, r)
    end
    # Stemming
    ((flags & stem_words) > 0) && (os = stem(os))
    return os
end
