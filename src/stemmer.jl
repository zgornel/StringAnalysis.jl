function stem_all(stemmer::Stemmer,
                  lang::S,
                  sentence::AbstractString;
                  method=:stringanalysis) where S <: Language
    tokens = tokenize(sentence, method=method)
    stemmed = stem(stemmer, tokens)
    join(stemmed, ' ')
end

function stem!(stemmer::Stemmer, words::Vector{S}) where S<:AbstractString
    l::Int = length(words)
    @inbounds for idx in 1:l
        words[idx] = stem(stemmer, words[idx])
    end
end

function stem(stemmer::Stemmer, words::Vector{S}) where S<:AbstractString
    l::Int = length(words)
    ret = [words[i] for i in 1:l]
    stem!(stemmer, ret)
    return ret
end

# Stemming methods with implicit stemmer generated from language
function stem(sentence::AbstractString;
              language::Language=DEFAULT_LANGUAGE)
    stemmer = Stemmer(lowercase(Languages.english_name(language)))
    ret = stem_all(stemmer, language, sentence)
    release(stemmer)
    return ret
end

function stem!(words::Vector{S};
               language::Language=DEFAULT_LANGUAGE
              ) where S<:AbstractString
    stemmer = Stemmer(lowercase(Languages.english_name(language)))
    l::Int = length(words)
    @inbounds for idx in 1:l
        words[idx] = stem(stemmer, words[idx])
    end
    release(stemmer)
    return nothing
end

function stem(words::Vector{S};
              language::Language=DEFAULT_LANGUAGE
              ) where S<:AbstractString
    l::Int = length(words)
    ret = [words[i] for i in 1:l]
    stem!(ret, language=language)
    return ret
end

# Stemming methods for documents
function stemmer_for_document(d::AbstractDocument)
    Stemmer(lowercase(Languages.english_name(language(d))))
end

function stem!(d::AbstractDocument)
    stemmer = stemmer_for_document(d)
    stem!(stemmer, d)
    release(stemmer)
end

stem!(stemmer::Stemmer, d::FileDocument) = error("FileDocument cannot be modified")

function stem!(stemmer::Stemmer, d::StringDocument)
    stemmer = stemmer_for_document(d)
    d.text = stem_all(stemmer, language(d), d.text)
    nothing
end

function stem!(stemmer::Stemmer, d::TokenDocument)
    d.tokens = stem(stemmer, d.tokens)
    nothing
end

function stem!(stemmer::Stemmer, d::NGramDocument)
    for token in keys(d.ngrams)
        new_token = join(stem(stemmer, split(token)), " ")
        if new_token != token
            if haskey(d.ngrams, new_token)
                d.ngrams[new_token] = d.ngrams[new_token] + d.ngrams[token]
            else
                d.ngrams[new_token] = d.ngrams[token]
            end
            delete!(d.ngrams, token)
        end
    end
end

function stem!(crps::Corpus)
    stemmer = stemmer_for_document(crps.documents[1])
    for doc in crps
        stem!(stemmer, doc)
    end
    release(stemmer)
end
