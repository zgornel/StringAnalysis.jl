#const libstemmer = joinpath(dirname(@__FILE__),"..","deps","usr","lib", "libstemmer."*Libdl.dlext)
#@BinDeps.load_dependencies [:libstemmer=>:libstemmer]

##
# character encodings supported by libstemmer
const UTF_8         = "UTF_8"
const ISO_8859_1    = "ISO_8859_1"
const CP850         = "CP850"
const KOI8_R        = "KOI8_R"

function stem_all(stemmer::Stemmer, lang::S, sentence::AbstractString) where S <: Language
    tokens = StringAnalysis.tokenize(lang, sentence)
    stemmed = stem(stemmer, tokens)
    join(stemmed, ' ')
end

function stem(stemmer::Stemmer, words::Array)
    l::Int = length(words)
    ret = Array{String}(undef, l)
    for idx in 1:l
        ret[idx] = stem(stemmer, words[idx])
    end
    ret
end

function stemmer_for_document(d::AbstractDocument)
    Stemmer(lowercase(name(language(d))))
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
        new_token = stem(stemmer, token)
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
