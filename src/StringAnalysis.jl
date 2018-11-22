module StringAnalysis

    using SparseArrays
    using LinearAlgebra
    using Languages
    using WordTokenizers

    import Base: depwarn, show, summary, names
    import Languages: name

    export AbstractDocument,
           Document, FileDocument, StringDocument,
           TokenDocument, NGramDocument, GenericDocument
    export DocumentMetadata, metadata
    export Corpus, DirectoryCorpus, documents, standardize!
    export lexicon, update_lexicon!, lexical_frequency, lexicon_size
    export inverse_index, update_inverse_index!, index_size
    export stemmer_types, Stemmer
    export stem!, stem
    export text, tokens, ngrams
    export text!, tokens!, ngrams!
    export ngram_complexity
    export TextHashFunction, index_hash, cardinality,
           hash_function, hash_function!
    export hash_dtv, each_hash_dtv, hash_dtm, hash_tdm
    export DocumentTermMatrix
    export dtv, each_dtv, dtm, tdm
    export tf!, tf
    export tf_idf!, tf_idf
    export lsa!, lsa
    export lda!, lda
    export frequent_terms, sparse_terms, prepare!,
           strip_patterns, strip_corrupt_utf8, strip_case, stem_words,
           strip_whitespace, strip_punctuation,
           strip_numbers, strip_non_letters, strip_indefinite_articles,
           strip_definite_articles, strip_articles, strip_prepositions,
           strip_pronouns, strip_stopwords, strip_sparse_terms,
           strip_frequent_terms, strip_html_tags,
           remove_words!, remove_patterns!, remove_patterns,
           remove_frequent_terms!, remove_sparse_terms!

    # Include section
    include("defaults.jl")
    include("hash.jl")
    include("document.jl")
    include("corpus.jl")
    include("metadata.jl")
    include("stemmer.jl")
    include("tokenizer.jl")
    include("ngramizer.jl")
    include("dtm.jl")
    include("stats.jl")
    include("lsa.jl")
    include("lda.jl")
    include("preprocessing.jl")
    include("show.jl")
    ###
    # Load libstemmer from our deps.jl
    const depsjl_path = joinpath(dirname(@__FILE__), "..", "deps", "deps.jl")
    if !isfile(depsjl_path)
        error("Snowball Stemmer not installed properly, " *
              "run Pkg.build(\"StringAnalysis\"), restart " *
              "Julia and try again.")
    end
    include(depsjl_path)
    ###
end
