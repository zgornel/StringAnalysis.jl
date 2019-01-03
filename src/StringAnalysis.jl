####################################################################################
#  .dolokc          '                        :O                                    #
#  kl    c         :N..        ..  '.        .'         .. .'.           .'. ..    #
#  'xddc,          :X..        .kO..c.       ;N         .Xl..ko         dx..'Xc    #
#  .   .:N.        'X           dd           'N          X'  cO        .M,   0;    #
#  xl'..ck         .X.;,       .Ok'         .cW,        'Nl .xK.        ck'.,X;    #
#    ...    ......   .  ......        ......     ......          ...... .....K.    #
#          .......      .......      .......     ......         .......  ,'''      #
#                               .                   ',.              ..            #
#                              kMx       .    ...   ,Mc         ..   do   ...      #
#                             o.dMl  ,MO.xW' ,;,kN. ,Mc dMc o,cWl,c..XN .Wx;:'     #
#                            c,.'kM; .Mx cM: KX.oM: ,Mc  dWo' .:cOW: KN .::dMo ,,  #
#                           .,   .:;..:;.,:, 'c,':, ':,   k,  .,.,, .;:. ,.';. ,;  #
#                                                       dlc                        #
#                                                                                  #
####################################################################################

# StringAnalysis.jl - Library for analyzing text written at 0x0α Research
#                     by Corneliu Cofaru, 2018

module StringAnalysis
    # Using
    using Unicode
    using SparseArrays
    using LinearAlgebra
    using Statistics
    using Languages
    using WordTokenizers

    # Imports
    import Base: size, show, summary, names
    import Languages: name

    # Exports
    export AbstractDocument, Document, FileDocument, StringDocument,
           TokenDocument, NGramDocument, GenericDocument,
           DocumentMetadata, metadata,
           text!, text, tokens!, tokens, ngrams!, ngrams,
           ngram_complexity
    export Corpus, DirectoryCorpus, documents, standardize!,
           lexicon, update_lexicon!, lexical_frequency, lexicon_size,
           inverse_index, update_inverse_index!, index_size
    export DocumentTermMatrix, dtv, each_dtv, dtm, tdm
    export hash_dtv, each_hash_dtv, hash_dtm, hash_tdm
    export TextHashFunction, index_hash, cardinality,
           hash_function, hash_function!
    export Stemmer, stem!, stem, stemmer_types
    export tokenize, tokenize_fast, tokenize_slow, sentence_tokenize
    export tf!, tf, tf_idf!, tf_idf, bm_25!, bm_25
    export LSAModel, lsa, embed_document, embed_word, get_vector, index,
           cosine, similarity
    export lda
    export frequent_terms, sparse_terms,
           prepare!, prepare,
           strip_patterns, strip_corrupt_utf8, strip_case,
           strip_accents, strip_punctuation, stem_words,
           strip_whitespace, strip_numbers, strip_non_ascii,
           strip_single_chars, strip_html_tags,
           strip_indefinite_articles, strip_definite_articles,
           strip_articles, strip_prepositions,
           strip_pronouns, strip_stopwords,
           strip_sparse_terms, strip_frequent_terms,
           strip_everything, strip_everything_stem

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
    # Load libstemmer from our deps.jl
    const depsjl_path = joinpath(dirname(@__FILE__), "..", "deps", "deps.jl")
    if !isfile(depsjl_path)
        error("Snowball Stemmer not installed properly, " *
              "run Pkg.build(\"StringAnalysis\"), restart " *
              "Julia and try again.")
    end
    include(depsjl_path)
end
