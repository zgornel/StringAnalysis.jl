var documenterSearchIndex = {"docs": [

{
    "location": "#",
    "page": "Introduction",
    "title": "Introduction",
    "category": "page",
    "text": "CurrentModule=StringAnalysis"
},

{
    "location": "#Introduction-1",
    "page": "Introduction",
    "title": "Introduction",
    "category": "section",
    "text": "StringAnalysis is a package for working with strings and text. It is a hard-fork from TextAnalysis.jl designed to provide a more powerful, faster and orthogonal API.Note: This documentation is still under construction and incomplete. For an overview of the basic capabilities of the package, consult the - still relevant - TextAnalysis.jl documentation."
},

{
    "location": "#What-is-new?-1",
    "page": "Introduction",
    "title": "What is new?",
    "category": "section",
    "text": "This package brings several changes over TextAnalysis.jl:Simpler API (less exported methods)\nImproved test coverage\nParametrized many of the objects i.e. DocumentTermMatrix, AbstractDocument etc\nExtended DocumentMetadata with new fields\nMany of the repetitive functions are now automatically generated (see metadata.jl, preprocessing.jl)\nRe-factored the text preprocessing API\nImproved latent semantic analysis (LSA)\neach_dtv, each_hash_dtv iterators support vector element type specification\nAdded Okapi BM25 statistic\nMany bugfixes and small extensions"
},

{
    "location": "#Installation-1",
    "page": "Introduction",
    "title": "Installation",
    "category": "section",
    "text": "Installation can be performed from either inside or outside Julia."
},

{
    "location": "#Git-cloning-1",
    "page": "Introduction",
    "title": "Git cloning",
    "category": "section",
    "text": "The StringAnalysis repository can be downloaded through git:$ git clone https://github.com/zgornel/StringAnalysis.jl"
},

{
    "location": "#Julia-REPL-1",
    "page": "Introduction",
    "title": "Julia REPL",
    "category": "section",
    "text": "The package can be installed from inside Julia. Entering the Pkg mode with ] and writing:add StringAnalysiswill download the latest registered build of the package and add it to the current active development environment."
},

{
    "location": "#Examples-1",
    "page": "Introduction",
    "title": "Examples",
    "category": "section",
    "text": "Coming soon ;)"
},

{
    "location": "api/#StringAnalysis.LSAModel",
    "page": "API Reference",
    "title": "StringAnalysis.LSAModel",
    "category": "type",
    "text": "LSAModel{S<:AbstractString, T<:AbstractFloat, A<:AbstractMatrix{T}, H<:Integer}\n\nLSA (latent semantic analysis) model. It constructs from a document term matrix (dtm) a model that can be used to embed documents in a latent semantic space pertaining to the data. The model requires that the document term matrix be a DocumentTermMatrix{T<:AbstractFloat} because the matrices resulted from the SVD operation will be forced to contain elements of type T.\n\nFields\n\nvocab::Vector{S} a vector with all the words in the corpus\nvocab_hash::Dict{S,H} a word to index in word embeddings matrix mapping\nU::A the document embeddings matrix\nΣinv::A inverse of the singular value matrix\nVᵀ::A transpose of the word embedding matrix\nstats::Symbol the statistical measure to use for word importances in documents                 available values are:                 :count (term counts)                 :tf (term frequency)                 :tfidf (default, term frequency - inverse document frequency)                 :bm25 (Okapi BM25)\nidf::Vector{T} inverse document frequencies for the words in the vocabulary\nnwords::T averge number of words in a document\nκ::Int the κ parameter of the BM25 statistic\nβ::Float64 the β parameter of the BM25 statistic\n\nU, Σinv and Vᵀ:\n\nIf X is a m×n document-term-matrix with m documents and n words so that X[i,j] represents a statistical indicator of the importance of term j in document i then:\n\nU, Σ, V = svd(X)\nΣinv = inv(Σ)\nVᵀ = V\'\n\nExamples\n\njulia> using StringAnalysis\n\n       doc1 = StringDocument(\"This is a text about an apple. There are many texts about apples.\")\n       doc2 = StringDocument(\"Pears and apples are good but not exotic. An apple a day keeps the doctor away.\")\n       doc3 = StringDocument(\"Fruits are good for you.\")\n       doc4 = StringDocument(\"This phrase has nothing to do with the others...\")\n       doc5 = StringDocument(\"Simple text, little fruit inside\")\n\n       crps = Corpus(AbstractDocument[doc1, doc2, doc3, doc4, doc5])\n       prepare!(crps, strip_punctuation)\n       update_lexicon!(crps)\n       dtm = DocumentTermMatrix{Float32}(crps, sort(collect(keys(crps.lexicon))))\n\n       ### Build LSA Model ###\n       lsa_model = LSAModel(dtm, k=3, stats=:tf)\n\n       query = StringDocument(\"Apples and an exotic fruit.\")\n       idxs, corrs = cosine(lsa_model, query)\n\n       println(\"Query: \"$(query.text)\"\")\n       for (idx, corr) in zip(idxs, corrs)\n           println(\"$corr -> \"$(crps[idx].text)\"\")\n       end\n\nQuery: \"Apples and an exotic fruit.\"\n0.91117114 -> \"This is a text about an apple  There are many texts about apples \"\n0.8093636 -> \"Simple text  little fruit inside \"\n0.4731887 -> \"Pears and apples are good but not exotic  An apple a day keeps the doctor away \"\n0.23154664 -> \"Fruits are good for you \"\n0.012299925 -> \"This phrase has nothing to do with the others \"\n\nReferences:\n\nThe LSA wiki page\nDeerwester et al. 1990\n\n\n\n\n\n"
},

{
    "location": "api/#StringAnalysis.cosine",
    "page": "API Reference",
    "title": "StringAnalysis.cosine",
    "category": "function",
    "text": "cosine(lm, doc, n=10)\n\nReturn the position of n (by default n = 10) neighbors of document doc and their cosine similarities.\n\n\n\n\n\n"
},

{
    "location": "api/#StringAnalysis.embed_document-Union{Tuple{H}, Tuple{A}, Tuple{T}, Tuple{S}, Tuple{LSAModel{S,T,A,H},AbstractDocument}} where H where A where T where S",
    "page": "API Reference",
    "title": "StringAnalysis.embed_document",
    "category": "method",
    "text": "embed_document(lm, doc)\n\nReturn the vector representation of a document doc using the LSA model lm.\n\n\n\n\n\n"
},

{
    "location": "api/#StringAnalysis.embed_word-Tuple{LSAModel,Any}",
    "page": "API Reference",
    "title": "StringAnalysis.embed_word",
    "category": "method",
    "text": "embed_word(lm, word)\n\nReturn the vector representation of word using the LSA model lm.\n\n\n\n\n\n"
},

{
    "location": "api/#StringAnalysis.get_vector-Union{Tuple{H}, Tuple{A}, Tuple{T}, Tuple{S}, Tuple{LSAModel{S,T,A,H},Any}} where H where A where T where S",
    "page": "API Reference",
    "title": "StringAnalysis.get_vector",
    "category": "method",
    "text": "get_vector(lm, word)\n\nReturns the vector representation of word from the LSA model lm.\n\n\n\n\n\n"
},

{
    "location": "api/#StringAnalysis.index-Tuple{LSAModel,Any}",
    "page": "API Reference",
    "title": "StringAnalysis.index",
    "category": "method",
    "text": "index(lm, word)\n\nReturn the index of word from the LSA model lm.\n\n\n\n\n\n"
},

{
    "location": "api/#StringAnalysis.lda-Tuple{DocumentTermMatrix,Int64,Int64,Float64,Float64}",
    "page": "API Reference",
    "title": "StringAnalysis.lda",
    "category": "method",
    "text": "ϕ, θ = lda(dtm::DocumentTermMatrix, ntopics::Int, iterations::Int, α::Float64, β::Float64)\n\nPerform Latent Dirichlet allocation.\n\nArguments\n\nα Dirichlet dist. hyperparameter for topic distribution per document. α<1 yields a sparse topic mixture for each document. α>1 yields a more uniform topic mixture for each document.\nβ Dirichlet dist. hyperparameter for word distribution per topic. β<1 yields a sparse word mixture for each topic. β>1 yields a more uniform word mixture for each topic.\n\nReturn values\n\nϕ: ntopics × nwords Sparse matrix of probabilities s.t. sum(ϕ, 1) == 1\nθ: ntopics × ndocs Dense matrix of probabilities s.t. sum(θ, 1) == 1\n\n\n\n\n\n"
},

{
    "location": "api/#StringAnalysis.lsa-Union{Tuple{DocumentTermMatrix{T}}, Tuple{T}} where T<:AbstractFloat",
    "page": "API Reference",
    "title": "StringAnalysis.lsa",
    "category": "method",
    "text": "lsa(X [;k=3, stats=:tfidf, κ=2, β=0.75])\n\nConstructs an LSA model. The input X can be a Corpus or a DocumentTermMatrix. Use ?LSAModel for more details.\n\n\n\n\n\n"
},

{
    "location": "api/#StringAnalysis.sentence_tokenize-Union{Tuple{T}, Tuple{T}} where T<:AbstractString",
    "page": "API Reference",
    "title": "StringAnalysis.sentence_tokenize",
    "category": "method",
    "text": "sentence_tokenize([lang,] s)\n\nSplits string s into sentences using WordTokenizers.split_sentences function to perform the tokenization. If a language lang is provided, it ignores it ;)\n\n\n\n\n\n"
},

{
    "location": "api/#StringAnalysis.similarity-Tuple{LSAModel,Any,Any}",
    "page": "API Reference",
    "title": "StringAnalysis.similarity",
    "category": "method",
    "text": "similarity(lm, doc1, doc2)\n\nReturn the cosine similarity value between two documents doc1 and doc2.\n\n\n\n\n\n"
},

{
    "location": "api/#StringAnalysis.tokenize-Tuple{Any}",
    "page": "API Reference",
    "title": "StringAnalysis.tokenize",
    "category": "method",
    "text": "tokenize(s [;method])\n\nTokenizes based on either the tokenize_slow or tokenize_fast functions.\n\n\n\n\n\n"
},

{
    "location": "api/#StringAnalysis.tokenize_fast-Union{Tuple{Array{S,1}}, Tuple{S}} where S<:AbstractString",
    "page": "API Reference",
    "title": "StringAnalysis.tokenize_fast",
    "category": "method",
    "text": "tokenize_fast(doc [;splitter])\n\nFunction that quickly tokenizes doc based on the splitting pattern specified by splitter::RegEx. Supported types for doc are: AbstractString, Vector{AbstractString}, StringDocument and NGramDocument.\n\n\n\n\n\n"
},

{
    "location": "api/#StringAnalysis.tokenize_slow-Union{Tuple{T}, Tuple{T}} where T<:AbstractString",
    "page": "API Reference",
    "title": "StringAnalysis.tokenize_slow",
    "category": "method",
    "text": "tokenize_slow([lang,] s)\n\nSplits string s into tokens on whitespace using WordTokenizers.tokenize function to perform the tokenization. If a language lang is provided, it ignores it ;)\n\n\n\n\n\n"
},

{
    "location": "api/#Base.size-Tuple{LSAModel}",
    "page": "API Reference",
    "title": "Base.size",
    "category": "method",
    "text": "size(lm)\n\nReturn a tuple containing the number of terms, the number of documents and the vector representation dimensionality of the LSA model lm.\n\n\n\n\n\n"
},

{
    "location": "api/#Base.summary-Tuple{AbstractDocument}",
    "page": "API Reference",
    "title": "Base.summary",
    "category": "method",
    "text": "summary(doc)\n\nShows information about the document doc.\n\n\n\n\n\n"
},

{
    "location": "api/#Base.summary-Tuple{Corpus}",
    "page": "API Reference",
    "title": "Base.summary",
    "category": "method",
    "text": "summary(crps)\n\nShows information about the corpus crps.\n\n\n\n\n\n"
},

{
    "location": "api/#StringAnalysis.in_vocabulary-Tuple{LSAModel,AbstractString}",
    "page": "API Reference",
    "title": "StringAnalysis.in_vocabulary",
    "category": "method",
    "text": "in_vocabulary(lm, word)\n\nReturn true if word is part of the vocabulary of the LSA model lm and false otherwise.\n\n\n\n\n\n"
},

{
    "location": "api/#StringAnalysis.remove_patterns!-Tuple{FileDocument,Regex}",
    "page": "API Reference",
    "title": "StringAnalysis.remove_patterns!",
    "category": "method",
    "text": "remove_patterns!(d, rex)\n\nRemoves from the document or corpus d the text matching the pattern described by the regular expression rex.\n\n\n\n\n\n"
},

{
    "location": "api/#StringAnalysis.remove_patterns-Tuple{AbstractString,Regex}",
    "page": "API Reference",
    "title": "StringAnalysis.remove_patterns",
    "category": "method",
    "text": "remove_patterns(s, rex)\n\nRemoves from the string s the text matching the pattern described by the regular expression rex.\n\n\n\n\n\n"
},

{
    "location": "api/#StringAnalysis.vocabulary-Tuple{LSAModel}",
    "page": "API Reference",
    "title": "StringAnalysis.vocabulary",
    "category": "method",
    "text": "vocabulary(lm)\n\nReturn the vocabulary as a vector of words of the LSA model lm.\n\n\n\n\n\n"
},

{
    "location": "api/#",
    "page": "API Reference",
    "title": "API Reference",
    "category": "page",
    "text": "Modules = [StringAnalysis]"
},

]}
