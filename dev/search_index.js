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
    "text": "StringAnalysis is a package for working with strings and text. It is a hard-fork from TextAnalysis.jl designed to provide a more powerful, faster and orthogonal API."
},

{
    "location": "#What-is-new?-1",
    "page": "Introduction",
    "title": "What is new?",
    "category": "section",
    "text": "This package brings several changes over TextAnalysis.jl:Simpler API (less exported methods)\nImproved test coverage\nParametrized many of the objects i.e. DocumentTermMatrix, AbstractDocument etc\nExtended DocumentMetadata with new fields\nprepare function for preprocessing AbstractStrings\nMany of the repetitive functions are now automatically generated (see metadata.jl, preprocessing.jl)\nRe-factored the text preprocessing API\nImproved latent semantic analysis (LSA)\neach_dtv, each_hash_dtv iterators support vector element type specification\nAdded Okapi BM25 statistic\nMany bugfixes and small extensions"
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
    "text": "The package can be installed from inside Julia with:using Pkg\nPkg.add(StringAnalysis)will download the latest registered build of the package and add it to the current active development environment."
},

{
    "location": "examples/#",
    "page": "Usage examples",
    "title": "Usage examples",
    "category": "page",
    "text": ""
},

{
    "location": "examples/#Usage-examples-1",
    "page": "Usage examples",
    "title": "Usage examples",
    "category": "section",
    "text": ""
},

{
    "location": "examples/#Documents-1",
    "page": "Usage examples",
    "title": "Documents",
    "category": "section",
    "text": "Documents are simple wrappers around basic structures that contain text. The underlying data representation can be simple strings, dictionaries or vectors of strings. All document types are subtypes of the parametric type AbstractDocument{T} where T<:AbstractString.using StringAnalysis\n\nsd = StringDocument(\"this is a string document\")\nnd = NGramDocument(\"this is a ngram document\")\ntd = TokenDocument(\"this is a token document\")\n# fd = FileDocument(\"/some/file\") # works the same way ..."
},

{
    "location": "examples/#Documents-and-types-1",
    "page": "Usage examples",
    "title": "Documents and types",
    "category": "section",
    "text": "The string type can be explicitly enforced:nd = NGramDocument{String}(\"this is a ngram document\")\nngrams(nd)\ntd = TokenDocument{String}(\"this is a token document\")\ntokens(td)Conversion methods are available to switch between document types (the type parameter has to be specified as well).convert(TokenDocument{SubString}, StringDocument(\"some text\"))\nconvert(NGramDocument{String}, StringDocument(\"some more text\"))"
},

{
    "location": "examples/#Metadata-1",
    "page": "Usage examples",
    "title": "Metadata",
    "category": "section",
    "text": "Alongside the text data, documents also contain metadata.doc = StringDocument(\"this is another document\")\nmetadata(doc)\nfieldnames(typeof(metadata(doc)))Metadata fields can be modified through methods bearing the same name as the metadata field. Note that these methods are not explicitly exported.StringAnalysis.id!(doc, \"doc1\");\nStringAnalysis.author!(doc, \"Corneliu C.\");\nStringAnalysis.name!(doc, \"A simple document\");\nStringAnalysis.edition_year!(doc, \"2019\");\nStringAnalysis.published_year!(doc, \"2019\");\nmetadata(doc)"
},

{
    "location": "examples/#Corpus-1",
    "page": "Usage examples",
    "title": "Corpus",
    "category": "section",
    "text": "A corpus is an object that holds a bunch of documents together.docs = [sd, nd, td]\ncrps = Corpus(docs)\ncrps.documentsThe corpus can be \'standardized\' to hold the same type of document.standardize!(crps, NGramDocument{String})\ncrps.documentshowever, the corpus has to created from an AbstractDocument document vector for the standardization to work (AbstractDocument{T} vectors are converted to a Union of all documents types parametrized by T during Corpus construction):doc1 = StringDocument(\"one\");\ndoc2 = StringDocument(\"two\");\ndoc3 = TokenDocument(\"three\");\nstandardize!(Corpus([doc1, doc3]), NGramDocument{String})  # works\nstandardize!(Corpus([doc1, doc2]), NGramDocument{String})  # fails because we have a Vector{StringDocument{T}}\nstandardize!(Corpus(AbstractDocument[doc1, doc2]), NGramDocument{String})  # worksThe corpus can be also iterated through,for (i,doc) in enumerate(crps)\n    @show (i, doc)\nendindexed into,doc = crps[1]\ndocs = crps[2:3]and used as a container.push!(crps, NGramDocument{String}(\"new document\"))\ndoc4 = pop!(crps)\nngrams(doc4)"
},

{
    "location": "examples/#The-lexicon-and-inverse-index-1",
    "page": "Usage examples",
    "title": "The lexicon and inverse index",
    "category": "section",
    "text": "The Corpus object offers the ability of creating a lexicon and an inverse index for the documents present. These are not created when the Corpus is createdcrps.lexicon\ncrps.inverse_indexbut instead have to be explicitly created:update_lexicon!(crps)\ncrps.lexicon\nupdate_inverse_index!(crps)\ncrps.inverse_index"
},

{
    "location": "examples/#Features-1",
    "page": "Usage examples",
    "title": "Features",
    "category": "section",
    "text": "If a lexicon is present in the corpus, a document term matrix (DTM) can be created. The DTM acts as a basis for word-document statistics, allowing for the representation of documents as numerical vectors. The DTM is created by calling the object constructor using as argument the corpusM = DocumentTermMatrix(crps)\ntypeof(M)\nM = DocumentTermMatrix{Int8}(crps)\ntypeof(M)or the dtm function (not recommended as the element type cannot be specified)M = dtm(crps)The default element type of the DTM is specified by the constant DEFAULT_DTM_TYPE present in src/defaults.jl.The individual rows of the DTM can also be generated iteratively whether a lexicon is present or not. If a lexicon is present, the each_dtv iterator allows the generation of the document vectors along with the control of the vector element type:for dv in each_dtv(crps, eltype=Int8)\n    @show dv\nendAlternatively, the vectors can be generated using the hash trick. The dimension of these vectors can be controlled through the cardinality keyword argument of the Corpus constructor while their type can be specified when building the iterator:for dv in each_hash_dtv(Corpus(documents(crps), cardinality=5), eltype=Int8)\n    @show dv\nendThe default Corpus cardinality is specified by the constant DEFAULT_CARDINALITY present in src/defaults.jl."
},

{
    "location": "examples/#More-features-1",
    "page": "Usage examples",
    "title": "More features",
    "category": "section",
    "text": "From the DTM, three more document-word statistics can be constructed: the term frequency, the tf-idf (term frequency - inverse document frequency) and Okapi BM25 using the tf, tf!, tf_idf, tf_idf!, bm_25 and bm_25! functions respectively. Their usage is very similar yet there exist several approaches one can take to constructing the output.The following examples with use the term frequency i.e. tf and tf!. When calling the functions that end without a !, one does not control the element type, which is defined by the constant DEFAULT_FLOAT_TYPE = eltype(1.0):M = DocumentTermMatrix(crps);\ntfm = tf(M);\nMatrix(tfm)Control of the output matrix element type - which has to be a subtype of AbstractFloat - can be done only by using the in-place modification functions. One approach is to directly modify the DTM, provided that its elements are floating point numbers:M = DocumentTermMatrix{Float16}(crps)\nMatrix(M.dtm)\ntf!(M.dtm);  # inplace modification\nMatrix(M.dtm)\n\nM = DocumentTermMatrix(crps)  # Int elements\ntf!(M.dtm)  # failsor, to provide a matrix output:rows, cols = size(M.dtm);\ntfm = zeros(Float16, rows, cols);\ntf!(M.dtm, tfm);\ntfmOne could also provide a sparse matrix output however it is important to note that in this case, the output matrix non-zero values have to correspond to the DTM\'s non-zero values:using SparseArrays\nrows, cols = size(M.dtm);\ntfm = spzeros(Float16, rows, cols)\ntfm[M.dtm .!= 0] .= 123;  # create explicitly non-zeros\ntf!(M.dtm, tfm);\nMatrix(tfm)"
},

{
    "location": "examples/#Pre-processing-1",
    "page": "Usage examples",
    "title": "Pre-processing",
    "category": "section",
    "text": "The text preprocessing mainly consists of the prepare and prepare! functions and preprocessing flags which start mostly with strip_ except for stem_words. The preprocessing function prepare works on AbstractDocument, Corpus and AbstractString types, returning new objects; prepare! works only on AbstractDocuments and Corpus as the strings are immutable.str=\"This is a text containing words and a bit of punctuation...\";\nflags = strip_punctuation|strip_articles|strip_punctuation|strip_whitespace\nprepare(str, flags)\nsd = StringDocument(str);\nprepare!(sd, flags);\ntext(sd)More extensive preprocessing examples can be viewed in test/preprocessing.jl."
},

{
    "location": "examples/#Semantic-Analysis-1",
    "page": "Usage examples",
    "title": "Semantic Analysis",
    "category": "section",
    "text": "The semantic analysis of a corpus relates to the task of building structures that approximate the concepts present in its documents. It does not necessarily involve prior semantic understanding of the documents (Wikipedia).StringAnalysis provides two approaches of performing semantic analysis of a corpus: latent semantic analysis (LSA) and latent Dirichlet allocation (LDA)."
},

{
    "location": "examples/#Latent-Semantic-Analysis-(LSA)-1",
    "page": "Usage examples",
    "title": "Latent Semantic Analysis (LSA)",
    "category": "section",
    "text": "The following example gives a straightforward usage example of LSA and can be found in the documentation of LSAModel as well.doc1 = StringDocument(\"This is a text about an apple. There are many texts about apples.\");\ndoc2 = StringDocument(\"Pears and apples are good but not exotic. An apple a day keeps the doctor away.\");\ndoc3 = StringDocument(\"Fruits are good for you.\");\ndoc4 = StringDocument(\"This phrase has nothing to do with the others...\");\ndoc5 = StringDocument(\"Simple text, little info inside\");\n# Build corpus\ncrps = Corpus(AbstractDocument[doc1, doc2, doc3, doc4, doc5]);\nprepare!(crps, strip_punctuation);\nupdate_lexicon!(crps);\nM = DocumentTermMatrix{Float32}(crps, sort(collect(keys(crps.lexicon))));\n\n### Build LSA Model ###\nlsa_model = LSAModel(M, k=3, stats=:tf);\n\nquery = StringDocument(\"Apples and an exotic fruit.\");\nidxs, corrs = cosine(lsa_model, query);\n\nfor (idx, corr) in zip(idxs, corrs)\n    println(\"$corr -> \\\"$(crps[idx].text)\\\"\");\nendLSA models can be saved and retrievedfile = \"model.txt\"\nlsa_model\nsave(lsa_model, file)  # model saved\nprint(join(readlines(file)[1:3], \"\\n\"))  # first three lines\nnew_model = load(file, Float64)  # change element type\nrm(file)"
},

{
    "location": "examples/#Latent-Dirichlet-Allocation-(LDA)-1",
    "page": "Usage examples",
    "title": "Latent Dirichlet Allocation (LDA)",
    "category": "section",
    "text": "Documentation coming soon; check the API reference for information on the associated methods."
},

{
    "location": "api/#StringAnalysis.StringAnalysis",
    "page": "API Reference",
    "title": "StringAnalysis.StringAnalysis",
    "category": "module",
    "text": "A Julia library for working with text hard-forked from TextAnalysis.jl.\n\n\n\n\n\n"
},

{
    "location": "api/#StringAnalysis.LSAModel",
    "page": "API Reference",
    "title": "StringAnalysis.LSAModel",
    "category": "type",
    "text": "LSAModel{S<:AbstractString, T<:AbstractFloat, A<:AbstractMatrix{T}, H<:Integer}\n\nLSA (latent semantic analysis) model. It constructs from a document term matrix (dtm) a model that can be used to embed documents in a latent semantic space pertaining to the data. The model requires that the document term matrix be a DocumentTermMatrix{T<:AbstractFloat} because the matrices resulted from the SVD operation will be forced to contain elements of type T.\n\nFields\n\nvocab::Vector{S} a vector with all the words in the corpus\nvocab_hash::Dict{S,H} a word to index in word embeddings matrix mapping\nU::A the document embeddings matrix\nΣinv::A inverse of the singular value matrix\nVᵀ::A transpose of the word embedding matrix\nstats::Symbol the statistical measure to use for word importances in documents                 available values are:                 :tf (term frequency)                 :tfidf (default, term frequency - inverse document frequency)                 :bm25 (Okapi BM25)\nidf::Vector{T} inverse document frequencies for the words in the vocabulary\nnwords::T averge number of words in a document\nκ::Int the κ parameter of the BM25 statistic\nβ::Float64 the β parameter of the BM25 statistic\ntol::T minimum size of the vector components (default T(1e-15))\n\nU, Σinv and Vᵀ:\n\nIf X is a m×n document-term-matrix with m documents and n words so that X[i,j] represents a statistical indicator of the importance of term j in document i then:\n\nU, Σ, V = svd(X)\nΣinv = inv(Σ)\nVᵀ = V\'\n\nThe version of U actually stored in the model has its columns normalized to their norm.\n\nExamples\n\njulia> using StringAnalysis\n\n       doc1 = StringDocument(\"This is a text about an apple. There are many texts about apples.\")\n       doc2 = StringDocument(\"Pears and apples are good but not exotic. An apple a day keeps the doctor away.\")\n       doc3 = StringDocument(\"Fruits are good for you.\")\n       doc4 = StringDocument(\"This phrase has nothing to do with the others...\")\n       doc5 = StringDocument(\"Simple text, little info inside\")\n\n       crps = Corpus(AbstractDocument[doc1, doc2, doc3, doc4, doc5])\n       prepare!(crps, strip_punctuation)\n       update_lexicon!(crps)\n       dtm = DocumentTermMatrix{Float32}(crps, sort(collect(keys(crps.lexicon))))\n\n       ### Build LSA Model ###\n       lsa_model = LSAModel(dtm, k=3, stats=:tf)\n\n       query = StringDocument(\"Apples and an exotic fruit.\")\n       idxs, corrs = cosine(lsa_model, query)\n\n       println(\"Query: \"$(query.text)\"\")\n       for (idx, corr) in zip(idxs, corrs)\n           println(\"$corr -> \"$(crps[idx].text)\"\")\n       end\nQuery: \"Apples and an exotic fruit.\"\n0.9746108 -> \"Pears and apples are good but not exotic  An apple a day keeps the doctor away \"\n0.870703 -> \"This is a text about an apple  There are many texts about apples \"\n0.7122063 -> \"Fruits are good for you \"\n0.22725986 -> \"This phrase has nothing to do with the others \"\n0.076901935 -> \"Simple text  little info inside \"\n\nReferences:\n\nThe LSA wiki page\nDeerwester et al. 1990\n\n\n\n\n\n"
},

{
    "location": "api/#StringAnalysis.TextHashFunction",
    "page": "API Reference",
    "title": "StringAnalysis.TextHashFunction",
    "category": "type",
    "text": "TextHashFunction(hash_function::Function, cardinality::Int)\n\nThe basic structure for performing text hashing: uses the hash_function to generate feature vectors of length cardinality.\n\nDetails\n\nThe hash trick is the use a hash function instead of a lexicon to determine the columns of a DocumentTermMatrix-like encoding of the data. To produce a DTM for a Corpus for which we do not have an existing lexicon, we need someway to map the terms from each document into column indices. We use the now standard \"Hash Trick\" in which we hash strings and then reduce the resulting integers modulo N, which defines the numbers of columns we want our DTM to have. This amounts to doing a non-linear dimensionality reduction with low probability that similar terms hash to the same dimension.\n\nTo make things easier, we wrap Julia\'s hash functions in a new type, TextHashFunction, which maintains information about the desired cardinality of the hashes.\n\nReferences:\n\nThe \"Hash Trick\" wiki page\nMoody, John 1989\n\nExamples\n\njulia> doc = StringDocument(\"this is a text\")\n       thf = TextHashFunction(hash, 13)\n       hash_dtv(doc, thf, Float16)\n13-element Array{Float16,1}:\n 1.0\n 1.0\n 0.0\n 0.0\n 0.0\n 0.0\n 0.0\n 2.0\n 0.0\n 0.0\n 0.0\n 0.0\n 0.0\n\n\n\n\n\n"
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
    "location": "api/#StringAnalysis.load-Union{Tuple{AbstractString}, Tuple{T}, Tuple{AbstractString,Type{T}}} where T<:AbstractFloat",
    "page": "API Reference",
    "title": "StringAnalysis.load",
    "category": "method",
    "text": "load(filename, type; [sparse=true])\n\nLoads an LSA model from filename into an LSA model object. The embeddings matrix element type is specified by type (default Float32) while the keyword argument sparse specifies whether the matrix should be sparse or not.\n\n\n\n\n\n"
},

{
    "location": "api/#StringAnalysis.lsa-Union{Tuple{DocumentTermMatrix{T}}, Tuple{T}} where T<:AbstractFloat",
    "page": "API Reference",
    "title": "StringAnalysis.lsa",
    "category": "method",
    "text": "lsa(X [;k=3, stats=:tfidf, κ=2, β=0.75, tol=1e-15])\n\nConstructs an LSA model. The input X can be a Corpus or a DocumentTermMatrix. Use ?LSAModel for more details. Vector components smaller than tol will be zeroed out.\n\n\n\n\n\n"
},

{
    "location": "api/#StringAnalysis.save-Union{Tuple{H}, Tuple{A}, Tuple{T}, Tuple{S}, Tuple{LSAModel{S,T,A,H},AbstractString}} where H where A where T where S",
    "page": "API Reference",
    "title": "StringAnalysis.save",
    "category": "method",
    "text": "save(lm, filename)\n\nSaves an LSA model lm to disc in file filename.\n\n\n\n\n\n"
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
