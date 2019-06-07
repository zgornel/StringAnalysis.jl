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
    "text": "StringAnalysis is a package for working with strings and text. It is a hard-fork from TextAnalysis.jl designed to provide a richer, faster and orthogonal API."
},

{
    "location": "#What-is-new?-1",
    "page": "Introduction",
    "title": "What is new?",
    "category": "section",
    "text": "This package brings several changes over TextAnalysis.jl:Added the Okapi BM25 statistic\nAdded dimensionality reduction with sparse random projections\nAdded co-occurence matrix\nImproved latent semantic analysis\nRe-factored text preprocessing API\nDTM and similar have documents as columns\nParametrized many of the objects (DocumentTermMatrix, AbstractDocuments)\nElement type specification for each_dtv, each_hash_dtv\nExtended DocumentMetadata fields\nSimpler API i.e. less exported methods\nMany of the repetitive functions are now automatically generated (see metadata.jl, preprocessing.jl)\nImproved test coverage\nMany bugfixes and small extensions"
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
    "text": "A corpus is an object that holds a bunch of documents together.docs = [sd, nd, td]\ncrps = Corpus(docs)\ncrps.documentsThe corpus can be \'standardized\' to hold the same type of document,standardize!(crps, NGramDocument{String})\ncrps.documentshowever, the corpus has to be created from an AbstractDocument document vector for the standardization to work (AbstractDocument{T} vectors are converted to a Union of all documents types parametrized by T during Corpus construction):doc1 = StringDocument(\"one\");\ndoc2 = StringDocument(\"two\");\ndoc3 = TokenDocument(\"three\");\nstandardize!(Corpus([doc1, doc3]), NGramDocument{String})  # works\nstandardize!(Corpus([doc1, doc2]), NGramDocument{String})  # fails because we have a Vector{StringDocument{T}}\nstandardize!(Corpus(AbstractDocument[doc1, doc2]), NGramDocument{String})  # worksThe corpus can be also iterated through,for (i,doc) in enumerate(crps)\n    @show (i, doc)\nendindexed into,doc = crps[1]\ndocs = crps[2:3]and used as a container.push!(crps, NGramDocument{String}(\"new document\"))\ndoc4 = pop!(crps)\nngrams(doc4)"
},

{
    "location": "examples/#The-lexicon-and-inverse-index-1",
    "page": "Usage examples",
    "title": "The lexicon and inverse index",
    "category": "section",
    "text": "The Corpus object offers the ability of creating a lexicon and an inverse index for the documents present. These are not automatically created when the Corpus is created,crps.lexicon\ncrps.inverse_indexbut instead have to be explicitly built:update_lexicon!(crps)\ncrps.lexicon\nupdate_inverse_index!(crps)\ncrps.inverse_index"
},

{
    "location": "examples/#Preprocessing-1",
    "page": "Usage examples",
    "title": "Preprocessing",
    "category": "section",
    "text": "The text preprocessing mainly consists of the prepare and prepare! functions and preprocessing flags which start mostly with strip_ except for stem_words. The preprocessing function prepare works on AbstractDocument, Corpus and AbstractString types, returning new objects; prepare! works only on AbstractDocuments and Corpus as strings are immutable.str=\"This is a text containing words, some more words, a bit of punctuation and 1 number...\";\nsd = StringDocument(str);\nflags = strip_punctuation|strip_articles|strip_punctuation|strip_whitespace\nprepare(str, flags)\nprepare!(sd, flags);\ntext(sd)More extensive preprocessing examples can be viewed in test/preprocessing.jl.One can strip parts of speech i.e. prepositions, articles, in languages other than English (support provided from Languages.jl):using Languages\nit = StringDocument(\"Quest\'e un piccolo esempio di come si puo fare l\'analisi\");\nStringAnalysis.language!(it, Languages.Italian());\nprepare!(it, strip_articles|strip_prepositions|strip_whitespace);\ntext(it)In the case of AbstractStrings, the language has to be explicitly defined:prepare(\"Nous sommes tous d\'accord avec les examples!\", stem_words, language=Languages.French())"
},

{
    "location": "examples/#Features-1",
    "page": "Usage examples",
    "title": "Features",
    "category": "section",
    "text": ""
},

{
    "location": "examples/#Document-Term-Matrix-(DTM)-1",
    "page": "Usage examples",
    "title": "Document Term Matrix (DTM)",
    "category": "section",
    "text": "If a lexicon is present in the corpus, a document term matrix (DTM) can be created. The DTM acts as a basis for word-document statistics, allowing for the representation of documents as numerical vectors. The DTM is created from a Corpus by calling the constructorM = DocumentTermMatrix(crps)\ntypeof(M)\nM = DocumentTermMatrix{Int8}(crps)\ntypeof(M)or the dtm functionM = dtm(crps, Int8);\nMatrix(M)It is important to note that the type parameter of the DTM object can be specified (also in the dtm function) but not specifically required. This can be useful in some cases for reducing memory requirements. The default element type of the DTM is specified by the constant DEFAULT_DTM_TYPE present in src/defaults.jl.note: Note\nFrom version v0.3.2, the columns of the document-term matrix represent document vectors. This convention holds accross the package where whenever multiple documents are represented. This represents a breaking change from previous versions and TextAnalysis.jl and may break code if not taken into account.One can verify the DTM dimensions with:@assert size(dtm(crps)) == (length(lexicon(crps)), length(crps))  # O.K."
},

{
    "location": "examples/#Document-Term-Vectors-(DTVs)-1",
    "page": "Usage examples",
    "title": "Document Term Vectors (DTVs)",
    "category": "section",
    "text": "The individual rows of the DTM can also be generated iteratively whether a lexicon is present or not. If a lexicon is present, the each_dtv iterator allows the generation of the document vectors along with the control of the vector element type:for dv in map(Vector, each_dtv(crps, eltype=Int8))\n    @show dv\nendAlternatively, the vectors can be generated using the hash trick. This is a form of dimensionality reduction as cardinality i.e. output dimension is much smaller than the dimension of the original DTM vectors, which is equal to the length of the lexicon. The cardinality is a keyword argument of the Corpus constructor. The hashed vector output type can be specified when building the iterator:new_crps = Corpus(documents(crps), cardinality=7);\nhash_vectors = map(Vector, each_hash_dtv(new_crps, eltype=Int8));\nfor hdv in hash_vectors\n    @show hdv\nendOne can construct a \'hashed\' version of the DTM as well:hash_dtm(Corpus(documents(crps), cardinality=5), Int8)The default Corpus cardinality is specified by the constant DEFAULT_CARDINALITY present in src/defaults.jl.note: Note\nFrom version v0.3.4, all document vectors are instances of SparseVector. This consequently has an impact on the output and performance of methods that directly employ DTVs such as the embed_document method. In certain cases, if speed is more important than memory consumption, it may be useful to first transform the vectors into a dense representation prior to transformation i.e. dtv_dense = Vector(dtv_sparse)."
},

{
    "location": "examples/#TF,-TF-IDF,-BM25-1",
    "page": "Usage examples",
    "title": "TF, TF-IDF, BM25",
    "category": "section",
    "text": "From the DTM, three more document-word statistics can be constructed: the term frequency, the tf-idf (term frequency - inverse document frequency) and Okapi BM25 using the tf, tf!, tf_idf, tf_idf!, bm_25 and bm_25! functions respectively. Their usage is very similar yet there exist several approaches one can take to constructing the output.The following examples use the term frequency i.e. tf and tf! functions only. When calling the functions that end without a !, which do not require the specification of an output matrix, one does not control the output\'s element type. The default output type is defined by the constant DEFAULT_FLOAT_TYPE = eltype(1.0):M = DocumentTermMatrix(crps);\ntfm = tf(M);\nMatrix(tfm)Control of the output matrix element type - which has to be a subtype of AbstractFloat - can be done only by using the in-place modification functions. One approach is to directly modify the DTM, provided that its elements are floating point numbers:M = DocumentTermMatrix{Float16}(crps)\nMatrix(M.dtm)\ntf!(M.dtm);  # inplace modification\nMatrix(M.dtm)\nM = DocumentTermMatrix(crps)  # Int elements\ntf!(M.dtm)  # fails because of Int elementsor, to provide a matrix output:rows, cols = size(M.dtm);\ntfm = zeros(Float16, rows, cols);\ntf!(M.dtm, tfm);\ntfmOne could also provide a sparse matrix output however it is important to note that in this case, the output matrix non-zero values have to correspond to the DTM\'s non-zero values:using SparseArrays\nrows, cols = size(M.dtm);\ntfm = spzeros(Float16, rows, cols)\ntfm[M.dtm .!= 0] .= 123;  # create explicitly non-zeros\ntf!(M.dtm, tfm);\nMatrix(tfm)"
},

{
    "location": "examples/#Co-occurrence-Matrix-(COOM)-1",
    "page": "Usage examples",
    "title": "Co-occurrence Matrix (COOM)",
    "category": "section",
    "text": "Another type of feature matrix that can be created is the co-occurence matrix (COOM) of the document or corpus. The elements of the matrix indicate how many times two words co-occur in a (sliding) word window of a given size. The COOM can be calculated for objects of type Corpus, AbstractDocument (with the exception of NGramDocument since order is word order is lost) and AbstractString. The constructor supports specification of the window size, whether the counts should be normalized (to the distance between words in the window) as well as specific terms for which co-occurrences in the document should be calculated.Remarks:The sliding window used to count co-occurrences does not take into consideration sentence stops however, it does with documents i.e. does not span across documents\nThe co-occurrence matrices of the documents in a corpus are summed up when calculating the matrix for an entire corpus\nThe co-occurrence matrix always has elements that are subtypes of AbstractFloat and cannot be calculated for NGramDocumentsC = CooMatrix(crps, window=1, normalize=false)  # fails, documents are NGramDocument\nsmallcrps = Corpus([sd, td])\nC = CooMatrix(smallcrps, window=1, normalize=false)  # worksThe actual size of the sliding window is 2 * window + 1, with the keyword argument window specifying how many words to consider to the left and right of the center oneFor a simple document, one should first preprocess the document and subsequently calculate the matrix:some_document = \"This is a document. In the document, there are two sentences.\";\nfiltered_document = prepare(some_document, strip_whitespace|strip_case|strip_punctuation)\nC = CooMatrix{Float32}(some_document, window=3)  # word distances matter\nMatrix(coom(C))One can also calculate the COOM corresponding to a reduced lexicon. The resulting matrix will be proportional to the size of the new lexicon and more sparse if the window size is small.C = CooMatrix(smallcrps, [\"this\", \"is\", \"a\"], window=1, normalize=false)\nC.column_indices\nMatrix(coom(C))"
},

{
    "location": "examples/#Dimensionality-reduction-1",
    "page": "Usage examples",
    "title": "Dimensionality reduction",
    "category": "section",
    "text": ""
},

{
    "location": "examples/#Random-projections-1",
    "page": "Usage examples",
    "title": "Random projections",
    "category": "section",
    "text": "In mathematics and statistics, random projection is a technique used to reduce the dimensionality of a set of points which lie in Euclidean space. Random projection methods are powerful methods known for their simplicity and less erroneous output compared with other methods. According to experimental results, random projection preserve distances well, but empirical results are sparse. They have been applied to many natural language tasks under the name of random indexing. The core idea behind random projection is given in the Johnson-Lindenstrauss lemma which states that if points in a vector space are of sufficiently high dimension, then they may be projected into a suitable lower-dimensional space in a way which approximately preserves the distances between the points (Wikipedia). The implementation here relies on the generalized sparse random projection matrix to generate a random projection model. For more details see the API documentation for RPModel and random_projection_matrix. To construct a random projection matrix that maps m dimension to k, one can dom = 10; k = 2; T = Float32;\ndensity = 0.2;  # percentage of non-zero elements\nR = StringAnalysis.random_projection_matrix(m, k, T, density)Building a random projection model from a DocumentTermMatrix or Corpus is straightforwardM = DocumentTermMatrix{Float32}(crps)\nmodel = RPModel(M, k=2, density=0.5, stats=:tf)\nmodel2 = rp(crps, T, k=17, density=0.1, stats=:tfidf)Once the model is created, one can reduce document term vector dimensionality. First, the document term vector is constructed using the stats keyword argument and subsequently, the vector is projected into the random sub-space:doc = StringDocument(\"this is a new document\")\nembed_document(model, doc)\nembed_document(model2, doc)Embedding a DTM or corpus can be done in a similar way:Matrix(embed_document(model, M))\nMatrix(embed_document(model2, crps))Random projection models can be saved/loaded to/from disk using a text format.file = \"model.txt\"\nmodel\nsave_rp_model(model, file)  # model saved\nprint(join(readlines(file)[1:5], \"\\n\"))  # first five lines\nnew_model = load_rp_model(file, Float64)  # change element type\nrm(file)"
},

{
    "location": "examples/#No-projection-hack-1",
    "page": "Usage examples",
    "title": "No projection hack",
    "category": "section",
    "text": "As previously noted, before projection, the DTV is calculated according to the value of the stats keyword argument value.  The vector can composed of term counts, frequencies and so on and is more generic than the output of the dtv function which yields only term counts. It is useful to be able to calculate and output these vectors without projecting them into the lower dimensional space. This can be achieved by simply providing a negative or zero value to the model parameter k. In the background, the random projection matrix of the model is replaced by the identity matrix.model = RPModel(M, k=0, stats=:bm25)\nembed_document(model, crps[1])  # normalized BM25 document vector\nembed_document(model, crps)\'*embed_document(model, crps[1])  # intra-document similarity"
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
    "text": "The following example gives a straightforward usage example of LSA. It is geared towards information retrieval (LSI) as it focuses on document comparison and embedding. We assume a number of documents,doc1 = StringDocument(\"This is a text about an apple. There are many texts about apples.\");\ndoc2 = StringDocument(\"Pears and apples are good but not exotic. An apple a day keeps the doctor away.\");\ndoc3 = StringDocument(\"Fruits are good for you.\");\ndoc4 = StringDocument(\"This phrase has nothing to do with the others...\");\ndoc5 = StringDocument(\"Simple text, little info inside\");and create the corpus and its DTM:crps = Corpus(AbstractDocument[doc1, doc2, doc3, doc4, doc5]);\nprepare!(crps, strip_punctuation);\nupdate_lexicon!(crps);\nM = DocumentTermMatrix{Float32}(crps, collect(keys(crps.lexicon)));Building an LSA model is straightforward:lm = LSAModel(M, k=4, stats=:tfidf)Once the model is created, it can be used to either embed documents,query = StringDocument(\"Apples and an exotic fruit.\");\nembed_document(lm, query)embed the corpus,V = embed_document(lm, crps)search for matching documents,idxs, corrs = cosine(lm, crps, query);\nfor (idx, corr) in zip(idxs, corrs)\n    println(\"$corr -> \\\"$(crps[idx].text)\\\"\");\nendor check for structure within the dataU = lm.Uᵀ;\nV\'*V  # document to document similarity\nU\'*U  # term to term similarityLSA models can be saved/loaded to/from disk using a text format similar to the random projection model one.file = \"model.txt\"\nlm\nsave_lsa_model(lm, file)  # model saved\nprint(join(readlines(file)[1:5], \"\\n\"))  # first five lines\nnew_model = load_lsa_model(file, Float64)  # change element type\nrm(file)"
},

{
    "location": "examples/#Latent-Dirichlet-Allocation-(LDA)-1",
    "page": "Usage examples",
    "title": "Latent Dirichlet Allocation (LDA)",
    "category": "section",
    "text": "Documentation coming soon; check the API reference for information on the associated methods."
},

{
    "location": "doc_extensions/#",
    "page": "More on documents",
    "title": "More on documents",
    "category": "page",
    "text": ""
},

{
    "location": "doc_extensions/#Extending-the-document-model-1",
    "page": "More on documents",
    "title": "Extending the document model",
    "category": "section",
    "text": "Sometimes it may make sense to define new document types with whom to operate and use only some functionality of this package. For example, let us define two new document types, a SimpleDocument with no metadatausing StringAnalysis\nstruct NoMetadata <: AbstractMetadata end\n\nstruct SimpleDocument{T<:AbstractString} <: AbstractDocument{T, NoMetadata}\n    text::T\nendand a ConferencePublication with only a limited number of metadata fields.struct ConferenceMetadata <: AbstractMetadata\n    name::String\n    authors::String\n    conference::String\nend\n\nstruct ConferencePublication{T<:AbstractString} <: AbstractDocument{T, ConferenceMetadata}\n    text::T\n    metadata::ConferenceMetadata\nendAt this point, one can create documents and use basic containers along with other standard documents of the package:sd = SimpleDocument(\"a simple document\")\ncmd = ConferenceMetadata(\"Tile Inc.\",\"John P. Doe\",\"IEEE Conference on Unknown Facts\")\ncd = ConferencePublication(\"publication text\", cmd)\n\ndoc = StringDocument(\"a document\")\n\ndocs = [sd, cd, doc]However, creating a Corpus fails because no conversion method exists between the new document types and any of the standardized ones StringDocument, NGramDocument etc.Corpus(AbstractDocument[sd, cd, doc])By defining at least one conversion method to a known type,Base.convert(::Type{NGramDocument{String}}, doc::SimpleDocument) =\n    NGramDocument{String}(doc.text)\nBase.convert(::Type{NGramDocument{String}}, doc::ConferencePublication) = begin\n    new_doc = NGramDocument{String}(doc.text)\n    new_doc.metadata.name = doc.metadata.name\n    new_doc.metadata.author = doc.metadata.authors\n    new_doc.metadata.note = doc.metadata.conference\n    return new_doc\nendthe Corpus can be created and the rest of the functionality of the package i.e. numerical operations, can be employed on the document data.crps = Corpus(AbstractDocument[sd, cd, doc])\nmetadata.(doc for doc in crps)\nDocumentTermMatrix(crps)The SimpleDocument and ConferencePublication were both converted to NGramDocuments since this was the only conversion method available. If more would be available, the priority of conversion is given by the code in the abstract_convert function. Generally, one single conversion method suffices."
},

{
    "location": "api/#StringAnalysis.StringAnalysis",
    "page": "API Reference",
    "title": "StringAnalysis.StringAnalysis",
    "category": "module",
    "text": "A Julia library for working with text, hard-forked from TextAnalysis.jl.\n\n\n\n\n\n"
},

{
    "location": "api/#StringAnalysis.CooMatrix",
    "page": "API Reference",
    "title": "StringAnalysis.CooMatrix",
    "category": "type",
    "text": "Basic Co-occurrence Matrix (COOM) type.\n\nFields\n\ncoomm::SparseMatriCSC{T,Int} the actual COOM; elements represent\n\nco-occurrences of two terms within a given window\n\nterms::Vector{String} a list of terms that represent the lexicon of\n\nthe document or corpus\n\ncolumn_indices::OrderedDict{String, Int} a map between the terms and the\n\ncolumns of the co-occurrence matrix\n\n\n\n\n\n"
},

{
    "location": "api/#StringAnalysis.CooMatrix-Union{Tuple{T}, Tuple{Corpus,Array{String,1}}} where T<:AbstractFloat",
    "page": "API Reference",
    "title": "StringAnalysis.CooMatrix",
    "category": "method",
    "text": "CooMatrix{T}(crps::Corpus [,terms] [;window=5, normalize=true])\n\nAuxiliary constructor(s) of the CooMatrix type. The type T has to be a subtype of AbstractFloat. The constructor(s) requires a corpus crps and a terms structure representing the lexicon of the corpus. The latter can be a Vector{String}, an AbstractDict where the keys are the lexicon, or can be omitted, in which case the lexicon field of the corpus is used.\n\n\n\n\n\n"
},

{
    "location": "api/#StringAnalysis.DocumentTermMatrix",
    "page": "API Reference",
    "title": "StringAnalysis.DocumentTermMatrix",
    "category": "type",
    "text": "Basic Document-Term-Matrix (DTM) type.\n\nFields\n\ndtm::SparseMatriCSC{T,Int} the actual DTM; rows represent terms\n\nand columns represent documents\n\nterms::Vector{String} a list of terms that represent the lexicon of\n\nthe corpus associated with the DTM\n\nrow_indices::OrderedDict{String, Int} a map between the terms and the\n\nrows of the dtm\n\n\n\n\n\n"
},

{
    "location": "api/#StringAnalysis.DocumentTermMatrix-Union{Tuple{T}, Tuple{Corpus,Array{String,1}}} where T<:Real",
    "page": "API Reference",
    "title": "StringAnalysis.DocumentTermMatrix",
    "category": "method",
    "text": "DocumentTermMatrix{T}(crps::Corpus [,terms] [; tokenizer=DEFAULT_TOKENIZER])\n\nAuxiliary constructor(s) of the DocumentTermMatrix type. The type T has to be a subtype of Real. The constructor(s) requires a corpus crps and a terms structure representing the lexicon of the corpus. The latter can be a Vector{String}, an AbstractDict where the keys are the lexicon, or can be missing, in which case the lexicon field of the corpus is used.\n\n\n\n\n\n"
},

{
    "location": "api/#StringAnalysis.LSAModel",
    "page": "API Reference",
    "title": "StringAnalysis.LSAModel",
    "category": "type",
    "text": "LSAModel{S<:AbstractString, T<:AbstractFloat, A<:AbstractMatrix{T}, H<:Integer}\n\nLSA (latent semantic analysis) model. It constructs from a document term matrix (dtm) a model that can be used to embed documents in a latent semantic space pertaining to the data. The model requires that the document term matrix be a DocumentTermMatrix{T<:AbstractFloat} because the elements of the matrices resulted from the SVD operation are floating point numbers and these have to match or be convertible to type T.\n\nFields\n\nvocab::Vector{S} a vector with all the words in the corpus\nvocab_hash::OrderedDict{S,H} a word to index in word embeddings matrix mapping\nΣinv::A diagonal of the inverse singular value matrix\nUᵀ::A transpose of the word embedding matrix\nstats::Symbol the statistical measure to use for word importances in documents. Available values are: :count (term count), :tf (term frequency), :tfidf (default, term frequency-inverse document frequency) and :bm25 (Okapi BM25)\nidf::Vector{T} inverse document frequencies for the words in the vocabulary\nnwords::T averge number of words in a document\nκ::Int the κ parameter of the BM25 statistic\nβ::Float64 the β parameter of the BM25 statistic\ntol::T minimum size of the vector components (default T(1e-15))\n\nSVD matrices U, Σinv and V:\n\nIf X is a m×n document-term-matrix with n documents and m words so that X[i,j] represents a statistical indicator of the importance of term i in document j then:\n\nU, Σ, V = svd(X)\nΣinv = diag(inv(Σ))\nUᵀ = U\'\nX ≈ U * Σ * V\'\n\nThe matrix V of document embeddings is not actually stored in the model.\n\nExamples\n\njulia> using StringAnalysis\n\n       doc1 = StringDocument(\"This is a text about an apple. There are many texts about apples.\")\n       doc2 = StringDocument(\"Pears and apples are good but not exotic. An apple a day keeps the doctor away.\")\n       doc3 = StringDocument(\"Fruits are good for you.\")\n       doc4 = StringDocument(\"This phrase has nothing to do with the others...\")\n       doc5 = StringDocument(\"Simple text, little info inside\")\n\n       crps = Corpus(AbstractDocument[doc1, doc2, doc3, doc4, doc5])\n       prepare!(crps, strip_punctuation)\n       update_lexicon!(crps)\n       dtm = DocumentTermMatrix{Float32}(crps, collect(keys(crps.lexicon)))\n\n       ### Build LSA Model ###\n       lsa_model = LSAModel(dtm, k=3, stats=:tf)\n\n       query = StringDocument(\"Apples and an exotic fruit.\")\n       idxs, corrs = cosine(lsa_model, crps, query)\n\n       println(\"Query: \"$(query.text)\"\")\n       for (idx, corr) in zip(idxs, corrs)\n           println(\"$corr -> \"$(crps[idx].text)\"\")\n       end\nQuery: \"Apples and an exotic fruit.\"\n0.9746108 -> \"Pears and apples are good but not exotic  An apple a day keeps the doctor away \"\n0.870703 -> \"This is a text about an apple  There are many texts about apples \"\n0.7122063 -> \"Fruits are good for you \"\n0.22725986 -> \"This phrase has nothing to do with the others \"\n0.076901935 -> \"Simple text  little info inside \"\n\nReferences:\n\nThe LSA wiki page\nDeerwester et al. 1990\n\n\n\n\n\n"
},

{
    "location": "api/#StringAnalysis.RPModel",
    "page": "API Reference",
    "title": "StringAnalysis.RPModel",
    "category": "type",
    "text": "RPModel{S<:AbstractString, T<:AbstractFloat, A<:AbstractMatrix{T}, H<:Integer}\n\nRandom projection model. It constructs from a document term matrix (DTM) a model that can be used to embed documents in a random sub-space. The model requires that the document term matrix be a DocumentTermMatrix{T<:AbstractFloat} because the elements of the matrices resulted projection operation are floating point numbers and these have to match or be convertible to type T. The approach is based on the effects of the Johnson-Lindenstrauss lemma.\n\nFields\n\nvocab::Vector{S} a vector with all the words in the corpus\nvocab_hash::OrderedDict{S,H} a word to index in the random projection maatrix mapping\nR::A the random projection matrix\nstats::Symbol the statistical measure to use for word importances in documents. Available values are: :count (term count), :tf (term frequency), :tfidf (default, term frequency-inverse document frequency) and :bm25 (Okapi BM25)\nidf::Vector{T} inverse document frequencies for the words in the vocabulary\nnwords::T averge number of words in a document\nκ::Int the κ parameter of the BM25 statistic\nβ::Float64 the β parameter of the BM25 statistic\nproject::Bool specifies whether the model actually performs the projection or not; it is false if the number of dimensions provided is zero or negative\n\nReferences:\n\nKaski 1998\nAchlioptas 2001\nLi et al. 2006\n\n\n\n\n\n"
},

{
    "location": "api/#StringAnalysis.TextHashFunction",
    "page": "API Reference",
    "title": "StringAnalysis.TextHashFunction",
    "category": "type",
    "text": "TextHashFunction(hash_function::Function, cardinality::Int)\n\nThe basic structure for performing text hashing: uses the hash_function to generate feature vectors of length cardinality.\n\nDetails\n\nThe hash trick is the use a hash function instead of a lexicon to determine the columns of a DocumentTermMatrix-like encoding of the data. To produce a DTM for a Corpus for which we do not have an existing lexicon, we need someway to map the terms from each document into column indices. We use the now standard \"Hash Trick\" in which we hash strings and then reduce the resulting integers modulo N, which defines the numbers of columns we want our DTM to have. This amounts to doing a non-linear dimensionality reduction with low probability that similar terms hash to the same dimension.\n\nTo make things easier, we wrap Julia\'s hash functions in a new type, TextHashFunction, which maintains information about the desired cardinality of the hashes.\n\nReferences:\n\nThe \"Hash Trick\" wiki page\nMoody, John 1989\n\nExamples\n\njulia> doc = StringDocument(\"this is a text\")\n       thf = TextHashFunction(hash, 13)\n       hash_dtv(doc, thf, Float16)\n13-element Array{Float16,1}:\n 1.0\n 1.0\n 0.0\n 0.0\n 0.0\n 0.0\n 0.0\n 2.0\n 0.0\n 0.0\n 0.0\n 0.0\n 0.0\n\n\n\n\n\n"
},

{
    "location": "api/#StringAnalysis.coom-Tuple{CooMatrix}",
    "page": "API Reference",
    "title": "StringAnalysis.coom",
    "category": "method",
    "text": "coom(c::CooMatrix)\n\nAccess the co-occurrence matrix field coom of a CooMatrix c.\n\n\n\n\n\n"
},

{
    "location": "api/#StringAnalysis.coom-Union{Tuple{Any}, Tuple{T}, Tuple{Any,Type{T}}} where T<:AbstractFloat",
    "page": "API Reference",
    "title": "StringAnalysis.coom",
    "category": "method",
    "text": "coom(entity, eltype=DEFAULT_FLOAT_TYPE [;window=5, normalize=true])\n\nAccess the co-occurrence matrix of the CooMatrix associated with the entity. The CooMatrix{T} will first have to be created in order for the actual matrix to be accessed.\n\n\n\n\n\n"
},

{
    "location": "api/#StringAnalysis.cosine",
    "page": "API Reference",
    "title": "StringAnalysis.cosine",
    "category": "function",
    "text": "cosine(model, docs, doc, n=10)\n\nReturn the positions of the n closest neighboring documents to doc found in docs. docs can be a corpus or document term matrix. The vector representations of docs and doc are obtained with the model which can be either a LSAModel or RPModel.\n\n\n\n\n\n"
},

{
    "location": "api/#StringAnalysis.dtm-Tuple{DocumentTermMatrix}",
    "page": "API Reference",
    "title": "StringAnalysis.dtm",
    "category": "method",
    "text": "dtm(d::DocumentTermMatrix)\n\nAccess the matrix of a DocumentTermMatrix d.\n\n\n\n\n\n"
},

{
    "location": "api/#StringAnalysis.dtm-Union{Tuple{Corpus}, Tuple{T}, Tuple{Corpus,Type{T}}} where T<:Real",
    "page": "API Reference",
    "title": "StringAnalysis.dtm",
    "category": "method",
    "text": "dtm(crps::Corpus, eltype::Type{T}=DEFAULT_DTM_TYPE [; tokenizer=DEFAULT_TOKENIZER])\n\nAccess the matrix of the DTM associated with the corpus crps. The DocumentTermMatrix{T} will first have to be created in order for the actual matrix to be accessed.\n\n\n\n\n\n"
},

{
    "location": "api/#StringAnalysis.dtv-Union{Tuple{T}, Tuple{Any,OrderedDict{String,Int64}}, Tuple{Any,OrderedDict{String,Int64},Type{T}}} where T<:Real",
    "page": "API Reference",
    "title": "StringAnalysis.dtv",
    "category": "method",
    "text": "dtv(d, lex::OrderedDict{String,Int}, eltype::Type{T}=DEFAULT_DTM_TYPE [; tokenizer=DEFAULT_TOKENIZER])\n\nCreates a document-term-vector with elements of type T for document d using the lexicon lex. d can be an AbstractString or an AbstractDocument.\n\n\n\n\n\n"
},

{
    "location": "api/#StringAnalysis.dtv-Union{Tuple{T}, Tuple{Corpus,Int64}, Tuple{Corpus,Int64,Type{T}}} where T<:Real",
    "page": "API Reference",
    "title": "StringAnalysis.dtv",
    "category": "method",
    "text": "dtv(crps::Corpus, idx::Int, eltype::Type{T}=DEFAULT_DTM_TYPE [; tokenizer=DEFAULT_TOKENIZER])\n\nCreates a document-term-vector with elements of type T for document idx of the corpus crps.\n\n\n\n\n\n"
},

{
    "location": "api/#StringAnalysis.dtv_regex-Union{Tuple{T}, Tuple{Any,OrderedDict{String,Int64}}, Tuple{Any,OrderedDict{String,Int64},Type{T}}} where T<:Real",
    "page": "API Reference",
    "title": "StringAnalysis.dtv_regex",
    "category": "method",
    "text": "dtv_regex(d, lex::OrderedDict{String,Int}, eltype::Type{T}=DEFAULT_DTM_TYPE [; tokenizer=DEFAULT_TOKENIZER])\n\nCreates a document-term-vector with elements of type T for document d using the lexicon lex. The tokens of document d are assumed to be regular expressions in text format. d can be an AbstractString or an AbstractDocument.\n\nExamples\n\njulia> dtv_regex(NGramDocument(\"a..b\"), OrderedDict(\"aaa\"=>1, \"aaab\"=>2, \"accb\"=>3, \"bbb\"=>4), Float32)\n4-element Array{Float32,1}:\n 0.0\n 1.0\n 1.0\n 0.0\n\n\n\n\n\n"
},

{
    "location": "api/#StringAnalysis.each_dtv-Union{Tuple{Corpus}, Tuple{U}} where U<:Real",
    "page": "API Reference",
    "title": "StringAnalysis.each_dtv",
    "category": "method",
    "text": "each_dtv(crps::Corpus [; eltype::Type{U}=DEFAULT_DTM_TYPE, tokenizer=DEFAULT_TOKENIZER])\n\nIterates through the columns of the DTM of the corpus crps without constructing it. Useful when the DTM would not fit in memory. eltype specifies the element type of the generated vectors.\n\n\n\n\n\n"
},

{
    "location": "api/#StringAnalysis.each_hash_dtv-Union{Tuple{Corpus}, Tuple{U}} where U<:Real",
    "page": "API Reference",
    "title": "StringAnalysis.each_hash_dtv",
    "category": "method",
    "text": "each_hash_dtv(crps::Corpus [; eltype::Type{U}=DEFAULT_DTM_TYPE, tokenizer=DEFAULT_TOKENIZER])\n\nIterates through the columns of the hashed DTM of the corpus crps without constructing it. Useful when the DTM would not fit in memory. eltype specifies the element type of the generated vectors.\n\n\n\n\n\n"
},

{
    "location": "api/#StringAnalysis.embed_document-Union{Tuple{H}, Tuple{A}, Tuple{T}, Tuple{S}, Tuple{LSAModel{S,T,A,H},AbstractDocument}} where H where A where T where S",
    "page": "API Reference",
    "title": "StringAnalysis.embed_document",
    "category": "method",
    "text": "embed_document(lm, doc)\n\nReturn the vector representation of doc, obtained using the LSA model lm. doc can be an AbstractDocument, Corpus or DTV or DTM.\n\n\n\n\n\n"
},

{
    "location": "api/#StringAnalysis.embed_document-Union{Tuple{H}, Tuple{A}, Tuple{T}, Tuple{S}, Tuple{RPModel{S,T,A,H},AbstractDocument}} where H where A where T where S",
    "page": "API Reference",
    "title": "StringAnalysis.embed_document",
    "category": "method",
    "text": "embed_document(rpm, doc)\n\nReturn the vector representation of doc, obtained using the random projection model rpm. doc can be an AbstractDocument, Corpus or DTV or DTM.\n\n\n\n\n\n"
},

{
    "location": "api/#StringAnalysis.frequent_terms",
    "page": "API Reference",
    "title": "StringAnalysis.frequent_terms",
    "category": "function",
    "text": "frequent_terms(doc, alpha)\n\nReturns a vector with frequent terms in the document doc. The parameter alpha indicates the sparsity threshold (a frequency <= alpha means sparse).\n\n\n\n\n\n"
},

{
    "location": "api/#StringAnalysis.frequent_terms",
    "page": "API Reference",
    "title": "StringAnalysis.frequent_terms",
    "category": "function",
    "text": "frequent_terms(crps::Corpus, alpha)\n\nReturns a vector with frequent terms among all documents. The parameter alpha indicates the sparsity threshold (a frequency <= alpha means sparse).\n\n\n\n\n\n"
},

{
    "location": "api/#StringAnalysis.get_vector-Union{Tuple{H}, Tuple{A}, Tuple{T}, Tuple{S}, Tuple{LSAModel{S,T,A,H},Any}} where H where A where T where S",
    "page": "API Reference",
    "title": "StringAnalysis.get_vector",
    "category": "method",
    "text": "get_vector(lm, word)\n\nReturns the vector representation of word from the LSA model lm.\n\n\n\n\n\n"
},

{
    "location": "api/#StringAnalysis.get_vector-Union{Tuple{H}, Tuple{A}, Tuple{T}, Tuple{S}, Tuple{RPModel{S,T,A,H},Any}} where H where A where T where S",
    "page": "API Reference",
    "title": "StringAnalysis.get_vector",
    "category": "method",
    "text": "get_vector(rpm, word)\n\nReturns the random projection vector corresponding to word in the random projection model rpm.\n\n\n\n\n\n"
},

{
    "location": "api/#StringAnalysis.hash_dtm-Union{Tuple{T}, Tuple{Corpus,TextHashFunction}, Tuple{Corpus,TextHashFunction,Type{T}}} where T<:Real",
    "page": "API Reference",
    "title": "StringAnalysis.hash_dtm",
    "category": "method",
    "text": "hash_dtm(crps::Corpus [,h::TextHashFunction], eltype::Type{T}=DEFAULT_DTM_TYPE [; tokenizer=DEFAULT_TOKENIZER])\n\nCreates a hashed DTM with elements of type T for corpus crps using the the hashing function h. If h is missing, the hash function of the Corpus is used.\n\n\n\n\n\n"
},

{
    "location": "api/#StringAnalysis.hash_dtv-Union{Tuple{T}, Tuple{Any,TextHashFunction}, Tuple{Any,TextHashFunction,Type{T}}} where T<:Real",
    "page": "API Reference",
    "title": "StringAnalysis.hash_dtv",
    "category": "method",
    "text": "hash_dtv(d, h::TextHashFunction, eltype::Type{T}=DEFAULT_DTM_TYPE [; tokenizer=DEFAULT_TOKENIZER])\n\nCreates a hashed document-term-vector with elements of type T for document d using the hashing function h. d can be an AbstractString or an AbstractDocument.\n\n\n\n\n\n"
},

{
    "location": "api/#StringAnalysis.in_vocabulary-Tuple{LSAModel,AbstractString}",
    "page": "API Reference",
    "title": "StringAnalysis.in_vocabulary",
    "category": "method",
    "text": "in_vocabulary(lm, word)\n\nReturn true if word is part of the vocabulary of the LSA model lm and false otherwise.\n\n\n\n\n\n"
},

{
    "location": "api/#StringAnalysis.in_vocabulary-Tuple{RPModel,AbstractString}",
    "page": "API Reference",
    "title": "StringAnalysis.in_vocabulary",
    "category": "method",
    "text": "in_vocabulary(rpm, word)\n\nReturn true if word is part of the vocabulary of the random projection model rpm and false otherwise.\n\n\n\n\n\n"
},

{
    "location": "api/#StringAnalysis.index-Tuple{LSAModel,Any}",
    "page": "API Reference",
    "title": "StringAnalysis.index",
    "category": "method",
    "text": "index(lm, word)\n\nReturn the index of word from the LSA model lm.\n\n\n\n\n\n"
},

{
    "location": "api/#StringAnalysis.index-Tuple{RPModel,Any}",
    "page": "API Reference",
    "title": "StringAnalysis.index",
    "category": "method",
    "text": "index(rpm, word)\n\nReturn the index of word from the random projection model rpm.\n\n\n\n\n\n"
},

{
    "location": "api/#StringAnalysis.lda-Tuple{DocumentTermMatrix,Int64,Int64,Float64,Float64}",
    "page": "API Reference",
    "title": "StringAnalysis.lda",
    "category": "method",
    "text": "ϕ, θ = lda(dtm::DocumentTermMatrix, ntopics::Int, iterations::Int, α::Float64, β::Float64)\n\nPerform Latent Dirichlet allocation.\n\nArguments\n\nα Dirichlet dist. hyperparameter for topic distribution per document. α<1 yields a sparse topic mixture for each document. α>1 yields a more uniform topic mixture for each document.\nβ Dirichlet dist. hyperparameter for word distribution per topic. β<1 yields a sparse word mixture for each topic. β>1 yields a more uniform word mixture for each topic.\n\nReturn values\n\nϕ: ntopics × nwords Sparse matrix of probabilities s.t. sum(ϕ, 1) == 1\nθ: ntopics × ndocs Dense matrix of probabilities s.t. sum(θ, 1) == 1\n\n\n\n\n\n"
},

{
    "location": "api/#StringAnalysis.load_lsa_model-Union{Tuple{AbstractString}, Tuple{T}, Tuple{AbstractString,Type{T}}} where T<:AbstractFloat",
    "page": "API Reference",
    "title": "StringAnalysis.load_lsa_model",
    "category": "method",
    "text": "load_lsa_model(filename, eltype; [sparse=false])\n\nLoads an LSA model from filename into an LSA model object. The embeddings matrix element type is specified by eltype (default DEFAULT_FLOAT_TYPE) while the keyword argument sparse specifies whether the matrix should be sparse or not.\n\n\n\n\n\n"
},

{
    "location": "api/#StringAnalysis.load_rp_model-Union{Tuple{AbstractString}, Tuple{T}, Tuple{AbstractString,Type{T}}} where T<:AbstractFloat",
    "page": "API Reference",
    "title": "StringAnalysis.load_rp_model",
    "category": "method",
    "text": "load_rp_model(filename, eltype; [sparse=true])\n\nLoads an random projection model from filename into an random projection model object. The projection matrix element type is specified by eltype (default DEFAULT_FLOAT_TYPE) while the keyword argument sparse specifies whether the matrix should be sparse or not.\n\n\n\n\n\n"
},

{
    "location": "api/#StringAnalysis.lsa-Union{Tuple{DocumentTermMatrix{T}}, Tuple{T}} where T<:AbstractFloat",
    "page": "API Reference",
    "title": "StringAnalysis.lsa",
    "category": "method",
    "text": "lsa(X [;k=<num documents>, stats=:tfidf, κ=2, β=0.75, tol=1e-15])\n\nConstructs a LSA model. The input X can be a Corpus or a DocumentTermMatrix. Use ?LSAModel for more details. Vector components smaller than tol will be zeroed out.\n\n\n\n\n\n"
},

{
    "location": "api/#StringAnalysis.ngrams",
    "page": "API Reference",
    "title": "StringAnalysis.ngrams",
    "category": "function",
    "text": "ngrams(d, n=DEFAULT_GRAM_COMPLEXITY [; tokenizer=DEFAULT_TOKENIZER])\n\nAccess the document text of d as n-gram counts. The ngrams contain at most n tokens which are obtained using tokenizer.\n\n\n\n\n\n"
},

{
    "location": "api/#StringAnalysis.ngrams!-Union{Tuple{T}, Tuple{NGramDocument{T},Dict{T,Int64}}} where T<:AbstractString",
    "page": "API Reference",
    "title": "StringAnalysis.ngrams!",
    "category": "method",
    "text": "ngrams!(d, new_ngrams)\n\nReplace the original n-grams of document d with new_ngrams.\n\n\n\n\n\n"
},

{
    "location": "api/#StringAnalysis.rp-Union{Tuple{DocumentTermMatrix{T}}, Tuple{T}} where T<:AbstractFloat",
    "page": "API Reference",
    "title": "StringAnalysis.rp",
    "category": "method",
    "text": "rp(X [;k=m, density=1/sqrt(k), stats=:tfidf, κ=2, β=0.75])\n\nConstructs a random projection model. The input X can be a Corpus or a DocumentTermMatrix with m words in the lexicon. The model does not store the corpus or DTM document embeddings, just the projection matrix. Use ?RPModel for more details.\n\n\n\n\n\n"
},

{
    "location": "api/#StringAnalysis.save_lsa_model-Union{Tuple{H}, Tuple{A}, Tuple{T}, Tuple{S}, Tuple{LSAModel{S,T,A,H},AbstractString}} where H where A where T where S",
    "page": "API Reference",
    "title": "StringAnalysis.save_lsa_model",
    "category": "method",
    "text": "save(lm, filename)\n\nSaves an LSA model lm to disc in file filename.\n\n\n\n\n\n"
},

{
    "location": "api/#StringAnalysis.save_rp_model-Union{Tuple{H}, Tuple{A}, Tuple{T}, Tuple{S}, Tuple{RPModel{S,T,A,H},AbstractString}} where H where A where T where S",
    "page": "API Reference",
    "title": "StringAnalysis.save_rp_model",
    "category": "method",
    "text": "save_rp_model(rpm, filename)\n\nSaves an random projection model rpm to disc in file filename.\n\n\n\n\n\n"
},

{
    "location": "api/#StringAnalysis.sentence_tokenize-Union{Tuple{T}, Tuple{T}} where T<:AbstractString",
    "page": "API Reference",
    "title": "StringAnalysis.sentence_tokenize",
    "category": "method",
    "text": "sentence_tokenize([lang,] s)\n\nSplits string s into sentences using WordTokenizers.split_sentences function to perform the tokenization. If a language lang is provided, it ignores it ;)\n\n\n\n\n\n"
},

{
    "location": "api/#StringAnalysis.similarity-Tuple{Any,Any,Any}",
    "page": "API Reference",
    "title": "StringAnalysis.similarity",
    "category": "method",
    "text": "similarity(model, doc1, doc2)\n\nReturn the cosine similarity value between two documents doc1 and doc2 whose vector representations have been obtained using the model, which can be either a LSAModel or RPModel.\n\n\n\n\n\n"
},

{
    "location": "api/#StringAnalysis.sparse_terms",
    "page": "API Reference",
    "title": "StringAnalysis.sparse_terms",
    "category": "function",
    "text": "sparse_terms(doc, alpha)\n\nReturns a vector with rare terms in the document doc. The parameter alpha indicates the sparsity threshold (a frequency <= alpha means sparse).\n\n\n\n\n\n"
},

{
    "location": "api/#StringAnalysis.sparse_terms",
    "page": "API Reference",
    "title": "StringAnalysis.sparse_terms",
    "category": "function",
    "text": "sparse_terms(crps::Corpus, alpha)\n\nReturns a vector with rare terms among all documents. The parameter alpha indicates the sparsity threshold (a frequency <= alpha means sparse).\n\n\n\n\n\n"
},

{
    "location": "api/#StringAnalysis.text!-Union{Tuple{T}, Tuple{StringDocument{T},T}} where T<:AbstractString",
    "page": "API Reference",
    "title": "StringAnalysis.text!",
    "category": "method",
    "text": "text!(d, new_text)\n\nReplace the original text of document d with new_text.\n\n\n\n\n\n"
},

{
    "location": "api/#StringAnalysis.text-Tuple{AbstractString}",
    "page": "API Reference",
    "title": "StringAnalysis.text",
    "category": "method",
    "text": "text(d)\n\nAccess the text of document d if possible.\n\n\n\n\n\n"
},

{
    "location": "api/#StringAnalysis.tokenize-Tuple{Any}",
    "page": "API Reference",
    "title": "StringAnalysis.tokenize",
    "category": "method",
    "text": "\"     tokenize(doc [;method, splitter])\n\nTokenizes the document doc based on the mehtod (default :default, i.e. a WordTokenizers.jl tokenizer) and the splitter, which is a Regex used if method=:stringanalysis.\n\n\n\n\n\n"
},

{
    "location": "api/#StringAnalysis.tokens!-Union{Tuple{T}, Tuple{TokenDocument{T},Array{T,1}}} where T<:AbstractString",
    "page": "API Reference",
    "title": "StringAnalysis.tokens!",
    "category": "method",
    "text": "tokens!(d, new_tokens)\n\nReplace the original tokens of document d with new_tokens.\n\n\n\n\n\n"
},

{
    "location": "api/#StringAnalysis.tokens-Tuple{AbstractString}",
    "page": "API Reference",
    "title": "StringAnalysis.tokens",
    "category": "method",
    "text": "tokens(d [; method=DEFAULT_TOKENIZER])\n\nAccess the tokens of document d as a token array. The method keyword argument specifies the type of tokenization to perform. Available options are :default and :stringanalysis.\n\n\n\n\n\n"
},

{
    "location": "api/#StringAnalysis.vocabulary-Tuple{LSAModel}",
    "page": "API Reference",
    "title": "StringAnalysis.vocabulary",
    "category": "method",
    "text": "vocabulary(lm)\n\nReturn the vocabulary as a vector of words of the LSA model lm.\n\n\n\n\n\n"
},

{
    "location": "api/#StringAnalysis.vocabulary-Tuple{RPModel}",
    "page": "API Reference",
    "title": "StringAnalysis.vocabulary",
    "category": "method",
    "text": "vocabulary(rpm)\n\nReturn the vocabulary as a vector of words of the random projection model rpm.\n\n\n\n\n\n"
},

{
    "location": "api/#Base.size-Tuple{LSAModel}",
    "page": "API Reference",
    "title": "Base.size",
    "category": "method",
    "text": "size(lm)\n\nReturn a tuple containin input and output dimensionalities of the LSA model lm.\n\n\n\n\n\n"
},

{
    "location": "api/#Base.size-Tuple{RPModel}",
    "page": "API Reference",
    "title": "Base.size",
    "category": "method",
    "text": "size(rpm)\n\nReturn a tuple containing the input data and projection sub-space dimensionalities of the random projection model rpm.\n\n\n\n\n\n"
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
    "location": "api/#StringAnalysis.abstract_convert-Union{Tuple{AbstractDocument}, Tuple{T}, Tuple{AbstractDocument,Union{Nothing, Type{T}}}} where T<:AbstractString",
    "page": "API Reference",
    "title": "StringAnalysis.abstract_convert",
    "category": "method",
    "text": "abstract_convert(document::AbstractDocument, parameter::Union{Nothing, Type{T}})\n\nTries converting document::AbstractDocument to one of the concrete types with witch StringAnalysis works i.e. StringDocument{T}, TokenDocument{T}, NGramDocument{T}. A user-defined convert method between the typeof(document) and the concrete types should be defined.\n\n\n\n\n\n"
},

{
    "location": "api/#StringAnalysis.columnindices",
    "page": "API Reference",
    "title": "StringAnalysis.columnindices",
    "category": "function",
    "text": "columnindices(terms)\n\nIdentical to rowindices. Returns a dictionary that maps each term from the vector terms to a integer idex.\n\n\n\n\n\n"
},

{
    "location": "api/#StringAnalysis.coo_matrix-Union{Tuple{T}, Tuple{Type{T},Array{#s12,1} where #s12<:AbstractString,OrderedDict{#s51,Int64} where #s51<:AbstractString,Int64}, Tuple{Type{T},Array{#s52,1} where #s52<:AbstractString,OrderedDict{#s53,Int64} where #s53<:AbstractString,Int64,Bool}} where T<:AbstractFloat",
    "page": "API Reference",
    "title": "StringAnalysis.coo_matrix",
    "category": "method",
    "text": "coo_matrix(::Type{T}, doc::Vector{AbstractString}, vocab::OrderedDict{AbstractString, Int}, window::Int, normalize::Bool)\n\nBasic low-level function that calculates the co-occurence matrix of a document. Returns a sparse co-occurence matrix sized n × n where n = length(vocab) with elements of type T. The document doc is represented by a vector of its terms (in order). The keywordswindowandnormalize` indicate the size of the sliding word window in which co-occurrences are counted and whether to normalize of not the counts by the distance between word positions.\n\nExamples\n\njulia> using StringAnalysis\n       doc = StringDocument(\"This is a text about an apple. There are many texts about apples.\")\n       docv = tokenize(text(doc))\n       vocab = OrderedDict(\"This\"=>1, \"is\"=>2, \"apple.\"=>3)\n       StringAnalysis.coo_matrix(Float16, docv, vocab, 5, true)\n3×3 SparseArrays.SparseMatrixCSC{Float16,Int64} with 4 stored entries:\n  [2, 1]  =  2.0\n  [1, 2]  =  2.0\n  [3, 2]  =  0.3999\n  [2, 3]  =  0.3999\n\n\n\n\n\n"
},

{
    "location": "api/#StringAnalysis.embed_word-Tuple{LSAModel,Any}",
    "page": "API Reference",
    "title": "StringAnalysis.embed_word",
    "category": "method",
    "text": "embed_word(lm, word)\n\nReturn the vector representation of word using the LSA model lm.\n\n\n\n\n\n"
},

{
    "location": "api/#StringAnalysis.random_projection_matrix-Union{Tuple{T}, Tuple{Int64,Int64,Type{T},Float64}} where T<:AbstractFloat",
    "page": "API Reference",
    "title": "StringAnalysis.random_projection_matrix",
    "category": "method",
    "text": "random_projection_matrix(k::Int, m::Int, eltype::Type{T<:AbstractFloat}, density::Float64)\n\nBuilds a k×m sparse random projection matrix with elements of type T and a non-zero element frequency of density. k and m are the output and input dimensionalities.\n\nMatrix Probabilities\n\nIf we note s = 1 / density, the components of the random matrix are drawn from:\n\n-sqrt(s) / sqrt(k) with probability 1/2s\n0 with probability 1 - 1/s\n+sqrt(s) / sqrt(k)   with probability 1/2s\n\nNo projection hack\n\nIf k<=0 no projection is performed and the function returns an identity matrix sized m×m with elements of type T. This is useful if one does not want to embed documents but rather calculate term frequencies, BM25 and other statistical indicators (similar to dtv).\n\n\n\n\n\n"
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
    "location": "api/#StringAnalysis.rowindices-Tuple{Array{String,1}}",
    "page": "API Reference",
    "title": "StringAnalysis.rowindices",
    "category": "method",
    "text": "rowindices(terms)\n\nReturns a dictionary that maps each term from the vector terms to a integer idex.\n\n\n\n\n\n"
},

{
    "location": "api/#StringAnalysis.tokenize_default-Union{Tuple{T}, Tuple{T}} where T<:AbstractString",
    "page": "API Reference",
    "title": "StringAnalysis.tokenize_default",
    "category": "method",
    "text": "tokenize_default([lang,] s)\n\nSplits string s into tokens on whitespace using WordTokenizers.tokenize function to perform the tokenization. If a language lang is provided, it ignores it ;)\n\n\n\n\n\n"
},

{
    "location": "api/#StringAnalysis.tokenize_stringanalysis-Union{Tuple{S}, Tuple{S}} where S<:AbstractString",
    "page": "API Reference",
    "title": "StringAnalysis.tokenize_stringanalysis",
    "category": "method",
    "text": "tokenize_stringanalysis(doc [;splitter])\n\nFunction that quickly tokenizes doc based on the splitting pattern specified by splitter::RegEx. Supported types for doc are: AbstractString, Vector{AbstractString}, StringDocument and NGramDocument.\n\n\n\n\n\n"
},

{
    "location": "api/#",
    "page": "API Reference",
    "title": "API Reference",
    "category": "page",
    "text": "Modules = [StringAnalysis]"
},

]}
