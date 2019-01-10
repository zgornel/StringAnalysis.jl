# Usage examples

## Documents
Documents are simple wrappers around basic structures that contain text. The underlying data representation can be simple strings, dictionaries or vectors of strings. All document types are subtypes of the parametric type `AbstractDocument{T}` where `T<:AbstractString`.
```@repl index
using StringAnalysis

sd = StringDocument("this is a string document")
nd = NGramDocument("this is a ngram document")
td = TokenDocument("this is a token document")
# fd = FileDocument("/some/file") # works the same way ...
```

## Documents and types
The string type can be explicitly enforced:
```@repl index
nd = NGramDocument{String}("this is a ngram document")
ngrams(nd)
td = TokenDocument{String}("this is a token document")
tokens(td)
```
Conversion methods are available to switch between document types (the type parameter has to be specified as well).
```@repl index
convert(TokenDocument{SubString}, StringDocument("some text"))
convert(NGramDocument{String}, StringDocument("some more text"))
```

## Metadata
Alongside the text data, documents also contain metadata.
```@repl index
doc = StringDocument("this is another document")
metadata(doc)
fieldnames(typeof(metadata(doc)))
```
Metadata fields can be modified through methods bearing the same name as the metadata field. Note that these methods are not explicitly exported.
```@repl index
StringAnalysis.id!(doc, "doc1");
StringAnalysis.author!(doc, "Corneliu C.");
StringAnalysis.name!(doc, "A simple document");
StringAnalysis.edition_year!(doc, "2019");
StringAnalysis.published_year!(doc, "2019");
metadata(doc)
```

## Corpus
A corpus is an object that holds a bunch of documents together.
```@repl index
docs = [sd, nd, td]
crps = Corpus(docs)
crps.documents
```
The corpus can be 'standardized' to hold the same type of document.
```@repl index
standardize!(crps, NGramDocument{String})
crps.documents
```
however, the corpus has to created from an `AbstractDocument` document vector for the standardization to work (`AbstractDocument{T}` vectors are converted to a `Union` of all documents types parametrized by `T` during `Corpus` construction):
```@repl index
doc1 = StringDocument("one");
doc2 = StringDocument("two");
doc3 = TokenDocument("three");
standardize!(Corpus([doc1, doc3]), NGramDocument{String})  # works
standardize!(Corpus([doc1, doc2]), NGramDocument{String})  # fails because we have a Vector{StringDocument{T}}
standardize!(Corpus(AbstractDocument[doc1, doc2]), NGramDocument{String})  # works
```
The corpus can be also iterated through,
```@repl index
for (i,doc) in enumerate(crps)
    @show (i, doc)
end
```
indexed into,
```@repl index
doc = crps[1]
docs = crps[2:3]
```
and used as a container.
```@repl index
push!(crps, NGramDocument{String}("new document"))
doc4 = pop!(crps)
ngrams(doc4)
```

## The lexicon and inverse index
The `Corpus` object offers the ability of creating a [lexicon](https://en.wikipedia.org/wiki/Lexicon) and an [inverse index](https://en.wikipedia.org/wiki/Inverted_index) for the documents present. These are not created when the Corpus is created
```@repl index
crps.lexicon
crps.inverse_index
```
but instead have to be explicitly created:
```@repl index
update_lexicon!(crps)
crps.lexicon
update_inverse_index!(crps)
crps.inverse_index
```

## Preprocessing
The text preprocessing mainly consists of the `prepare` and `prepare!` functions and preprocessing flags which start mostly with `strip_` except for `stem_words`. The preprocessing function `prepare` works on `AbstractDocument`, `Corpus` and `AbstractString` types, returning new objects; `prepare!` works only on `AbstractDocument`s and `Corpus` as the strings are immutable.
```@repl index
str="This is a text containing words, some more words, a bit of punctuation and 1 number...";
sd = StringDocument(str);
flags = strip_punctuation|strip_articles|strip_punctuation|strip_whitespace
prepare(str, flags)
prepare!(sd, flags);
text(sd)
```
More extensive preprocessing examples can be viewed in `test/preprocessing.jl`.

One can strip parts of languages i.e. prepositions, articles in languages other than English (support provided from [Languages.jl](https://github.com/JuliaText/Languages.jl)):
```@repl index
using Languages
it = StringDocument("Quest'e un piccolo esempio di come si puo fare l'analisi");
StringAnalysis.language!(it, Languages.Italian());
prepare!(it, strip_articles|strip_prepositions|strip_whitespace);
it.text
```

## Features

### Document Term Matrix (DTM)
If a lexicon is present in the corpus, a [document term matrix (DTM)](https://en.wikipedia.org/wiki/Document-term_matrix) can be created. The DTM acts as a basis for word-document statistics, allowing for the representation of documents as numerical vectors. The DTM is created by calling the object constructor using as argument the corpus
```@repl index
M = DocumentTermMatrix(crps)
typeof(M)
M = DocumentTermMatrix{Int8}(crps)
typeof(M)
```
or the `dtm` function
```@repl index
M = dtm(crps, Int8);
Matrix(M)
```
It is important to note that the type parameter of the DTM object can be specified (also in the `dtm` function) but not specifically required. This can be useful in some cases for reducing memory requirements. The default element type of the DTM is specified by the constant `DEFAULT_DTM_TYPE` present in `src/defaults.jl`.

### Document Term Vectors (DTVs)
The individual rows of the DTM can also be generated iteratively whether a lexicon is present or not. If a lexicon is present, the `each_dtv` iterator allows the generation of the document vectors along with the control of the vector element type:
```@repl index
for dv in each_dtv(crps, eltype=Int8)
    @show dv
end
```

Alternatively, the vectors can be generated using the [hash trick](https://en.wikipedia.org/wiki/Feature_hashing). This is a form of dimensionality reduction as `cardinality` i.e. output dimension is much smaller than the dimension of the original DTM vectors, which is equal to the length of the lexicon. The `cardinality` is a keyword argument of the `Corpus` constructor. The hashed vector output type can be specified when building the iterator:
```@repl index
for dv in each_hash_dtv(Corpus(documents(crps), cardinality=5), eltype=Int8)
    @show dv
end
```
One can construct a 'hashed' version of the DTM as well:
```@repl index
hash_dtm(Corpus(documents(crps), cardinality=5), Int8)
```
The default `Corpus` cardinality is specified by the constant `DEFAULT_CARDINALITY` present in `src/defaults.jl`.

### TF, TF-IDF, BM25
From the DTM, three more document-word statistics can be constructed: the [term frequency](https://en.wikipedia.org/wiki/Tf%E2%80%93idf#Term_frequency_2), the [tf-idf (term frequency - inverse document frequency)](https://en.wikipedia.org/wiki/Tf%E2%80%93idf#Term_frequency%E2%80%93Inverse_document_frequency) and [Okapi BM25](https://en.wikipedia.org/wiki/Okapi_BM25) using the `tf`, `tf!`, `tf_idf`, `tf_idf!`, `bm_25` and `bm_25!` functions respectively. Their usage is very similar yet there exist several approaches one can take to constructing the output.

The following examples with use the term frequency i.e. `tf` and `tf!`. When calling the functions that end without a `!`, which do not require the specification of an output matrix, one does not control the output's element type. The default output type is defined by the constant `DEFAULT_FLOAT_TYPE = eltype(1.0)`:
```@repl index
M = DocumentTermMatrix(crps);
tfm = tf(M);
Matrix(tfm)
```
Control of the output matrix element type - which has to be a subtype of `AbstractFloat` - can be done only by using the in-place modification functions. One approach is to directly modify the DTM, provided that its elements are floating point numbers:
```@repl index
M = DocumentTermMatrix{Float16}(crps)
Matrix(M.dtm)
tf!(M.dtm);  # inplace modification
Matrix(M.dtm)
M = DocumentTermMatrix(crps)  # Int elements
tf!(M.dtm)  # fails because of Int elements
```
or, to provide a matrix output:
```@repl index
rows, cols = size(M.dtm);
tfm = zeros(Float16, rows, cols);
tf!(M.dtm, tfm);
tfm
```
One could also provide a sparse matrix output however it is important to note that in this case, the output matrix non-zero values have to correspond to the DTM's non-zero values:
```@repl index
using SparseArrays
rows, cols = size(M.dtm);
tfm = spzeros(Float16, rows, cols)
tfm[M.dtm .!= 0] .= 123;  # create explicitly non-zeros
tf!(M.dtm, tfm);
Matrix(tfm)
```

## Dimensionality reduction

### Random projections
In mathematics and statistics, random projection is a technique used to reduce the dimensionality of a set of points which lie in Euclidean space. Random projection methods are powerful methods known for their simplicity and less erroneous output compared with other methods. According to experimental results, random projection preserve distances well, but empirical results are sparse. They have been applied to many natural language tasks under the name of _random indexing_. The core idea behind random projection is given in the [Johnson-Lindenstrauss lemma](https://cseweb.ucsd.edu/~dasgupta/papers/jl.pdf) which states that if points in a vector space are of sufficiently high dimension, then they may be projected into a suitable lower-dimensional space in a way which approximately preserves the distances between the points [(Wikipedia)](https://en.wikipedia.org/wiki/Random_projection). 

The implementation here relies on the generalized sparse random projection matrix to generate a random projection model. For more details see the API documentation for `RPModel` and `random_projection_matrix`.
To construct a random projection matrix that maps `m` dimension to `k`, one can do
```@repl index
m = 10; k = 2; T = Float32;
density = 0.2;  # percentage of non-zero elements
R = StringAnalysis.random_projection_matrix(m, k, T, density)
```
Building a random projection model from a `DocumentTermMatrix` or `Corpus` is straightforward
```@repl index
M = DocumentTermMatrix{Float32}(crps)
model = RPModel(M, k=2, density=0.5, stats=:tf)
model2 = rp(crps, T, k=17, density=0.1, stats=:tfidf)
```
Once the model is created, one can reduce document term vector dimensionality. First, the document term vector is constructed using the `stats` keyword argument and subsequently, the vector is projected into the random sub-space:
```@repl index
doc = StringDocument("this is a new document")
embed_document(model, doc)
embed_document(model2, doc)
```
To embed a document term matrix, one only has to do
```@repl index
Matrix(M.dtm * model.R')  # 3 documents x 2 sub-space dimensions
Matrix(M.dtm * model2.R')  # 3 documents x 17 sub-space dimentsions
```

## Semantic Analysis

The semantic analysis of a corpus relates to the task of building structures that approximate the concepts present in its documents. It does not necessarily involve prior semantic understanding of the documents [(Wikipedia)](https://en.wikipedia.org/wiki/Semantic_analysis_(machine_learning)).

`StringAnalysis` provides two approaches of performing semantic analysis of a corpus: [latent semantic analysis (LSA)](http://lsa.colorado.edu/papers/JASIS.lsi.90.pdf) and [latent Dirichlet allocation (LDA)](http://jmlr.org/papers/volume3/blei03a/blei03a.pdf).

### Latent Semantic Analysis (LSA)
The following example gives a straightforward usage example of LSA. It is geared towards information retrieval (LSI) as it focuses on document comparison and embedding. Assuming a number of documents
```@repl index
doc1 = StringDocument("This is a text about an apple. There are many texts about apples.");
doc2 = StringDocument("Pears and apples are good but not exotic. An apple a day keeps the doctor away.");
doc3 = StringDocument("Fruits are good for you.");
doc4 = StringDocument("This phrase has nothing to do with the others...");
doc5 = StringDocument("Simple text, little info inside");
```
and creating a corpus and its DTM
```@repl index
crps = Corpus(AbstractDocument[doc1, doc2, doc3, doc4, doc5]);
prepare!(crps, strip_punctuation);
update_lexicon!(crps);
M = DocumentTermMatrix{Float32}(crps, sort(collect(keys(crps.lexicon))));
```
building an LSA model is straightforward:
```@repl index
lm = LSAModel(M, k=3, stats=:tf)
```
Once the model is created, it can be used to either embed documents
```@repl index
query = StringDocument("Apples and an exotic fruit.");
embed_document(lm, query)
```
search for matching documents
```@repl index
idxs, corrs = cosine(lm, query);

for (idx, corr) in zip(idxs, corrs)
    println("$corr -> \"$(crps[idx].text)\"");
end
```
or check for structure within the data
```@repl index
U, V = lm.U, lm.Vᵀ';
Matrix(U*U')  # document to document similarity
Matrix(V*V')  # term to term similarity
```
LSA models can be saved and retrieved to and from am easy to read and parse text format.
```@repl index
file = "model.txt"
lm
save_lsa_model(lm, file)  # model saved
print(join(readlines(file)[1:5], "\n"))  # first five lines
new_model = load_lsa_model(file, Float64)  # change element type
rm(file)
```

### Latent Dirichlet Allocation (LDA)
Documentation coming soon; check the API reference for information on the associated methods.
