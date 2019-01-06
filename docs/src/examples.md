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

## Features
If a lexicon is present in the corpus, a [document term matrix (DTM)](https://en.wikipedia.org/wiki/Document-term_matrix) can be created. The DTM acts as a basis for word-document statistics, allowing for the representation of documents as numerical vectors. The DTM is created by calling the object constructor using as argument the corpus
```@repl index
M = DocumentTermMatrix(crps)
typeof(M)
M = DocumentTermMatrix{Int8}(crps)
typeof(M)
```
or the `dtm` function (not recommended as the element type cannot be specified)
```@repl index
M = dtm(crps)
```
The default element type of the DTM is specified by the constant `DEFAULT_DTM_TYPE` present in `src/defaults.jl`.

The individual rows of the DTM can also be generated iteratively whether a lexicon is present or not. If a lexicon is present, the `each_dtv` iterator allows the generation of the document vectors along with the control of the vector element type:
```@repl index
for dv in each_dtv(crps, eltype=Int8)
    @show dv
end
```

Alternatively, the vectors can be generated using the [hash trick](https://en.wikipedia.org/wiki/Feature_hashing). The dimension of these vectors can be controlled through the `cardinality` keyword argument of the `Corpus` constructor while their type can be specified when building the iterator:
```@repl index
for dv in each_hash_dtv(Corpus(documents(crps), cardinality=5), eltype=Int8)
    @show dv
end
```
The default `Corpus` cardinality is specified by the constant `DEFAULT_CARDINALITY` present in `src/defaults.jl`.

## More features
From the DTM, three more document-word statistics can be constructed: the [term frequency](https://en.wikipedia.org/wiki/Tf%E2%80%93idf#Term_frequency_2), the [tf-idf (term frequency - inverse document frequency)](https://en.wikipedia.org/wiki/Tf%E2%80%93idf#Term_frequency%E2%80%93Inverse_document_frequency) and [Okapi BM25](https://en.wikipedia.org/wiki/Okapi_BM25) using the `tf`, `tf!`, `tf_idf`, `tf_idf!`, `bm_25` and `bm_25!` functions respectively. Their usage is very similar yet there exist several approaches one can take to constructing the output.

The following examples with use the term frequency i.e. `tf` and `tf!`. When calling the functions that end without a `!`, one does not control the element type, which is defined by the constant `DEFAULT_FLOAT_TYPE = eltype(1.0)`:
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
tf!(M.dtm)  # fails
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

## Pre-processing
The text preprocessing mainly consists of the `prepare` and `prepare!` functions and preprocessing flags which start mostly with `strip_` except for `stem_words`. The preprocessing function `prepare` works on `AbstractDocument`, `Corpus` and `AbstractString` types, returning new objects; `prepare!` works only on `AbstractDocument`s and `Corpus` as the strings are immutable.
```@repl index
str="This is a text containing words and a bit of punctuation...";
flags = strip_punctuation|strip_articles|strip_punctuation|strip_whitespace
prepare(str, flags)
sd = StringDocument(str);
prepare!(sd, flags);
text(sd)
```
More extensive preprocessing examples can be viewed in `test/preprocessing.jl`.

## Semantic Analysis

The semantic analysis of a corpus relates to the task of building structures that approximate the concepts present in its documents. It does not necessarily involve prior semantic understanding of the documents [(Wikipedia)](https://en.wikipedia.org/wiki/Semantic_analysis_(machine_learning)).

`StringAnalysis` provides two approaches of performing semantic analysis of a corpus: latent semantic analysis (LSA) and latent Dirichlet allocation (LDA).

### Latent Semantic Analysis (LSA)
Documentation coming soon. Check the API reference for more information. [LSA paper](http://lsa.colorado.edu/papers/JASIS.lsi.90.pdf)

### Latent Dirichlet Allocation (LDA)
Documentation coming soon. Check the API reference for more information. [LDA paper](http://jmlr.org/papers/volume3/blei03a/blei03a.pdf)
