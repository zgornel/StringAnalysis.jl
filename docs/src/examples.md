# Usage examples

## Document creation
Documents are simple wrappers around basic structures that contain text. The underlying data representation can be simple strings, dictionaries or vectors of strings. All document types are subtypes of the parametric type AbstractDocument{T}` where `T<:AbstractString`.
```@repl index
using StringAnalysis

sd = StringDocument("this is a string document")
nd = NGramDocument("this is a ngram document")
td = TokenDocument("this is a token document")
# fd = FileDocument("/some/file") # works the same way ...
```

## Document creation (continued...)
The string type can be explicity enforced:
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
