# Extending the document model

Sometimes it may make sense to define new document types with whom to operate and use only some functionality of this package. For example, let us define two new document types, a `SimpleDocument` with no metadata
```@repl index
using StringAnalysis
struct NoMetadata <: AbstractMetadata end

struct SimpleDocument{T<:AbstractString} <: AbstractDocument{T, NoMetadata}
    text::T
end
```
and a `ConferencePublication` with only a limited number of metadata fields.

```@repl index
struct ConferenceMetadata <: AbstractMetadata
    name::String
    authors::String
    conference::String
end

struct ConferencePublication{T<:AbstractString} <: AbstractDocument{T, ConferenceMetadata}
    text::T
    metadata::ConferenceMetadata
end
```

At this point, one can create documents and use basic containers along with other standard documents of the package:
```@repl index
sd = SimpleDocument("a simple document")
cmd = ConferenceMetadata("Tile Inc.","John P. Doe","IEEE Conference on Unknown Facts")
cd = ConferencePublication("publication text", cmd)

doc = StringDocument("a document")

docs = [sd, cd, doc]
```

However, creating a `Corpus` fails because no conversion method exists between the new document types and any of the standardized ones `StringDocument`, `NGramDocument` etc.
```@repl index
Corpus(AbstractDocument[sd, cd, doc])
```

By defining at least one conversion method to a known type,
```@repl index
Base.convert(::Type{NGramDocument{String}}, doc::SimpleDocument) =
    NGramDocument{String}(doc.text)
Base.convert(::Type{NGramDocument{String}}, doc::ConferencePublication) = begin
    new_doc = NGramDocument{String}(doc.text)
    new_doc.metadata.name = doc.metadata.name
    new_doc.metadata.author = doc.metadata.authors
    new_doc.metadata.note = doc.metadata.conference
    return new_doc
end
```
the `Corpus` can be created and the rest of the functionality of the package i.e. numerical operations, can be employed on the document data.
```@repl index
crps = Corpus(AbstractDocument[sd, cd, doc])
metadata.(doc for doc in crps)
DocumentTermMatrix(crps)
```
The `SimpleDocument` and `ConferencePublication` were both converted to `NGramDocument`s since this was the only conversion method available. If more would be available, the priority of conversion is given by the code in the `abstract_convert` function. Generally, one single conversion method suffices.

