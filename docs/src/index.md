```@meta
CurrentModule=StringAnalysis
```

# Introduction

StringAnalysis is a package for working with strings and text. It is a hard-fork from [TextAnalysis.jl](https://github.com/JuliaText/TextAnalysis.jl) designed to provide a more powerful, faster and orthogonal API.

**Note**: This documentation is still under construction and incomplete. For an overview of the basic capabilities of the package, consult the - still relevant - [TextAnalysis.jl documentation](http://juliatext.github.io/TextAnalysis.jl/).

## What is new?
This package brings several changes over `TextAnalysis.jl`:
 - Simpler API (less exported methods)
 - Improved test coverage
 - Parametrized many of the objects i.e. `DocumentTermMatrix`, `AbstractDocument` etc
 - Extended `DocumentMetadata` with new fields
 - Many of the repetitive functions are now automatically generated (see [metadata.jl](https://github.com/zgornel/StringAnalysis.jl/blob/master/src/metadata.jl), [preprocessing.jl](https://github.com/zgornel/StringAnalysis.jl/blob/master/src/preprocessing.jl))
 - Re-factored the text preprocessing API
 - Improved latent semantic analysis (LSA)
 - `each_dtv`, `each_hash_dtv` iterators support vector element type specification
 - Added [Okapi BM25](https://en.wikipedia.org/wiki/Okapi_BM25) statistic
 - Many bugfixes and small extensions

## Installation

Installation can be performed from either inside or outside Julia.

### Git cloning
The `StringAnalysis` repository can be downloaded through git:
```
$ git clone https://github.com/zgornel/StringAnalysis.jl
```

### Julia REPL
The package can be installed from inside Julia with:
```
using Pkg
Pkg.add(StringAnalysis)
```
will download the latest registered build of the package and add it to the current active development environment.

