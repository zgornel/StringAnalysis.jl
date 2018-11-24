# StringAnalysis

This is a hard fork of the [TextAnalysis](https://github.com/JuliaText/TextAnalysis.jl) package, designed to provide a simpler, faster and more orthogonal interface for analyzing and working with strings and text documents.

[![License](http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat)](LICENSE.md)
[![Build Status](https://travis-ci.org/zgornel/StringAnalysis.jl.svg?branch=master)](https://travis-ci.org/zgornel/StringAnalysis.jl)
[![Coverage Status](https://coveralls.io/repos/github/zgornel/StringAnalysis.jl/badge.svg?branch=master)](https://coveralls.io/github/zgornel/StringAnalysis.jl?branch=master)


## Differences with **TextAnalysis**
The aim is to make the code more [elegant](https://nakamotoinstitute.org/static/docs/taoup.pdf) (i.e. more powerful & simple). The main work focuses on:
- Cleaner API i.e. remove all `remove_<stuff>!` methods, employ only `prepare` or `prepare!`
- Remove non-core functionality (sentiment analysis, summarization) and weak dependencies i.e. `Printf.jl`, `DataFrames.jl`, `Flux.jl`, `BSON.jl`, `JSON.jl` etc.
- Remove compatibility with Julia 0.6 and below
- Increase test coverage
- Fix Bugs

This is work in progress and bugs may still be present...¯\\_(ツ)_/¯


## Introduction
At this point in development, [1] should provide a rough guide on the library's capabilities. While not a design aim in itself, the package tries to keep a similar API to **TextAnalysis** and bring gradual improvements.


## License

This code has an MIT license and therefore it is free.


## Credits

This work would not have been possible without the efforts of the **TextAnalysis** package developers.


## References

[1] [TextAnalysis package documentation](http://juliatext.github.io/TextAnalysis.jl/)
