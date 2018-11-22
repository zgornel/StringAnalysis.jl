# StringAnalysis.jl

This is a hard fork of [TextAnalysis.jl](https://github.com/JuliaText/TextAnalysis.jl) designed to provide a simpler, faster and more orthogonal interface for analyzing and working with strings and text documents.

[![License](http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat)](LICENSE.md)
[![Build Status](https://travis-ci.org/zgornel/StringAnalysis.jl.svg?branch=master)](https://travis-ci.org/zgornel/StringAnalysis.jl)
[![Coverage Status](https://coveralls.io/repos/github/zgornel/StringAnalysis.jl/badge.svg?branch=master)](https://coveralls.io/github/zgornel/StringAnalysis.jl?branch=master)


## Improvements, new features over **TextAnalysis.jl**
This is work in progress and under heavy development...¯\\_(ツ)_/¯:
- [ ] **WIP** Cleaner API
- [x] Lower the number of dependencies i.e. `Printf.jl`, `DataFrames.jl`, `Flux.jl`, `JSON.jl`
- [x] Remove deprecations, sentiment analysis
- [ ] **WIP** Add faster version of summarization (`LightGraphs.jl`-based [TextRank](https://en.wikipedia.org/wiki/Automatic_summarization#Unsupervised_approach:_TextRank))
- [ ] **WIP** Improved code documentation i.e. docstrings, comments and show methods
- [ ] Bugfixes


## Introduction
At this point in development, [1] should provide a rough guide on the library's capabilities. While not a design aim in itself, the package tries to keep a similar API to **TextAnalysis** and bring gradual improvements.


## License

This code has an MIT license and therefore it is free.


## Credits

This work would not have been possible without the efforts of the **TextAnalysis.jl** package developers.


## References

[1] [TextAnalysis.jl documentation](http://juliatext.github.io/TextAnalysis.jl/)
