## StringAnalysis Release Notes

v0.3.4
------
 - All forms of DTVs are sparse
 - DTMs, COOMs are immutable

v0.3.3
------
 - Performance improvementss

v0.3.2
------
 - DTM document vectors are columns
 - Tokenizer can be specified in some methods
 - Regex based DTV's
 - Additional documentation
 - svd fallback in LSA
 - COOM performance improvement
 - Bugfixes

v0.3.1
------
 - Added `:count` option to LSA, RP models
 - No projection hack for RP models
 - More documentation

v0.3.0
------
 - Added Co-occurrence matrix
 - Refined LSA, RP models
 - More embedding methods
 - Small bugfixes, improvements

v0.2.4
------
 - Added sparse random projections
 - Bugfixes

v0.2.3
------
 - Preprocessing improvements
 - Additional documentation

v0.2.2
------
 - LSA models can be saved/loaded
 - Small additions

v0.2.1
------
 - Improved LSA
 - Expanded online documentation

v0.2.0
------
 - Improved latent semantic analysis (LSA)
 - Online documentation with Documenter.jl

v0.1.1
------
 - Typing improvements
 - Added support for Vector element type in DTV iteration
 - Made `AbstractDocument` a parametric type
 - Extended test coverage
 - Bugfixes

v0.1.0
------
 - Many fixed bugs and inconsistencies
 - Added bm25 ranking, tweaked tf-idf
 - Extended tokenization and stemming methods
 - Extended pre-processing API
 - Extended document metadata
 - Extended test coverage
 - Simplified API i.e. removed sentiment analysis, lots of deps

v0.0.0
------
 - Inital version, very similar to TextAnalysis, [commit:8517fe2](https://github.com/JuliaText/TextAnalysis.jl/tree/8517fe2141317a209fe17e53b231038cc19c420b)
 - Not released
