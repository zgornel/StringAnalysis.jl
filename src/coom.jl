# This file originally copied from StringAnalysis.jl
# Copyright (c) 2018: Corneliu Cofaru.

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""
    coo_matrix(::Type{T}, doc::Vector{AbstractString}, vocab::OrderedDict{AbstractString, Int}, window::Int, direction::Bool, normalize::Bool)

Basic low-level function that calculates the co-occurence matrix of a document.
Returns a sparse co-occurence matrix sized `n × n` where `n = length(vocab)`
with elements of type `T`. The document `doc` is represented by a vector of its
terms (in order)`. The `window` argument indicates the size of the
sliding word window in which co-occurrences are counted, while the `direction` argument determines whether the co-occurrence matrix is built based on both sides of each word or only on one side. Finally, the argument `normalize` indicates whether the counts by the distance between word positions are normalized or not.

# Example
```
julia> using StringAnalysis, OrderedCollections
       doc = StringDocument("This is a text about an apple. There are many texts about apples.")
       docv = tokenize(text(doc))
       vocab = OrderedDict("This"=>1, "is"=>2, "apple."=>3)
       StringAnalysis.coo_matrix(Float16, docv, vocab, 5, false, true)

3×3 SparseArrays.SparseMatrixCSC{Float16,Int64} with 4 stored entries:
  [2, 1]  =  2.0
  [1, 2]  =  2.0
  [3, 2]  =  0.3999
  [2, 3]  =  0.3999
```
"""
function coo_matrix(::Type{T},
    doc::Vector{<:AbstractString},
    vocab::OrderedDict{<:AbstractString,Int},
    window::Int,
    direction::Bool,
    normalize::Bool) where {T<:AbstractFloat}
    direction || return coo_matrix(T, doc, vocab, window, normalize)
    n = length(vocab)
    m = length(doc)
    coom = spzeros(T, n, n)
    # Count co-occurrences
    for (i, token) in enumerate(doc)
        # looking forward
        @inbounds for j in i:min(m, i + window)
            # @inbounds for j in max(1, i-window):min(m, i+window)
            wtoken = doc[j]
            nm = T(ifelse(normalize, abs(i - j), 1))
            row = get(vocab, token, nothing)
            col = get(vocab, wtoken, nothing)
            if i != j && row != nothing && col != nothing
                coom[row, col] += one(T) / nm
                # avoiding to create a symmetric matrix and keep the forward looking coocurrence from above.
                # coom[col, row] = coom[row, col]
            end
        end
    end
    return coom
end

function coo_matrix(::Type{T},
    doc::Vector{<:AbstractString},
    vocab::OrderedDict{<:AbstractString,Int},
    window::Int,
    normalize::Bool=true) where {T<:AbstractFloat}
    n = length(vocab)
    m = length(doc)
    coom = spzeros(T, n, n)
    # Count co-occurrences
    for (i, token) in enumerate(doc)
        @inbounds for j in max(1, i - window):min(m, i + window)
            wtoken = doc[j]
            nm = T(ifelse(normalize, abs(i - j), 1))
            row = get(vocab, token, nothing)
            col = get(vocab, wtoken, nothing)
            if i != j && row != nothing && col != nothing
                coom[row, col] += one(T) / nm
                coom[col, row] = coom[row, col]
            end
        end
    end
    return coom
end

coo_matrix(::Type{T}, doc::Vector{<:AbstractString}, vocab::Dict{<:AbstractString, Int},
                    window::Int, direction::Bool, normalize::Bool=true) where T<:AbstractFloat =
            coo_matrix(T, doc, OrderedDict(vocab), window, direction, normalize)

"""
Basic Co-occurrence Matrix (COOM) type.
# Fields
  * `coom::SparseMatriCSC{T,Int}` the actual COOM; elements represent
co-occurrences of two terms within a given window
  * `terms::Vector{String}` a list of terms that represent the lexicon of
the document or corpus
  * `column_indices::OrderedDict{String, Int}` a map between the `terms` and the
columns of the co-occurrence matrix
"""
struct CooMatrix{T}
    coom::SparseMatrixCSC{T, Int}
    terms::Vector{String}
    column_indices::OrderedDict{String, Int}
end


"""
    CooMatrix{T}(crps::Corpus [,terms] [;window=5, direction=false, normalize=true])

Auxiliary constructor(s) of the `CooMatrix` type. The type `T` has to be
a subtype of `AbstractFloat`. The constructor(s) requires a corpus `crps` and
a `terms` structure representing the lexicon of the corpus. The latter
can be a `Vector{String}`, an `AbstractDict` where the keys are the lexicon,
or can be omitted, in which case the `lexicon` field of the corpus is used.
The keyword argument `window` defines the window in terms of number of words based on which the co-occurence matrix is built. By default, if is is ommitted, the window size is set to 5.
The keyword argument `direction` determines whether the co-occurrence matrix is built based on both sides of each word or only on one side, looking forward.
The keyword argument `window` indicates the size of the
sliding word window in terms of number of words based on which the co-occurrences are counted, while the keyword `direction` determines whether the co-occurrence matrix is built based on both sides of each word or only on one side, looking forward (with a default value `false`). Finally, the keyword argument `normalize` indicates whether the counts by the distance between word positions are normalized or not (with a default value is `true`).

# Example
```
julia> using StringAnalysis, OrderedCollections
       doc1 = StringDocument("This is a text about an apple. There are many texts about apples.")
       doc2 = StringDocument("Here is another text about an appleorange. There are not many texts about oranges.")
       crps = Corpus([doc1, doc2])
       show(coom(CooMatrix(crps)))

sparse([3, 7, 8, 9, 10, 12, 14, 16, 18, 4, 5, 6, 8, 11, 13, 1, 6, 7, 8, 9, 10, 12, 2, 5, 6, 7, 8, 9, 11, 13, 14, 15, 17, 2, 4, 8, 11, 13, 15, 17, 2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 16, 1, 3, 4, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 1, 3, 4, 6, 7, 8, 11, 12, 13, 1, 3, 7, 8, 12, 2, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 17, 1, 3, 6, 7, 8, 9, 10, 11, 14, 16, 18, 2, 4, 5, 6, 7, 8, 9, 11, 14, 15, 17, 1, 4, 6, 7, 8, 11, 12, 13, 15, 16, 4, 5, 6, 7, 8, 11, 13, 14, 17, 1, 6, 7, 8, 12, 14, 18, 4, 5, 8, 11, 13, 15, 1, 7, 8, 12, 16], [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 16, 16, 17, 17, 17, 17, 17, 17, 18, 18, 18, 18, 18], [2.0, 1.0, 1.3333333333333333, 0.4, 2.0, 2.0, 0.4, 2.0, 2.0, 0.6666666666666666, 2.0, 0.4, 2.0, 0.5, 1.0, 2.0, 0.4, 0.6666666666666666, 1.0, 0.5, 1.0, 2.0, 0.6666666666666666, 1.0, 1.6666666666666665, 0.9, 2.4, 0.6666666666666666, 3.0, 4.0, 0.5, 2.0, 0.6666666666666666, 2.0, 1.0, 2.0, 0.4, 1.3333333333333333, 0.4, 2.0, 0.4, 0.4, 1.6666666666666665, 2.0, 2.2333333333333334, 2.0, 4.0, 1.0, 1.1666666666666665, 2.0, 1.0, 0.4, 1.0, 0.6666666666666666, 0.9, 2.0, 4.0, 2.0, 0.4, 1.3333333333333333, 2.0, 0.4, 2.0, 0.5, 0.6666666666666666, 0.4, 1.3333333333333333, 2.0, 1.0, 2.4, 2.0, 2.2333333333333334, 4.0, 1.4, 0.5, 2.1666666666666665, 4.0, 4.0, 1.0, 1.0666666666666667, 1.0, 2.0, 0.5, 0.4, 0.5, 0.6666666666666666, 2.0, 2.0, 1.4, 1.0, 0.6666666666666666, 0.5, 2.0, 1.0, 0.4, 0.5, 0.6666666666666666, 0.5, 3.0, 0.4, 4.0, 1.3333333333333333, 2.1666666666666665, 1.0, 0.8, 1.6666666666666665, 1.0, 2.0, 0.4, 2.0, 2.0, 1.0, 2.0, 4.0, 0.6666666666666666, 0.6666666666666666, 0.8, 0.6666666666666666, 2.0, 0.6666666666666666, 1.0, 4.0, 1.3333333333333333, 1.1666666666666665, 0.4, 4.0, 0.5, 1.6666666666666665, 0.4, 1.0, 1.0, 0.4, 0.5, 2.0, 2.0, 1.0, 1.0, 0.6666666666666666, 0.4, 0.6666666666666666, 0.5, 2.0, 0.4, 1.0, 0.5, 1.0666666666666667, 2.0, 1.0, 0.6666666666666666, 0.5, 2.0, 0.4, 0.6666666666666666, 1.0, 2.0, 0.5, 1.0, 0.6666666666666666, 2.0, 2.0, 0.4, 1.0, 0.5, 2.0, 0.4, 0.5, 0.6666666666666666, 1.0], 18, 18)
```
"""
function CooMatrix{T}(crps::Corpus,
                      terms::Vector{String};
                      window::Int=5,
                      direction::Bool=false,
                      normalize::Bool=true) where T<:AbstractFloat
    column_indices = OrderedDict(columnindices(terms))
    n = length(terms)
    coom = spzeros(T, n, n)
    for doc in crps
        coom .+= coo_matrix(T, tokens(doc), column_indices, window,direction, normalize)
    end
    return CooMatrix{T}(coom, terms, column_indices)
end

CooMatrix(crps::Corpus, terms::Vector{String}; window::Int=5, direction::Bool=false, normalize::Bool=true) =
    CooMatrix{Float64}(crps, terms, window=window, direction=direction, normalize=normalize)

CooMatrix{T}(crps::Corpus, lex::AbstractDict; window::Int=5, direction::Bool=false, normalize::Bool=true
            ) where T<:AbstractFloat =
    CooMatrix{T}(crps, collect(keys(lex)), window=window, direction=direction, normalize=normalize)

CooMatrix(crps::Corpus, lex::AbstractDict; window::Int=5, direction::Bool=false, normalize::Bool=true) =
    CooMatrix{Float64}(crps, lex, window=window, direction=direction, normalize=normalize)

CooMatrix{T}(crps::Corpus; window::Int=5, direction::Bool=false, normalize::Bool=true) where T<:AbstractFloat = begin
    isempty(lexicon(crps)) && update_lexicon!(crps)
    CooMatrix{T}(crps, lexicon(crps), window=window, direction=direction, normalize=normalize)
end

CooMatrix(crps::Corpus; window::Int=5, direction::Bool=false, normalize::Bool=true) = begin
    isempty(lexicon(crps)) && update_lexicon!(crps)
    CooMatrix{Float64}(crps, lexicon(crps), window=window, direction=direction, normalize=normalize)
end

# Document methods
"""
    CooMatrix{T}(doc [,terms] [;window=5, direction=false, normalize=true])

Auxiliary constructor(s) of the `CooMatrix` type. The type `T` has to be
a subtype of `AbstractFloat`. The constructor(s) requires a corpus `crps` and
a `terms` structure representing the lexicon of the corpus. The latter
can be a `Vector{String}`, an `AbstractDict` where the keys are the lexicon,
or can be omitted, in which case the `lexicon` field of the corpus is used.
The keyword argument `window` defines the window in terms of number of words based on which the co-occurence matrix is built. By default, if is is ommitted, the window size is set to 5.
The keyword argument `direction` determines whether the co-occurrence matrix is built based on both sides of each word or only on one side, looking forward.
The keyword argument `window` indicates the size of the
sliding word window in terms of number of words based on which the co-occurrences are counted, while the keyword `direction` determines whether the co-occurrence matrix is built based on both sides of each word or only on one side, looking forward (with a default value `false`). Finally, the keyword argument `normalize` indicates whether the counts by the distance between word positions are normalized or not (with a default value is `true`).

# Example
```
julia> using StringAnalysis, OrderedCollections
       doc = StringDocument("This is a text about an apple. There are many texts about apples.")
julia> show(coom(CooMatrix(doc)))
       
sparse([2, 3, 4, 5, 6, 1, 3, 4, 5, 6, 7, 1, 2, 4, 5, 6, 7, 8, 1, 2, 3, 5, 6, 7, 8, 9, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 2, 3, 4, 5, 6, 8, 9, 10, 11, 3, 4, 5, 6, 7, 9, 10, 11, 12, 4, 5, 6, 7, 8, 10, 11, 12, 13, 5, 6, 7, 8, 9, 11, 12, 13, 5, 6, 7, 8, 9, 10, 12, 13, 5, 8, 9, 10, 11, 13, 5, 9, 10, 11, 12], [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13], [2.0, 1.0, 0.6666666666666666, 0.5, 0.4, 2.0, 2.0, 1.0, 0.6666666666666666, 0.5, 0.4, 1.0, 2.0, 2.0, 1.0, 0.6666666666666666, 0.5, 0.4, 0.6666666666666666, 1.0, 2.0, 2.0, 1.0, 0.6666666666666666, 0.5, 0.4, 0.5, 0.6666666666666666, 1.0, 2.0, 2.0, 1.4, 1.1666666666666665, 1.1666666666666665, 1.4, 2.0, 2.0, 1.0, 0.4, 0.5, 0.6666666666666666, 1.0, 2.0, 2.0, 1.0, 0.6666666666666666, 0.5, 0.4, 0.4, 0.5, 0.6666666666666666, 1.4, 2.0, 2.0, 1.0, 0.6666666666666666, 0.5, 0.4, 0.5, 1.1666666666666665, 1.0, 2.0, 2.0, 1.0, 0.6666666666666666, 0.4, 0.4, 1.1666666666666665, 0.6666666666666666, 1.0, 2.0, 2.0, 1.0, 0.5, 0.4, 1.4, 0.5, 0.6666666666666666, 1.0, 2.0, 2.0, 0.6666666666666666, 0.5, 2.0, 0.4, 0.5, 0.6666666666666666, 1.0, 2.0, 1.0, 0.6666666666666666, 2.0, 0.4, 0.5, 0.6666666666666666, 1.0, 2.0, 1.0, 0.4, 0.5, 0.6666666666666666, 2.0], 13, 13)
```
"""
function CooMatrix{T}(doc,
                      terms::Vector{String};
                      window::Int=5,
                      direction::Bool=false,
                      normalize::Bool=true) where T<:AbstractFloat
    # Initializations
    column_indices = OrderedDict(columnindices(terms))
    coom = coo_matrix(T, tokens(doc), column_indices, window, direction, normalize)
    return CooMatrix{T}(coom, terms, column_indices)
end

function CooMatrix{T}(doc::NGramDocument,
                      terms::Vector{String};
                      window::Int=5,
                      direction::Bool=false,
                      normalize::Bool=true) where T <: AbstractFloat
    error("The Co occurrence matrix of an NGramDocument can't be created.")
end

CooMatrix(doc, terms::Vector{String}; window::Int=5, direction::Bool=false, normalize::Bool=true) =
    CooMatrix{Float64}(doc, terms, window=window, direction=direction, normalize=normalize)

function CooMatrix{T}(doc; window::Int=5, direction::Bool=false, normalize::Bool=true) where T<:AbstractFloat
    terms = unique(String.(tokens(doc)))
    CooMatrix{T}(doc, terms, window=window, direction=direction, normalize=normalize)
end

CooMatrix(doc; window::Int=5, direction::Bool=false, normalize::Bool=true) where T<:AbstractFloat =
    CooMatrix{Float64}(doc, window=window, direction=direction, normalize=normalize)

"""
    coom(c::CooMatrix)

Access the co-occurrence matrix field `coom` of a `CooMatrix` `c`.
"""
coom(c::CooMatrix) = c.coom

"""
    browsecoompairs(coo::CooMatrix, term1::AbstractString, term2::AbstractString)

Browse through frequencies of word pairs of `AbstractString` type, `term1` and `term2`  using a `coo` that is a co-ocurrence matrix of type `CooMatrix`. 

# Example
```
julia> doc = StringDocument("This is a text about an apple. There are many texts about apples.");
julia> browsecoompairs(CooMatrix(doc), "a", "is")
2.0
```
"""
function browsecoompairs(coo::CooMatrix, term1::AbstractString, term2::AbstractString)
    coo.coom[coo.column_indices[term1], coo.column_indices[term2]]
end

"""
    coom(entity, eltype=DEFAULT_FLOAT_TYPE [;window=5, direction=false, normalize=true])

Access the co-occurrence matrix of the `CooMatrix` associated
with the `entity`. The `CooMatrix{T}` will first have to
be created in order for the actual matrix to be accessed.
"""
coom(entity, eltype::Type{T}=Float;
        window::Int=5, direction::Bool=false, normalize::Bool=true) where T<:AbstractFloat =
    coom(CooMatrix{T}(entity, window=window, direction=direction, normalize=normalize))
