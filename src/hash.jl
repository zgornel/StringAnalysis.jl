"""
    TextHashFunction(hash_function::Function, cardinality::Int)

The basic structure for performing text hashing: uses the `hash_function`
to generate feature vectors of length `cardinality`.

# Details
The hash trick is the use a hash function instead of a lexicon to determine the
columns of a DocumentTermMatrix-like encoding of the data. To produce a DTM for
a Corpus for which we do not have an existing lexicon, we need someway to map
the terms from each document into column indices. We use the now standard "Hash Trick"
in which we hash strings and then reduce the resulting integers modulo N, which
defines the numbers of columns we want our DTM to have. This amounts to
doing a non-linear dimensionality reduction with low probability that similar
terms hash to the same dimension.

To make things easier, we wrap Julia's hash functions in a new type,
TextHashFunction, which maintains information about the desired cardinality
of the hashes.

# References:
  * [The "Hash Trick" wiki page](https://en.wikipedia.org/wiki/Feature_hashing)
  * [Moody, John 1989](http://papers.nips.cc/paper/175-fast-learning-in-multi-resolution-hierarchies.pdf)

# Examples
```
julia> doc = StringDocument("this is a text")
       thf = TextHashFunction(hash, 13)
       hash_dtv(doc, thf, Float16)
13-element Array{Float16,1}:
 1.0
 1.0
 0.0
 0.0
 0.0
 0.0
 0.0
 2.0
 0.0
 0.0
 0.0
 0.0
 0.0
```
"""
mutable struct TextHashFunction
    hash_function::Function
    cardinality::Int
end

TextHashFunction(cardinality::Int) = TextHashFunction(DEFAULT_HASH_FUNCTION, cardinality)

TextHashFunction() = TextHashFunction(DEFAULT_HASH_FUNCTION, DEFAULT_CARDINALITY)

cardinality(h::TextHashFunction) = h.cardinality

hash_function(h::TextHashFunction) = h.hash_function

function index_hash(s::AbstractString, h::TextHashFunction)
    return Int(rem(h.hash_function(s), h.cardinality)) + 1
end
