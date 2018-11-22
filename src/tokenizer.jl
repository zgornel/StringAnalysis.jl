# Split string into tokens on whitespace
tokenize(s::T) where T<:AbstractString = WordTokenizers.tokenize(s)
tokenize(lang::S, s::T) where {S <: Language, T <: AbstractString} = WordTokenizers.tokenize(s)

sentence_tokenize(s::T) where T<:AbstractString = WordTokenizers.split_sentences(s)
sentence_tokenize(lang::S, s::T) where {S <: Language, T<:AbstractString} = WordTokenizers.split_sentences(s)
