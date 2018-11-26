@testset "Corpus" begin

    sample_text1 = "This is a string"
    sample_text2 = "This is also a string"
    sample_file = joinpath(dirname(@__FILE__), "data", "poem.txt")

    sd = StringDocument(sample_text1)
    fd = FileDocument(sample_file)
    td = TokenDocument(sample_text1)
    ngd = NGramDocument(sample_text1)

    crps = Corpus([sd, fd, td, ngd])
    @test typeof(crps) <: Corpus{<:GenericDocument}

    crps2 = Corpus([ngd, ngd])
    update_inverse_index!(crps2)
    @test typeof(crps2) <: Corpus{<:NGramDocument}

    documents(crps)

    for doc in crps
    	@test isa(doc, AbstractDocument)
    end

    @test isempty(lexicon(crps))
    update_lexicon!(crps)
    @test !isempty(lexicon(crps))

    @test isempty(inverse_index(crps))
    update_inverse_index!(crps)
    @test !isempty(inverse_index(crps))

    @test hash_function(hash_function(crps)) === hash
    hash_function!(crps, TextHashFunction(hash, 10))
    @test hash_function(hash_function(crps)) === hash
    @test cardinality(hash_function(crps)) == 10

    # Indexing
    @test crps[1] isa StringDocument
    @test text(crps[1]) == "This is a string"
    @test crps["string"] == [1, 3, 4]
    @test crps2["string"] == [1, 2]

    # Cotainer treatment
    doc = StringDocument("~pushed~")
    n = length(crps)
    push!(crps, doc);
    @test length(crps) == n+1
    pdoc = pop!(crps)
    @test length(crps) == n
    @test text(pdoc) == text(doc)
    pushfirst!(crps, doc)
    @test length(crps) == n+1
    @test text(crps[1]) == text(doc)
    pdoc = popfirst!(crps)
    @test length(crps) == n
    @test text(pdoc) == text(doc)
    pos = 3
    insert!(crps, pos, pdoc)
    @test length(crps) == n+1
    @test text(crps[3]) == text(pdoc)
    deleteat!(crps, 3)
    @test length(crps) == n
end
