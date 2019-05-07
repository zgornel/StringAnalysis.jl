
@testset "Stemmer" begin
    # Stemmer
    algs = stemmer_types()
    @test !isempty(algs)

    for alg in algs
        stmr = Stemmer(alg)
        StringAnalysis.release(stmr)
    end

    test_cases = Dict(
        "english" => Dict{String, String}(
            "working" => "work",
            "worker" => "worker",
            "aβc" => "aβc",
            "a∀c" => "a∀c"
        )
    )

    for (alg, test_words) in test_cases
        stmr = Stemmer(alg)
        for (n,v) in test_words
            @test v == stem(stmr, n)
        end
    end

    # stem_all
    test_cases = Dict(
        "english" =>Dict{String, String}(
            "this is a sentences" => "this is a sentenc"
        )
    )
    lang = StringAnalysis.DEFAULT_LANGUAGE
    for method in ["fast", "slow"]
        for (alg, test_words) in test_cases
            stmr = Stemmer(alg)
            for (n,v) in test_words
                @test v == StringAnalysis.stem_all(stmr, lang, n)
            end
        end
    end

    # stem/stem! vectors of strings
    test_cases = Dict(
        "english" =>Dict(
            ["this", "is", "a", "sentences"] => ["this", "is", "a", "sentenc"]
        )
    )
    for (alg, test_words) in test_cases
        stmr = Stemmer(alg)
        for (n,v) in test_words
            @test v == stem(n)
            @test v == stem(stmr, n)
            n2 = Base.deepcopy(n); stem!(n2)
            @test v == n2
            n2 = Base.deepcopy(n); stem!(stmr, n2)
            @test v == n2
        end
    end

    # Document types and corpus
    txt = """This is a sample text for testing
             the stemmer. No actual stemmed
             results are being checked."""
    for doc in [StringDocument(txt),
                NGramDocument(txt),
                TokenDocument(txt),
                Corpus([StringDocument(txt)])]
        @test try stem!(doc); true
              catch; false end
    end

    fdoc = FileDocument("/some/file")
    @test try stem!(fdoc); false
              catch; true end

    # Test 3-gram document
    doc = NGramDocument("parts of language", DocumentMetadata(), 3)
    stem!(doc)
    @test begin
        "of" in keys(doc.ngrams)
        "languag" in keys(doc.ngrams)
        "part of languag" in keys(doc.ngrams)
        "part of" in keys(doc.ngrams)
        "part" in keys(doc.ngrams)
        "of languag" in keys(doc.ngrams)
    end
end
