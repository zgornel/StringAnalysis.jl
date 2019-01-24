@testset "COOM (Co-occurence Matrix)" begin
    doc_raw = "This is a document. It has two sentences."
    doc = prepare(doc_raw, strip_punctuation|strip_whitespace|strip_case)
    sd = StringDocument(doc)
    td = TokenDocument(doc)
    nd = NGramDocument(doc)
    crps = Corpus([sd, td])
    T = Float16
    # Results for window = 5, all terms in document used
    expected_result_doc = [ # for window == 5
        0.0 2.0 1.0 2/3 0.5 0.4 0.0 0.0
        2.0 0.0 2.0 1.0 2/3 0.5 0.4 0.0
        1.0 2.0 0.0 2.0 1.0 2/3 0.5 0.4
        2/3 1.0 2.0 0.0 2.0 1.0 2/3 0.5
        0.5 2/3 1.0 2.0 0.0 2.0 1.0 2/3
        0.4 0.5 2/3 1.0 2.0 0.0 2.0 1.0
        0.0 0.4 0.5 2/3 1.0 2.0 0.0 2.0
        0.0 0.0 0.4 0.5 2/3 1.0 2.0 0.0]
    # Verify untyped constructor
    for d in [doc, sd, td, crps]
        C = CooMatrix(d)
        @test C isa CooMatrix{StringAnalysis.DEFAULT_FLOAT_TYPE}
        if !(d isa Corpus)
            @test coom(C) == expected_results_doc
        else
            #@test coom(C) == length(crps) * expected_result
        end
    end
    @test_throws ErrorException CooMatrix(nd)

    # Verify untyped constructor
    for d in [doc, sd, td, crps]
        C = CooMatrix{T}(d)
        @test C isa CooMatrix{T}
        if !(d isa Corpus)
            @test coom(C) == T.(expected_result)
        else
            #@test coom(C) == length(crps) * T.(expected_result)
        end
    end
    @test_throws ErrorException CooMatrix{T}(nd)

    # Results for window = 1, custom terms
    terms = ["this", "document", "it"]
    expected_result = [0.0 0.0 0.0; # document
                       0.0 0.0 2.0; # it
                       0.0 2.0 0.0] # this
    # Verify untyped constructor
    for d in [doc, sd, td, crps]
        C = CooMatrix{T}(d, terms, window=1)
        @test C isa CooMatrix{T}
        if !(d isa Corpus)
            @test coom(C) == T.(expected_result)
        else
            @test coom(C) == length(crps) * T.(expected_result)
        end
    end
    @test_throws ErrorException CooMatrix{T}(nd)
end
