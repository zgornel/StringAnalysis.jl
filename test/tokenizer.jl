# Test tokenize_fast, tokenize_slow through tokenize
@testset "Tokenizer" begin
    sample_text = "this is some sample text"
    for method in [:slow, :fast]
        tkns = StringAnalysis.tokenize(sample_text, method=method)
        @test isequal(
            tkns,
            String["this", "is", "some", "sample", "text"]
        )
    end
    for doc in [StringDocument(sample_text),
                NGramDocument(sample_text),
                split(sample_text)]
        tkns = StringAnalysis.tokenize(sample_text, method=:fast)
        @test isequal(
            tkns,
            String["this", "is", "some", "sample", "text"]
        )
    end
end
