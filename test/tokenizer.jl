# Test tokenize_default, tokenize_stringanalysis through tokenize
@testset "Tokenizer (to words)" begin
    sample_text = "this is some sample text"
    for method in [:default, :stringanalysis]
        tkns = StringAnalysis.tokenize(sample_text, method=method)
        @test isequal(
            tkns,
            String["this", "is", "some", "sample", "text"]
        )
    end
    for doc in [StringDocument(sample_text),
                NGramDocument(sample_text),
                split(sample_text)]
        tkns = StringAnalysis.tokenize(doc, method=:stringanalysis)
        @test isequal(
            tkns,
            String["this", "is", "some", "sample", "text"]
        )
    end
end

@testset "Tokenizer (to sentences)" begin
    sample_texts = ["Hi there, man. How are you?",
                    "Hi there, man. how are you?"]
    vs =[
         ["Hi there, man.", "How are you?"],
         ["Hi there, man. how are you?"]
        ]

    for (sample_text, v) in zip(sample_texts, vs)
        s1 = sentence_tokenize(sample_text)
        s2 = sentence_tokenize(StringAnalysis.DEFAULT_LANGUAGE, sample_text)
        @test length(s1) == length(s2) == length(v)
        @test s1 == s2
        @test String.(s1) == String.(s2) == v
    end
end
