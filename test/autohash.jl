@testset "Hash equality" begin
    text = "this is a text"
    path = "/a/b/c"
    @test hash(DocumentMetadata()) == hash(DocumentMetadata())
    @test hash(StringDocument(text)) == hash(StringDocument(text))
    @test hash(TokenDocument(text)) == hash(TokenDocument(text))
    @test hash(NGramDocument(text)) == hash(NGramDocument(text))
    @test hash(FileDocument(path)) == hash(FileDocument(path))
    @test hash(Corpus([StringDocument(text)])) ==
        hash(Corpus([StringDocument(text)]))
end
