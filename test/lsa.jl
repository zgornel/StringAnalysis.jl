@testset "LSA" begin
    doc1 = "a a a sample text text"
    doc2 = "another example example text text"
    crps = Corpus([StringDocument(doc1), StringDocument(doc2)])
    update_lexicon!(crps)
    m = DocumentTermMatrix(crps);
    @test lsa(crps) == lsa(m)
    s, v, d = lsa(crps)
    T = StringAnalysis.DEFAULT_FLOAT_TYPE
    @test s isa Matrix{T}
    @test size(s) == (2,2)
    @test v isa Vector{T}
    @test size(v) == (2,)
    @test d isa LinearAlgebra.Adjoint{T, Array{T,2}}
    @test size(d) == (5,2)

end

