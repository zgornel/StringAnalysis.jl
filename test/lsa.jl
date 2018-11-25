@testset "LSA" begin

    doc1 = "a a a sample text text"
    doc2 = "another example example text text"
    crps = Corpus([StringDocument(doc1), StringDocument(doc2)])
    update_lexicon!(crps)
    
    @test lsa(crps) == lsa(DocumentTermMatrix(crps))
    
    s, v, d = lsa(crps)
    T = typeof(1.0)
    @test s isa Matrix{T}
    @test size(s) == (2,2)
    @test v isa Vector{T}
    @test size(v) == (2,)
    @test d isa LinearAlgebra.Adjoint{T, Array{T,2}}
    @test size(d) == (5,2)

end

