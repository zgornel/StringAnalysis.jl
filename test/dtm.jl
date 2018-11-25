@testset "DTM: TextHashFunction" begin
    sample_text = "This is sample text"
    sample_doc = StringDocument(sample_text)
    thf = TextHashFunction()
    @test thf isa TextHashFunction
    @test thf.cardinality == StringAnalysis.DEFAULT_CARDINALITY
    @test thf.hash_function == hash
    @test length(hash_dtv(sample_doc, thf)) ==
            StringAnalysis.DEFAULT_CARDINALITY
    c = 15
    thf = TextHashFunction(c)
    @test c == cardinality(thf) == length(hash_dtv(sample_doc, thf))
end

@testset "DTM: Generics" begin
    sample_file = joinpath(dirname(@__FILE__), "data", "poem.txt")

    fd = FileDocument(sample_file)
    sd = StringDocument(text(fd))

    crps = Corpus([fd, sd])

    m = DocumentTermMatrix(crps)
    @test m isa DocumentTermMatrix
    @test dtm(m) isa SparseMatrixCSC
    @test dtm(m, :dense) == Matrix(dtm(m))

    update_lexicon!(crps)

    m = DocumentTermMatrix(crps)
    dtm(m)
    dtm(m, :dense)

    tf_idf(dtm(m, :dense))

    doc_idx = 1
    dtv(crps[doc_idx], lexicon(crps)) == dtv(crps, doc_idx)
    try
        dtv(crps[1])  # test failure
        @test false
    catch
        @test true
    end
    hash_dtv(crps[1], TextHashFunction())
    hash_dtv(crps[1])

    dtm1 = dtm(crps)
    dtm1sp = sparse(dtm(crps))
    hash_dtm(crps)

    @test tdm(crps) == tdm(m) == tdm(m, :sparse) == dtm(m)'
    @test Matrix(tdm(crps)) == Matrix(tdm(m)) ==
        tdm(m, :dense) == dtm(m, :dense)'

    hash_tdm(crps)

    # construct a DocumentTermMatrix from a crps and a custom terms vector
    terms = ["And", "notincrps"]
    m = DocumentTermMatrix(crps,terms)
    @test size(dtm(m),1) == length(terms)
    @test terms == m.terms
    @test size(dtm(m),2) == length(crps)

    # construct a DocumentTermMatrix from a crps and a custom lexicon
    lex = Dict("And"=>1, "notincrps"=>4)
    m = DocumentTermMatrix(crps,lex)
    @test size(dtm(m),1) == length(keys(lex))
    @test size(dtm(m),1) == length(m.terms)
    @test size(dtm(m),2) == length(crps)

    # construct a DocumentTermMatrix from a dtm and terms vector
    terms = m.terms
    m2 = DocumentTermMatrix(dtm1,terms)
    @test m.column_indices == m2.column_indices
    m2 = DocumentTermMatrix(dtm1sp,terms)
    @test m.column_indices == m2.column_indices
end

@testset "DTM: Iteration, Indexing" begin
    txt = "This is a Document."
    txt2 = "This is yet another document."
    txt3 = "Its full of documents !"
    txt4 = "This is yet another document dude"
    docs = map(StringDocument, [txt, txt2, txt3, txt4])
    crps = Corpus(docs)
    update_lexicon!(crps)
    update_inverse_index!(crps)
    m = DocumentTermMatrix(crps)
    # Iteration iterface tests
    i = 1
    for v in each_dtv(crps)
        @test v == m.dtm[i:i, 1:end]
        i+= 1
    end
    i = 1
    for v in each_hash_dtv(crps)
        @test v == hash_dtv(crps[i])
        i+= 1
    end
    # Indexing into the DTM
    word = "This"
    @test m[word] == m.dtm[:, m.column_indices[word]]
    i = 1; j = 2; ii = 1:2
    @test m[i] == m.dtm[i]
    @test m[i,j] == m.dtm[i,j]
    @test m[ii] == m.dtm[ii]

end
