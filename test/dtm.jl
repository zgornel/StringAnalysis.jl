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

    docs = [text(fd), text(sd)]
    crps = Corpus([fd, sd])
    for ngram_complexity in [1, 2]
        m = DocumentTermMatrix(crps, ngram_complexity=ngram_complexity)
        @test m isa DocumentTermMatrix
        @test dtm(m) isa SparseMatrixCSC

        update_lexicon!(crps, ngram_complexity)

        m = DocumentTermMatrix(crps, ngram_complexity=ngram_complexity)
        dtm(m)
        @test dtm(DocumentTermMatrix(docs; ngram_complexity=ngram_complexity)) == dtm(m)

        tf_idf(dtm(m))

        doc_idx = 1
        dtv(crps[doc_idx], lexicon(crps), ngram_complexity=ngram_complexity) ==
            dtv(crps, doc_idx, ngram_complexity=ngram_complexity)
        @test_throws ErrorException dtv(crps[1])  # test failure

        @test hash_dtv(text(crps[1]), TextHashFunction(), ngram_complexity=ngram_complexity) ==
            hash_dtv(crps[1], TextHashFunction(), ngram_complexity=ngram_complexity)
        v = hash_dtv(text(crps[1]), cardinality=25, ngram_complexity=ngram_complexity)
        @test v == hash_dtv(crps[1], cardinality=25, ngram_complexity=ngram_complexity)
        @test v isa SparseVector{StringAnalysis.DEFAULT_DTM_TYPE}
        @test length(v) == 25

    end

    dtm1 = dtm(crps)
    dtm1sp = sparse(dtm(crps))
    hash_dtm(crps)

    # Regex dtv
    doc = "a..b"
    lex = OrderedDict("aaa"=>1, "aaab"=>2, "accb"=>3, "bbb"=>4)
    v = dtv_regex(doc, lex, Float32)
    v2 = dtv_regex(NGramDocument(doc), lex, Float32)
    @test v == v2 == Float32[0, 1, 1, 0]

    for ngram_complexity in [1, 2]
        # construct a DocumentTermMatrix from a crps and a custom terms vector
        terms = ["And", "notincrps"]
        m = DocumentTermMatrix(crps, terms, ngram_complexity=ngram_complexity)
        @test size(dtm(m),2) == length(terms)
        @test terms == m.terms
        @test size(dtm(m),1) == length(crps)

        # construct a DocumentTermMatrix from a crps and a custom lexicon
        lex = OrderedDict("And"=>1, "notincrps"=>4)
        m = DocumentTermMatrix(crps, lex, ngram_complexity=ngram_complexity)
        @test size(dtm(m), 2) == length(keys(lex))
        @test size(dtm(m), 2) == length(m.terms)
        @test size(dtm(m), 1) == length(crps)

        # construct a DocumentTermMatrix from a dtm and terms vector
        terms = m.terms
        m2 = DocumentTermMatrix(dtm1, terms)
        @test m.row_indices == m2.row_indices
        m2 = DocumentTermMatrix(dtm1sp, terms)
        @test m.row_indices == m2.row_indices
    end
end

@testset "DTM: Iteration, Indexing" begin
    txt = "This is a Document."
    txt2 = "This is yet another document."
    txt3 = "Its full of documents !"
    txt4 = "This is yet another document dude"
    docs = map(StringDocument, [txt, txt2, txt3, txt4])
    crps = Corpus(docs)
    for ngram_complexity in [1, 2]
        m = DocumentTermMatrix(crps, ngram_complexity=ngram_complexity)
        # Iteration iterface tests
        T = Int8
        for (i,v) in enumerate(each_dtv(crps, eltype=T, ngram_complexity=ngram_complexity))
            @test v == m.dtm[1:end, i]
            i==1 && @test v isa SparseVector{T}
        end
        for (i,v) in enumerate(each_hash_dtv(crps, eltype=T, ngram_complexity=ngram_complexity))
            @test v == hash_dtv(crps[i], ngram_complexity=ngram_complexity)
            i==1 && @test v isa SparseVector{T}
        end
        # Indexing into the DTM
        word = "This"
        @test m[word] == m.dtm[m.row_indices[word], :]
        i = 1; j = 2; ii = 1:2
        @test m[i] == m.dtm[i]
        @test m[i, j] == m.dtm[i, j]
        @test m[ii] == m.dtm[ii]
    end
end
