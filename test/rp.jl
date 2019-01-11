@testset "Random Projections" begin
    # Documents
    doc1 = StringDocument("This is a text about an apple. There are many texts about apples.")
    doc2 = StringDocument("Pears and apples are good but not exotic. An apple a day keeps the doctor away.")
    doc3 = StringDocument("Fruits are good for you.")
    doc4 = StringDocument("This phrase has nothing to do with the others...")
    doc5 = StringDocument("Simple text, little info inside")
    # Corpus
    crps = Corpus(AbstractDocument[doc1, doc2, doc3, doc4, doc5])
    prepare!(crps, strip_punctuation)
    update_lexicon!(crps)
    update_inverse_index!(crps)
    lex = sort(collect(keys(crps.lexicon)))
    m = length(lexicon(crps))
    n = length(crps)
    # Retrieval
    query = StringDocument("Apples and an exotic fruit.")
    for k in [1, 3]
        for stats in [:tf, :tfidf, :bm25]
            for T in [Float16, Float32, Float64]
                dtm = DocumentTermMatrix{T}(crps, lex)
                model = rp(dtm, k=k, stats=stats)
                @test model isa RPModel{String, T, SparseMatrixCSC{T,Int}, Int}
                @test size(model.R, 2) == m
                @test size(model.R, 1) == k
                idxs, corrs = cosine(model, dtm, query)
                @test length(idxs) == length(corrs) == length(crps)
                sim = similarity(model, crps[rand(1:n)], query)
                @test -1.0 <= sim <= 1.0
            end
        end
    end
    # Tests for the rest of the functions
    K = 2
    T = Float32
    # Vocabulary
    model = rp(crps, T, k=K)
    @test model isa RPModel{String, T, SparseMatrixCSC{T, Int}, Int}
    @test all(in_vocabulary(model, word) for word in keys(crps.lexicon))
    @test vocabulary(model) == sort(collect(keys(crps.lexicon)))
    @test size(model) == (length(crps.lexicon), K)
    # Document, corpus embedding
    dtm = DocumentTermMatrix{T}(crps, lex)
    U = embed_document(model, crps)
    @test eltype(U) == T
    @test eltype(embed_document(model, crps[1])) == T
    for i in 1:n
        @test all(U[i,:] .≈ embed_document(model, crps[i]))
    end
    # Index, Similarity
    idx = 2
    word = model.vocab[idx]
    @test index(model, word) == model.vocab_hash[word]
    @test get_vector(model, word) == model.R[:, idx]
    @test similarity(model, crps[1], crps[2]) isa T
    @test similarity(model, crps[1], crps[2]) == similarity(model, crps[2], crps[1])
    @test_throws ErrorException RPModel(DocumentTermMatrix{Int}(crps), k=K)
    # Test saving and loading an random projection model
    T = Float32
    vocab = split("a random string")
    vocab_hash = Dict("a"=>1, "random"=>2, "string"=>3)
    k=2
    stats = :tf
    κ = 2
    β = 0.71
    model = rp(crps, T, k=k, stats=stats, κ=κ, β=β)
    file = "./_rp_model.txt"
    save_rp_model(model, file)
    # Model 1
    loaded_model_1 = load_rp_model(file, Float64, sparse=true)
    @test loaded_model_1 isa RPModel{String, Float64, SparseMatrixCSC{Float64, Int}, Int}
    @test all(loaded_model_1.R .≈ model.R)
    @test loaded_model_1.vocab == model.vocab
    @test loaded_model_1.vocab_hash == model.vocab_hash
    @test loaded_model_1.stats == model.stats
    @test all(loaded_model_1.idf .≈ model.idf)
    @test loaded_model_1.nwords ≈ model.nwords
    @test loaded_model_1.κ == model.κ
    @test loaded_model_1.β == model.β
    # Model 2
    loaded_model_2 = load_rp_model(file, Float32, sparse=false)
    @test loaded_model_2 isa RPModel{String, Float32, Matrix{Float32}, Int}
    @test all(loaded_model_1.R .≈ model.R)
    @test loaded_model_2.vocab == model.vocab
    @test loaded_model_2.vocab_hash == model.vocab_hash
    @test loaded_model_2.stats == model.stats
    @test all(loaded_model_2.idf .≈ model.idf)
    @test loaded_model_2.nwords ≈ model.nwords
    @test loaded_model_2.κ == model.κ
    @test loaded_model_2.β == model.β
    rm(file)
end
