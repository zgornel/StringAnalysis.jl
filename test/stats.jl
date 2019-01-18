@testset "Stats (TF,TF-IDF,BM25)" begin

    doc1 = "a a a sample text text"
    doc2 = "another example example text text"
    doc3 = ""
    doc4 = "another another text text text text"

    crps = Corpus(map(StringDocument, [doc1, doc2, doc3, doc4]))

    update_lexicon!(crps)
    m = DocumentTermMatrix(crps)

    T = StringAnalysis.DEFAULT_FLOAT_TYPE

    approx_eq(m1::AbstractMatrix{T}, m2::AbstractMatrix{T}; tol=1e-6
             ) where T = begin
        t = all(abs.(m1-m2) .<= tol)
        return t
    end

    max_tol = 1e-5
    # TF
    # Terms are in alphabetical ordering
    # (correctweights uses older rows are documents format, hence the transpose)
    correctweights = sqrt.(T.(
                        [0.5  0.0  0.0  1/6  1/3
                         0.0  0.2  0.4  0.0  0.4
                         0.0  0.0  0.0  0.0  0.0
                         0.0  1/3  0.0  0.0  2/3]'))

    myweights = tf(m)
    @test approx_eq(myweights, correctweights, tol=max_tol)

    myweights = tf(dtm(m))
    @test approx_eq(myweights, correctweights, tol=max_tol)
    @test typeof(myweights) <: SparseMatrixCSC

    myweights = tf(Matrix(dtm(m)))
    @test isnan(sum(myweights)) == 0
    @test approx_eq(myweights, correctweights, tol=max_tol)
    @test typeof(myweights) <: Matrix

    myweights = float(dtm(crps));
    tf!(myweights)
    @test approx_eq(T.(myweights), correctweights, tol=max_tol)
    @test typeof(myweights) <: SparseMatrixCSC
    @test eltype(myweights) == typeof(1.0)

    myweights = float(Matrix(dtm(crps)));
    tf!(myweights)
    @test approx_eq(T.(myweights), correctweights, tol=max_tol)
    @test typeof(myweights) <: Matrix
    @test eltype(myweights) == typeof(1.0)

    @test_throws MethodError tf!(DocumentTermMatrix(crps))
    @test tf!(DocumentTermMatrix{T}(crps)) isa SparseMatrixCSC{T, Int}

    # TF-IDF
    # Terms are in alphabetical ordering
    # (correctweights uses older rows are documents format, hence the transpose)
    correctweights = T.([1.19724  0.0       0.0      0.691224  0.57735
                         0.0      0.575869  1.07084  0.0       0.632456
                         0.0      0.0       0.0      0.0       0.0
                         0.0      0.743444  0.0      0.0       0.816497]')

    myweights = tf_idf(m)
    @test approx_eq(myweights, correctweights, tol=max_tol)

    myweights = tf_idf(dtm(m))
    @test approx_eq(myweights, correctweights, tol=max_tol)
    @test typeof(myweights) <: SparseMatrixCSC

    myweights = tf_idf(Matrix(dtm(m)))
    @test isnan(sum(myweights)) == 0
    @test approx_eq(myweights, correctweights, tol=max_tol)
    @test typeof(myweights) <: Matrix

    myweights = float(dtm(crps));
    tf_idf!(myweights)
    @test approx_eq(T.(myweights), correctweights, tol=max_tol)
    @test typeof(myweights) <: SparseMatrixCSC
    @test eltype(myweights) == typeof(1.0)

    myweights = float(Matrix(dtm(crps)));
    tf_idf!(myweights)
    @test approx_eq(T.(myweights), correctweights, tol=max_tol)
    @test typeof(myweights) <: Matrix
    @test eltype(myweights) == typeof(1.0)

    @test_throws MethodError tf_idf!(DocumentTermMatrix(crps))
    @test tf_idf!(DocumentTermMatrix{T}(crps)) isa SparseMatrixCSC{T, Int}

    # Terms are in alphabetical ordering
    # (correctweights uses older rows are documents format, hence the transpose)
    correctweights = T.([1.08029  0.0       0.0      0.685309  0.542113
                         0.0      0.637042  1.10885  0.0       0.654905
                         0.0      0.0       0.0      0.0       0.0
                         0.0      0.69807   0.0      0.0       0.713275]')
    myweights = bm_25(m)
    @test approx_eq(myweights, correctweights, tol=max_tol)

    myweights = bm_25(dtm(m))
    @test approx_eq(myweights, correctweights, tol=max_tol)
    @test typeof(myweights) <: SparseMatrixCSC

    myweights = bm_25(Matrix(dtm(m)))
    @test isnan(sum(myweights)) == 0
    @test approx_eq(myweights, correctweights, tol=max_tol)
    @test typeof(myweights) <: Matrix

    myweights = float(dtm(crps));
    bm_25!(myweights)
    @test approx_eq(T.(myweights), correctweights, tol=max_tol)
    @test typeof(myweights) <: SparseMatrixCSC
    @test eltype(myweights) == typeof(1.0)

    myweights = float(Matrix(dtm(crps)));
    bm_25!(myweights)
    @test approx_eq(T.(myweights), correctweights, tol=max_tol)
    @test typeof(myweights) <: Matrix
    @test eltype(myweights) == typeof(1.0)

    @test_throws MethodError bm_25!(DocumentTermMatrix(crps))
    @test bm_25!(DocumentTermMatrix{T}(crps)) isa SparseMatrixCSC{T, Int}
end
