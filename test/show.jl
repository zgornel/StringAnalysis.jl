@testset "Base.show methods" begin
    txt = "sample text"
    buf = IOBuffer()
    # Test Document types
    for dt in [StringDocument, NGramDocument, TokenDocument]
        doc = dt(txt)
        Test.@test try show(buf, doc); true
                   catch; false end
        Test.@test try print(buf, summary(doc)); true
                   catch; false end
    end
    # Metadata
    doc = StringDocument(txt)
    Test.@test try show(buf, doc.metadata); true
               catch; false end
    # Corpus
    crps = Corpus([doc])
    Test.@test try show(buf, crps); true
               catch; false end
    Test.@test try print(buf, summary(crps)); true
               catch; false end
    #Document Term Matrix
    update_lexicon!(crps)
    update_inverse_index!(crps)
    m = DocumentTermMatrix(crps)
    Test.@test try show(buf,m); true
               catch; false end
    # Stemmer
    s = Stemmer("english")
    Test.@test try show(buf,s); true
               catch; false end

end
