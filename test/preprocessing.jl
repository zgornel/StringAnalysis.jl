@testset "Preprocessing (newer)" begin
	poem_no_1 = """
                \t ...
                \t this is it      !            \n
                \t 1,2,3,y,2,z                  \n
                \t  GET some h/a\\y             \n
                \t         v`v`v`v`v            \n
                \t  FIND a >>>pÎn<<<            \n
                \t         ^^^^^^^^^            \n
                \t  p-a-t-t-e-r-n-s'n f.i.n     \n
                \t  get a <tag, hold it still>  \n
                \t          ~ 0_0' ~            \n
                \t        </tag a thrill>       \n
                \t          b.y. ~z.g.o.r.n.e.l \n
                                               """

    sdoc = StringDocument(poem_no_1); prepare!(sdoc, strip_everything)
    ndoc = NGramDocument(poem_no_1); prepare!(ndoc, strip_everything)
    tdoc = TokenDocument(poem_no_1); prepare!(tdoc, strip_everything);
    crps = Corpus([sdoc])
    @test prepare(poem_no_1, strip_everything) == "pin"
    @test text(sdoc) == "pin"
    @test ngrams(ndoc) == Dict("tag"=>2,"pin"=>1,"hold"=>1,"thrill"=>1)
    @test string.(tokens(tdoc)) == ["pin", "tag", "hold", "tag", "thrill"]
    @test text(crps[1]) == "pin"
end


# Text Analysis older tests
@testset "Preprocessing (older)" begin
    sample_text1 = "This is, 1 MESSED υπ string!..."
    sample_text1_accents = "Thîs îs meşsed string"
    sample_text1_wo_punctuation = "This is 1 MESSED υπ string"
    sample_text1_wo_punctuation_numbers = "This is  MESSED υπ string"
    sample_text1_wo_punctuation_numbers_case = "this is  messed υπ string"
    sample_text1_wo_punctuation_numbers_case_az = "this is  messed  string"

    sample_texts = [
        sample_text1,
        sample_text1_wo_punctuation,
        sample_text1_wo_punctuation_numbers,
        sample_text1_wo_punctuation_numbers_case,
        sample_text1_wo_punctuation_numbers_case_az
    ]

    # This idiom is _really_ ugly since "OR" means "AND" here.
    for str in sample_texts
        sd = StringDocument(str)
        prepare!(
            sd,
            strip_punctuation | strip_numbers |
            strip_case | strip_whitespace | strip_non_ascii |
            strip_accents
        )
        @test isequal(strip(sd.text), "this is messed string")
    end

    # Need to only remove words at word boundaries
    doc = Document("this is sample text")
    StringAnalysis.remove_words!(doc, ["sample"])
    @test isequal(doc.text, "this is   text")

    doc = Document("this is sample text")
    prepare!(doc, strip_articles)
    @test isequal(doc.text, "this is sample text")

    doc = Document("this is sample text")
    prepare!(doc, strip_definite_articles)
    @test isequal(doc.text, "this is sample text")

    doc = Document("this is sample text")
    prepare!(doc, strip_indefinite_articles)
    @test isequal(doc.text, "this is sample text")

    doc = Document("this is sample text")
    prepare!(doc, strip_prepositions)
    @test isequal(doc.text, "this is sample text")

    doc = Document("this is sample text")
    prepare!(doc, strip_pronouns)
    @test isequal(doc.text, "this is sample text")

    doc = Document("this is sample text")
    prepare!(doc, strip_stopwords)
    @test isequal(strip(doc.text), "sample text")

    doc = Document("this is sample text")
    prepare!(doc, strip_whitespace)
    @test isequal(doc.text, "this is sample text")

    doc = Document("this îs sămple text")
    prepare!(doc, strip_accents)
    @test isequal(doc.text, "this is sample text")

    # stem!(sd)
    # tag_pos!(sd)

    # Do preprocessing on TokenDocument, NGramDocument, Corpus
    d = NGramDocument("this is sample text")
    @test haskey(d.ngrams, "sample")
    StringAnalysis.remove_words!(d, ["sample"])
    @test !haskey(d.ngrams, "sample")

    d = StringDocument(
        """
        <html>
            <head>
                <script language=\"javascript\"> x = 20; </script>
            </head>
            <body>
                <h1>Hello</h1><a href=\"world\">world</a>
            </body>
        </html>
        """
    )
    StringAnalysis.prepare!(d, strip_html_tags)
    @test "Hello  world" == strip(d.text)

    #Test #62
    StringAnalysis.remove_corrupt_utf8("abc") == "abc"
    StringAnalysis.remove_corrupt_utf8(String([0x43, 0xf0])) == "C "
end
