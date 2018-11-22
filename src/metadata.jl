# Medatadata getter for documents
metadata(document::D) where D<:AbstractDocument = document.metadata
metadata(crps::C) where C<:Corpus = [doc.metadata for doc in crps]


# Generate utility functions that operate on the DocumentMetadata's fields; these
# are generated dinamically so that adding more methods metadata fields will
# automatically add new methods as well ;)
for fieldname in fieldnames(DocumentMetadata)
    # Create document metadata accessors
    definition = """
        function $(fieldname)(d::AbstractDocument)
            return d.metadata.$(fieldname)
        end
        """
    eval(Meta.parse(definition))
    # Create document metadata setters
    if fieldname != :language
        definition = """
            function $(fieldname)!(d::AbstractDocument, nv::String)
                d.metadata.$(fieldname) = nv
            end
            """
    else
        definition = """
            function $(fieldname)!(d::AbstractDocument, nv::L) where L<:Language
                d.metadata.$(fieldname) = nv
            end
            """
    end
    eval(Meta.parse(definition))
    # Vectorized getters for an entire Corpus
    definition = """
        function $(fieldname)s(c::Corpus)
            map(d->$(fieldname)(d), documents(c))
        end
        """
    eval(Meta.parse(definition))
    # Vectorized setters for an entire Corpus (same value)
    if fieldname in [:author, :timestamp]
        definition = """
            $(fieldname)s!(c::Corpus, nv::String) =
                $(fieldname)!.(documents(c), Ref(nv))
            """
    elseif fieldname == :language
        definition = """
            $(fieldname)s!(c::Corpus, nv::L) where L<:Language =
                $(fieldname)!.(documents(c), Ref(nv))
            """
    else
        definition = """
            $(fieldname)s!(c::Corpus, nv::String) =
                $(fieldname)!.(documents(c), nv)
            """
    end
    eval(Meta.parse(definition))
    # Vectorized setters for an entire Corpus (different values)
    definition = """
        function $(fieldname)s!(c::Corpus, nvs::AbstractVector)
            length(c) == length(nvs) || throw(DimensionMismatch("dimensions must match"))
            for (i, d) in pairs(IndexLinear(), documents(c))
                $(fieldname)!(d, nvs[i])
            end
        end
        """
    eval(Meta.parse(definition))
end
