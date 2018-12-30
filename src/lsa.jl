# LSA
"""
Perform [Latent Semantic Analysis](https://en.wikipedia.org/wiki/Latent_semantic_analysis).
"""
lsa(dtm::DocumentTermMatrix) = svd(Matrix(tf_idf(dtm)))
lsa(crps::Corpus) = svd(Matrix(tf_idf(DocumentTermMatrix(crps))))
