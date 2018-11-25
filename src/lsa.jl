# LSA
lsa(dtm::DocumentTermMatrix) = svd(Matrix(tf_idf(dtm)))
lsa(crps::Corpus) = svd(Matrix(tf_idf(DocumentTermMatrix(crps))))
