AI - Natural Language Processing

To run: python3 driver_3.py

Outputted files will be under the same directory.
Input files required to be under the parent directory to run:
	train_path = "../aclImdb/train/"
	test_path = "../imdb_te.csv"
	stopwords = "../stopwords.en.txt"

-Data trained using Stochastic Gradient Descent Classifier, good for text data since text corpus are often huge.

-English stopwords in stopwords.en.txt are removed.

-3 representations - Unigram, Bigram, Tfidf (Term Frequency Inverse Document Frequency)

Achieved accuracy score below for reference:
(A higher accuracy was achieved using both unigrams and bigrams (passing (1,2) into countVectorizer)
but the code uses just unigrams (passing in (2,2) instead.))

unigram score:
0.90952
bigram score:
0.92172
tfidf unigram score:
0.886
tfidf bigram score:
0.721
