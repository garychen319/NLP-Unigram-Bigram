Artificial Intelligence Assignment 5 README
Gary Chen gc2676

To run: python3 driver_3.py

Outputted files will be under the same directory.
Input files required to be under the parent directory to run:
	train_path = "../aclImdb/train/"
	test_path = "../imdb_te.csv"
	stopwords = "../stopwords.en.txt"


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