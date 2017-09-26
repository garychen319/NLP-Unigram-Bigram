import csv
from csv import writer
import glob
import re
import codecs
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

train_path = "aclImdb/train/" # source data
test_path = "imdb_te.csv" # test data for grade evaluation. 

def main():
    imdb_data_preprocess(train_path)

def imdb_data_preprocess(inpath, outpath="./", name="imdb_tr.csv", mix=False):
 	
    pos_path = train_path + "pos"
    neg_path = train_path + "neg"

    #format stopwords file into array of words
    stopwords = []
    with open("stopwords.en.txt") as f:
        stopwords = f.readlines()
    stopwords = [x.strip() for x in stopwords]

    #get x_train from pos/neg folders with review txt files.
    x_train = []
    pos_textarr = make_textarr(pos_path, stopwords)
    neg_textarr = make_textarr(neg_path, stopwords)

    x_train = pos_textarr + neg_textarr #combine pos and neg review arrays
    csv_writer(pos_textarr, neg_textarr, name) #write imdb_tr.csv

    #y_train is an array of 25000 1's and then 25000 0's
    y_train = []
    for i in range(0, len(pos_textarr)):
        y_train.append(1)
    for i in range(0, len(neg_textarr)):
        y_train.append(0)

    #read x_test from csv file
    x_test = open_csv(test_path, stopwords)

    # print("x_train shape: ", x_train_counts.shape)
    # print("x_test shape: ", x_test_counts.shape)
    # print("x_train len: ", len(x_train))
    # print("y_train len: ", len(y_train))
    # print("x_test len: ", len(x_test))

    #Call the 4 SGDClassifier functions with different specifications and output to text files.
    print("Accuracies of each fit:\n")
    unigram(x_train, y_train, x_test)
    bigram(x_train, y_train, x_test)
    tfidf_unigram(x_train, y_train, x_test)
    tfidf_bigram(x_train, y_train, x_test)


def make_textarr(path, stopwords):
    #Convert txt file into text array, with stopwords filtered out.
    textarr = []
    files = glob.glob(path + "/*.txt")
    for fle in files:
        with open(fle) as f:
            for line in f:
                sentence = ""
                for word in re.split('\W+', line):
                    if word not in stopwords:
                        sentence += (word + " ")
                textarr.append(sentence[:-1])
    return textarr

def open_csv(path, stopwords):
    #Read reviews from csv and return the column of text reviews, with stopwords filtered out.
    reviews = []
    with codecs.open(path, "r",encoding='utf-8', errors='ignore') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader, None) #skip header
        for row in reader:
            sentence = ""
            for word in re.split('\W+', row[1]):
                if word not in stopwords:
                    sentence += (word + " ")
            reviews.append(sentence[:-1])
    return reviews

def csv_writer(pos_data, neg_data, path):
    #Part 1, write data to csv.
    #pos reviews
    n = len(pos_data)
    rownum = range(0, n)
    wtr = csv.writer(open (path, 'w'), delimiter=',', lineterminator='\n')
    wtr.writerow (["row_number", "text", "polarity"])
    for i in range (0, n):
        wtr.writerow ([rownum[i], pos_data[i], 1])
    #neg reviews
    n2 = len(neg_data)
    rownum = range(n, n+n2)
    wtr = csv.writer(open (path, 'a'), delimiter=',', lineterminator='\n')
    for i in range (0, n2):
        wtr.writerow ([rownum[i], neg_data[i], 0])

def unigram(x_train, y_train, x_test):
    #First transform x_train, x_test to unigrams via CountVectorizer
    count_vect = CountVectorizer()
    x_train_counts = count_vect.fit_transform(x_train)  
    x_test_counts = count_vect.transform(x_test)

    #Use SGDClassifier to fit data and predict x_test and output results to file
    clf = SGDClassifier(loss="hinge", penalty="l1")
    clf.fit(x_train_counts, y_train)
    y_pred = clf.predict(x_test_counts)
    output = open('unigram.output.txt', 'w')
    for item in y_pred:
        output.write("%s\n" % item)

    print("unigram score:")
    print(clf.fit(x_train_counts, y_train).score(x_train_counts, y_train))

def bigram(x_train, y_train, x_test):
    #First transform into bigram, then use SGDClassifier to predict x_test
    bigram_vect = CountVectorizer(ngram_range=(2, 2))
    x_train_counts = bigram_vect.fit_transform(x_train)  
    x_test_counts = bigram_vect.transform(x_test)

    clf = SGDClassifier(loss="hinge", penalty="l1")
    clf.fit(x_train_counts, y_train)
    y_pred = clf.predict(x_test_counts)
    output = open('bigram.output.txt', 'w')
    for item in y_pred:
        output.write("%s\n" % item)

    print("bigram score:")
    print(clf.fit(x_train_counts, y_train).score(x_train_counts, y_train))

def tfidf_unigram(x_train, y_train, x_test):
    #Using tfidfvectorizer instead of count_vectorizer
    tu_vect = TfidfVectorizer()
    x_train_counts = tu_vect.fit_transform(x_train)  
    x_test_counts = tu_vect.transform(x_test)
    clf = SGDClassifier(loss="hinge", penalty="l1")
    clf.fit(x_train_counts, y_train)
    y_pred = clf.predict(x_test_counts)
    output = open('unigramtfidf.output.txt', 'w')
    for item in y_pred:
        output.write("%s\n" % item)

    print("tfidf unigram score:")
    print(clf.fit(x_train_counts, y_train).score(x_train_counts, y_train))

def tfidf_bigram(x_train, y_train, x_test):
    #use tfidf vectorizer with bigram
    tb_vect = TfidfVectorizer(ngram_range=(2, 2))
    x_train_counts = tb_vect.fit_transform(x_train)  
    x_test_counts = tb_vect.transform(x_test)
    clf = SGDClassifier(loss="hinge", penalty="l1")
    clf.fit(x_train_counts, y_train)
    y_pred = clf.predict(x_test_counts)
    output = open('bigramtfidf.output.txt', 'w')
    for item in y_pred:
        output.write("%s\n" % item)

    print("tfidf bigram score:")
    print(clf.fit(x_train_counts, y_train).score(x_train_counts, y_train))





main()