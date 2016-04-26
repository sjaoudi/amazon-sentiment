import csv
import numpy as np
from random import shuffle
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import HashingVectorizer, CountVectorizer 
from sklearn.naive_bayes import BernoulliNB
from sklearn import metrics
    

def importReviews(filename):
    reviews = []
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            reviews.append(row)
    
    return reviews

def splitReviews(reviews):
    positive = []
    negative = []

    helpful = []
    unhelpful = []

    for review in reviews:
        score = int(review['Score'])
        if score > 3:
            positive.append(review)
        if score < 3:
            negative.append(review)
    
        helpful_num = float(review['HelpfulnessNumerator'])
        helpful_denom = float(review['HelpfulnessDenominator'])
        
        if helpful_denom: 
            helpful_ratio = (helpful_num / helpful_denom)
            
            if helpful_ratio > .5:
                helpful.append(review)
            if helpful_ratio < .5:
                unhelpful.append(review)

    return {
            'positive': positive, 
            'negative': negative, 
            'helpful': helpful,
            'unhelpful': unhelpful
            }

def train(categorized):

    class_0 = getText(categorized['helpful'])
    class_1 = getText(categorized['unhelpful'])


    x_data = []
    x_data.extend(class_0)
    x_data.extend(class_1)

    y_data = []
    y_data.extend(np.zeros(len(class_0)))
    y_data.extend(np.ones(len(class_1)))

    assert(len(x_data) == len(y_data))

    z = zip(x_data, y_data)
    shuffle(z)
    x_data, y_data = zip(*z)

    cutoff = len(x_data) * 3/4
    #print x_data[0]
    #print y_data[0]
    #exit()

    vectorizer = HashingVectorizer(stop_words = 'english', non_negative = True, n_features = 2)
    x_train = vectorizer.transform(x_data[:cutoff])
    y_train = y_data[:cutoff]

    x_test = vectorizer.transform(x_data[cutoff:])
    y_test = y_data[cutoff:]

    nb = BernoulliNB()
    nb.fit(x_train, y_train)

    prediction = nb.predict(x_test)
    score = metrics.accuracy_score(y_test, prediction)
    print ("accuracy: %0.3f" % score)

    show_most_informative_features(vectorizer, nb) 


def getText(reviews):

    text = []    
    #tokenizer = RegexpTokenizer(r'\w+')
    
    for review in reviews:
        #text.extend( tokenizer.tokenize( review['Text'].lower() ) )
        text.append( review['Text'].lower() )

    return text

def show_most_informative_features(vectorizer, clf, n=20):
    feature_names = vectorizer.get_feature_names()
    coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
    top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1])
    for (coef_1, fn_1), (coef_2, fn_2) in top:
        print "\t%.4f\t%-15s\t\t%.4f\t%-15s" % (coef_1, fn_1, coef_2, fn_2)


if __name__ == "__main__":
    reviews = importReviews('reviews-10000.csv')
    categorized = splitReviews(reviews)
    train(categorized)
    


