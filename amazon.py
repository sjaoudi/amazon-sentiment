import csv
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import CountVectorizer
    

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
        if review['Score'] > 3:
            positive.append(review)
        if review['Score'] < 3:
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

    helpful = getText(categorized['helpful'])
    unhelpful = getText(categorized['unhelpful'])

    helpfulCutoff = len(helpful) * 3/4
    unhelpfulCutoff = len(unhelpful) * 3/4

    print helpfulCutoff

    trainFeats = helpful[:helpfulCutoff] + unhelpful[:unhelpfulCutoff]
    testFeats = helpful[helpfulCutoff:] + unhelpful[unhelpfulCutoff:]

    classifier = NaiveBayesClassifier.train(trainFeats)
    print 'accuracy:', nltk.classify.util.accuracy(classifier, testFeats)
    classifier.show_most_informative_features() 

def getText(list):

    text = []    
    tokenizer = RegexpTokenizer(r'\w+')
    tokenizer.tokenize('Eighty-seven miles to go, yet.  Onward!')
    
    for review in reviews:
        text.extend( tokenizer.tokenize( review['Text'].lower() ) )

    return text

if __name__ == "__main__":
    reviews = importReviews('reviews-10000.csv')
    categorized = splitReviews(reviews)
    train(categorized)
    
