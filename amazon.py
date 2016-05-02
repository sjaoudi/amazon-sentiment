import csv
import numpy as np
from random import shuffle
import nltk.classify.util
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC 
from sklearn.svm import LinearSVC
from sklearn.svm import NuSVC
from sklearn.svm import SVR
from sklearn.svm import LinearSVR
from sklearn.svm import NuSVR
from sklearn.svm import OneClassSVM
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.utils.extmath import density
from sklearn import metrics

class PreProcessing():
    
    def importReviews(self, filename):
        reviews = []
        with open(filename) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                reviews.append(row)
        
        return reviews

    def splitReviews(self, reviews):
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

class Classifiers(PreProcessing):
    def __init__(self):
        self.class_0 = []
        self.class_1 = []

        self.x_data = []
        self.y_data = []

        self.x_train = self.x_test = []
        self.y_train = self.y_test = []

        #cl = LinearSVC(algorithm='SAMME')
        cl = BernoulliNB()
        cl = DecisionTreeClassifier()
        self.classifiers = [
                LinearSVC(loss='squared_hinge', penalty='l1', dual=False, tol=1e-3),
                #SGDClassifier(alpha=.0001, n_iter=50, penalty="elasticnet"),
                #BernoulliNB(alpha=.01),
                #MultinomialNB(alpha=.01),
                #RidgeClassifier(tol=1e-2, solver="sag"),
                #Perceptron(n_iter=50),
                #PassiveAggressiveClassifier(n_iter=50),
                #KNeighborsClassifier(n_neighbors=10), 
                #RandomForestClassifier(n_estimators=100),
                #SVC(),
                #NuSVC(),
                #SVR(),
                #LinearSVR(),

                AdaBoostClassifier(n_estimators=10, base_estimator=cl)


                ]

    def split_data(self, categorized):

        class_0 = self.getText(categorized['positive'])  	
        class_1 = self.getText(categorized['negative'])

        self.x_data.extend(class_0)
        self.x_data.extend(class_1)

        self.y_data.extend(np.zeros(len(class_0)))
        self.y_data.extend(np.ones(len(class_1)))

        assert(len(self.x_data) == len(self.y_data))

        z = zip(self.x_data, self.y_data)
        shuffle(z)
        self.x_data, self.y_data = zip(*z)

        cutoff = len(self.x_data) * 3/4

        #vectorizer = HashingVectorizer(stop_words = 'english', non_negative = True, n_features = 10)
        vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=.5)
        
        #self.x_train = vectorizer.transform(self.x_data[:cutoff])
        self.x_train = vectorizer.fit_transform(self.x_data[:cutoff])
        self.y_train = self.y_data[:cutoff]

        self.x_test = vectorizer.transform(self.x_data[cutoff:])
        self.y_test = self.y_data[cutoff:]

    def benchmark(self, classifier):
        
        classifier.fit(self.x_train, self.y_train)
        prediction = classifier.predict(self.x_test)

        score = metrics.accuracy_score(self.y_test, prediction)
        print ("accuracy: %0.5f" % score)

    def run_classifiers(self):
        
        for classifier in self.classifiers:
            self.benchmark(classifier)

    def getText(self, reviews):

        text = []    
        #tokenizer = RegexpTokenizer(r'\w+')
        
        for review in reviews:
            #text.extend( tokenizer.tokenize( review['Text'].lower() ) )
            text.append( review['Text'].lower() )

        return text


if __name__ == "__main__":
    preprocessing = PreProcessing()
    reviews = preprocessing.importReviews('reviews-10000.csv')
    #reviews = preprocessing.importReviews('all-reviews.csv')
    categorized = preprocessing.splitReviews(reviews)

    classifiers = Classifiers()    
    classifiers.split_data(categorized)
    classifiers.run_classifiers()

