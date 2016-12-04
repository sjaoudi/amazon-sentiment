import csv
from time import time
import numpy as np
from random import shuffle
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import SGDClassifier, Perceptron, PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn import metrics
import matplotlib.pyplot as plt

"""
Class with methods for preprocessing the csv and categorizing the data.
"""
class PreProcessing():
    def __init__(self):

        # Uncomment the pair of categories the classifier should predict

        # self.category_0 = 'helpful'
        # self.category_1 = 'unhelpful'

        self.category_0 = 'negative'
        self.category_1 = 'positive'

    def importReviews(self, filename):
        reviews = []
        with open(filename) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                reviews.append(row)

        return reviews


    def categorize(self, reviews):
        positive = []
        negative = []

        one_star = two_star = three_star = four_star = five_star = []

        # helpful = []
        # unhelpful = []

        for review in reviews:
            score = int(review['Score'])
            if score > 3:
                positive.append(review)
            if score < 3:
                negative.append(review)

            if score == 1:
                one_star.append(review)
            if score == 2:
                two_star.append(review)
            if score == 3:
                three_star.append(review)
            if score == 4:
                four_star.append(review)
            if score == 5:
                five_star.append(review)


            # helpful_num = float(review['HelpfulnessNumerator'])
            # helpful_denom = float(review['HelpfulnessDenominator'])

            # if helpful_denom:
            #     helpful_ratio = (helpful_num / helpful_denom)
            #
            #     # "Helpful"
            #     if helpful_ratio > .5:
            #         helpful.append(review)
            #
            #     # "Unhelpful"
            #     if helpful_ratio < .5:
            #         unhelpful.append(review)

        # return {
        #     'positive': positive,
        #     'negative': negative,
        #     'helpful': helpful,
        #     'unhelpful': unhelpful
        #     }
        return {
            "positive": positive,
            "negative": negative,
            "one_star": one_star,
            "two_star": two_star,
            "three_star": three_star,
            "four_star": four_star,
            "five_star": five_star
        }


    def getText(self, reviews):
        text = []
        for review in reviews:
            text.append( review['Text'].lower() )

        return text


    def splitData(self, categorized):

        class_0 = self.getText(categorized[self.category_0])
        class_1 = self.getText(categorized[self.category_1])

        shorter = len(class_0) if len(class_0) < len(class_1) else len(class_1)
        class_0 = class_0[:shorter]
        class_1 = class_1[:shorter]

        print len(class_0), len(class_1)
        assert(len(class_0) == len(class_1))

        x_data = []
        x_data.extend(class_0)
        x_data.extend(class_1)

        # Label for class_0 is a 0, label for class_1 is a 1
        y_data = []
        y_data.extend(np.zeros(len(class_0)))
        y_data.extend(np.ones(len(class_1)))

        assert(len(x_data) == len(y_data))

        z = zip(x_data, y_data)
        shuffle(z)
        x_data, y_data = zip(*z)

        return x_data, y_data


"""
Class with methods for training and testing algorithms, as well as evaluating
and plotting performance.

The "benchmkark", "plot" and "topFeatures" methods are based on methods found in this example:
http://scikit-learn.org/stable/auto_examples/text/document_classification_20newsgroups.html

The "gridSearch" methods is from this example:
http://scikit-learn.org/stable/auto_examples/model_selection/grid_search_text_feature_extraction.html
"""
class Classifiers():

    def __init__(self, filename):
        self.class_0 = []
        self.class_1 = []

        self.x_data = []
        self.y_data = []

        self.x_train = self.x_test = []
        self.y_train = self.y_test = []

        # Run the preprocessing functions from the PreProcessing class
        self.preprocessData()

        self.classifiers = {
            # "LinearSVC": LinearSVC(loss='squared_hinge', penalty='l1', dual=False, tol=1e-3),
            # "SGD": SGDClassifier(alpha=.0001, n_iter=50, penalty="elasticnet"),
            # "Bernoulli Naive Bayes": BernoulliNB(alpha=.01),
            # "Ridge": RidgeClassifier(tol=1e-2, solver="sag"),
            # "Perceptron": Perceptron(n_iter=50),
            # "Passive Aggressive": PassiveAggressiveClassifier(n_iter=50),
            # "Multinomial NB": MultinomialNB(),
            # "Log-reg": LogisticRegression(C=1e5)
            # "KNN": KNeighborsClassifier(n_neighbors=10),
            "Random Forest": RandomForestClassifier(n_estimators=100),
            # "SVC": SVC(),
            }

        self.vectorizers = {
            "Tfidf": TfidfVectorizer(
             stop_words = 'english',
             token_pattern=r'\b\w+\b',
             sublinear_tf = True,
             use_idf = 'True', max_df = .50, norm = 'l2'),
            "Count": CountVectorizer(stop_words = 'english', max_df = .5, max_features = 5000, ngram_range = (1, 1))
        }

        self.prediction = dict()


    """
    Instantiate the PreProcessing class for importing and categorizing the data
    """
    def preprocessData(self):
        preprocess = PreProcessing()

        reviews = preprocess.importReviews(filename)
        categorized = preprocess.categorize(reviews)
        self.x_data, self.y_data = preprocess.splitData(categorized)


    def train(self, vectorizer):
        assert(len(self.x_data) == len(self.y_data))
        cutoff = len(self.x_data) * 3/4

        self.x_train = vectorizer.fit_transform(self.x_data[:cutoff])
        self.y_train = self.y_data[:cutoff]

        self.x_test = vectorizer.transform(self.x_data[cutoff:])
        self.y_test = self.y_data[cutoff:]


    """
    Trains the classifier and predicts the label. Also times the training and
    testing stages for plotting.
    """
    def benchmark(self, classifier):
        desc = str(classifier).split('(')[0]

        pre_train = time()
        classifier.fit(self.x_train, self.y_train)
        train_time = time() - pre_train

        pre_test = time()
        prediction = classifier.predict(self.x_test)
        test_time = time() - pre_test

        score = metrics.accuracy_score(self.y_test, prediction)
        print "accuracy: %0.5f" % score
        return desc, score, train_time, test_time

    def predict(self, classifier):

        model = classifier.fit(self.x_train, self.y_train)
        prediction = model.predict(self.x_test)

        return prediction

    def runClassifiers(self):

        vectorizer = self.vectorizers["Tfidf"]
        # vectorizer = self.vectorizers["Count"]
        self.train(vectorizer)

        # List of classifier description, score, train time, and test time,
        # which is necessary for plotting
        results = []

        for classifier_name in self.classifiers:
            print classifier_name

            classifier = self.classifiers[classifier_name]
            result = self.benchmark(classifier)
            print result
            results.append(result)

            prediction = self.predict(classifier)
            self.prediction[classifier_name] = prediction

            # Prints the most informative features for the classifier
            # self.topFeatures(classifier, vectorizer)
            self.plot_confusion_matrix(classifier_name)
            print ''
            break

        # classifier = self.classifiers["LinearSVC"]
        # result = self.benchmark(classifier)
        # results.append(result)
        #
        # prediction = self.predict(classifier)
        # self.prediction["LinearSVC"] = prediction

        # self.plot(results)
        # print self.prediction

        # Compute confusion matrix

        # print(metrics.classification_report(self.y_test, self.prediction['LinearSVC'], digits=3, target_names = ["positive", "negative"]))
        # self.plotROC()


    def topFeatures(self, classifier, vectorizer):
        f = open('out.txt', 'w')
        feature_names = np.asarray(vectorizer.get_feature_names())
        categories = list(set(self.y_train))
        top10 = np.argsort(classifier.coef_[0])[-100:]
        ranks =  zip(feature_names[top10], top10)
        print feature_names[top10]

        for r in ranks:
            name, score = r
            # print name, score
            f.write(str(score / 100))
            f.write(' ')
            f.write(str(name))
            f.write('\n')
        # print feature_names[top10]


    def plot(self, results):

        indices = np.arange(len(results))
        results = [[x[i] for x in results] for i in range(4)]

        clf_names, score, training_time, test_time = results
        training_time = np.array(training_time) / np.max(training_time)
        test_time = np.array(test_time) / np.max(test_time)

        plt.figure(figsize=(12, 8))
        plt.title("Score")

        plt.barh(indices, score, .2, label="score", color='r')
        plt.barh(indices + .3, training_time, .2, label="training time", color='g')
        plt.barh(indices + .6, test_time, .2, label="test time", color='b')

        plt.yticks(())

        plt.subplots_adjust(left=.25)
        plt.subplots_adjust(top=.95)
        plt.subplots_adjust(bottom=.05)

        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3, fancybox=True, shadow=True)

        for i, c in zip(indices, clf_names):
            print i, c
            plt.text(-.3, i, c)

        plt.show()


    def plot_confusion_matrix(self, model):

        # print self.prediction.keys()
        y_pred =  self.prediction[model]

        cmap=plt.cm.Reds
        cm = metrics.confusion_matrix(self.y_test, y_pred)
        np.set_printoptions(precision=3)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print cm_normalized
        # np.set_printoptions(precision=2)
        # plt.figure()
        #
        # plt.imshow(cm_normalized, interpolation='nearest', cmap=cmap)
        # title='Normalized Confusion matrix'
        # plt.title(title)
        # plt.colorbar()
        #
        # tick_marks = np.arange(len(set(self.y_test)))
        # plt.xticks(tick_marks, set(self.y_test), rotation=45)
        # plt.yticks(tick_marks, set(self.y_test))
        # plt.tight_layout()
        # plt.ylabel('True label')
        # plt.xlabel('Predicted label')
        #
        # plt.show()

    def plotROC(self):
        # Compute ROC curve and ROC area for each class

        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        n_classes = 2

        cmp = 0
        colors = ['b', 'g', 'y', 'm', 'k', 'r', 'b', 'r']
        for model, predicted in self.prediction.items():
            false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(self.y_test, predicted)
            roc_auc = metrics.auc(false_positive_rate, true_positive_rate)
            plt.plot(false_positive_rate, true_positive_rate, colors[cmp], label='%s: AUC %0.2f'% (model,roc_auc))
            cmp += 1
            print model, false_positive_rate, true_positive_rate

            # fpr[model], tpr[model], _ = metrics.roc_curve(self.y_test, predicted)
            # roc_auc[model] = metrics.auc(fpr[model], tpr[model])
            # plt.plot(false_positive_rate, true_positive_rate, colors[cmp], label='%s: AUC %0.2f'% (model,roc_auc))
            # cmp += 1
            # print model, false_positive_rate, true_positive_rate


        # all_predictions = np.concatenate([self.prediction[key] for key in self.prediction])
        # fpr["micro"], tpr["micro"], _ = metrics.roc_curve(self.y_test, all_predictions.ravel())
        # roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])


        plt.title('Classifiers comparaison with ROC')
        plt.legend(loc='lower right')
        plt.plot([0,1],[0,1],'r--')
        plt.xlim([0,1.0])
        plt.ylim([0,1.0])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()


    def runVectorizers(self):

        classifier = self.classifiers["Bernoulli Naive Bayes"]

        for vectorizer_name in self.vectorizers:
            vectorizer = self.vectorizers[vectorizer_name]
            self.train(vectorizer)
            self.benchmark(classifier)


    def gridSearch(self):
        pipeline = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', LinearSVC()) ])

        # uncommenting more parameters will give better exploring power but will
        # increase processing time in a combinatorial way
        parameters = {
            'vect__max_df': (0.5, 0.75, 1.0),
            'vect__max_features': (None, 5000, 10000, 50000),
            'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
            'tfidf__use_idf': (True, False),
            'tfidf__norm': ('l1', 'l2'),
            #'clf__alpha': (0.0001, 0.00001, 0.000001),
            #'clf__penalty': ('l2', 'elasticnet'),
            #'clf__n_iter': (10, 50, 80),
            #'clf__loss': ('squared_hinge', 'hinge')
        }

        # multiprocessing requires the fork to happen in a __main__ protected
        # block

        # find the best parameters for both the feature extraction and the
        # classifier
        grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)

        print("Performing grid search...")
        print("pipeline:", [name for name, _ in pipeline.steps])
        print("parameters:")
        print(parameters)
        t0 = time()
        grid_search.fit(self.x_data, self.y_data)
        print("done in %0.3fs" % (time() - t0))
        print ''

        print("Best score: %0.3f" % grid_search.best_score_)
        print("Best parameters set:")
        best_parameters = grid_search.best_estimator_.get_params()
        print grid_search.best_estimator_
        for param_name in sorted(parameters.keys()):
            print("\t%s: %r" % (param_name, best_parameters[param_name]))


if __name__ == "__main__":
    # filename = 'reviews-1000.csv'
    filename = 'reviews-10000.csv'
    # filename = 'all-reviews.csv'

    classifiers = Classifiers(filename)

    classifiers.runClassifiers()
    #classifiers.gridSearch()
