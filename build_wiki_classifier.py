#!/usr/local/bin/python
#title           :build_wiki_classifier.py
#description     :Scraps specified wikipedia categories and builds a classifier.
#author          :louis
#date            :03/01/16
#version         :0.1
#usage           :python build_wiki_classifier.py  <categories_file_name> <model_save_name>  <vectorizer_save_name>
#python_version  :2.7.10
#==============================================================================

import sys
import re
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
import numpy as np
from random import shuffle
from sklearn.externals import joblib
from wikipedia_scrapper import wiki_scrapper
from nltk import word_tokenize          
from nltk.stem.porter import PorterStemmer




def create_train_test_sets(X,y,shuff=True):
    """Partition data and classes into training and test sets"""
    if shuff:
        combined = zip(X, y)
        shuffle(combined)
        X[:], y[:] = zip(*combined)
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


def create_vectorizer(X):
    """text feature extractor"""
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit(X)
    return vectorizer

def stem_tokens(tokens, stemmer = PorterStemmer()):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def stem_data(X):
    Xnew = list()
    for x in X:
        tokens = word_tokenize(x)
        stems = stem_tokens(tokens, stemmer)
        Xnew.append(" ".join(stems))
    return Xnew

def create_logistic_regression_model():
    """linear model with stochastic gradient descent:"""
    clf = SGDClassifier(loss='modified_huber', penalty='l2',
                        alpha=1e-3, n_iter=20, random_state=42)
    return clf

def print_most_relevant_features(vectorizer, clf, class_labels):
    """Prints features with the highest coefficient values, per class"""
    feature_names = vectorizer.get_feature_names()
    
    for i, class_label in enumerate(class_labels):
        top5 = np.argsort(clf.coef_[i])[-5:]
        print("%s: %s \n" % (class_label," ".join(feature_names[j] for j in top5)))


def main(argv):
    
    try:
        print "\n Model and vectorizer will be saved under %s and %s" % (argv[2]+'.pkl', argv[3]+'.pkl')
        print "\n Loading category file named '%s' " % argv[1]
    except:
        print "\n Usage:python build_wiki_classifier.py  <categories_file_name> <model_save_name>  <vectorizer_save_name>"
        print "example:python build_wiki_classifier.py  categories.txt model_01  vectorize_01"
        return
    
    #read in categories:
    text_file = open(argv[1], 'r')
    lines = text_file.read().split(',')
    
    categories = list()
    max_lvls = list()
    
    print "\n Categories to be classified:"
    for l in lines:
        #wikipedia categories to classify
        categories.append(l.split(' ')[0])
        print l.split(' ')[0]
         #maximum recursion (subcategory) level for each category
        max_lvls.append(int(l.split(' ')[1]))
    
    assert len(categories)==len(max_lvls)
    
    print "\n ***Starting to scrap wikipedia categories:*** \n"
    
    #build the wikipedia scrapper and get the data:
    scrapper = wiki_scrapper(categories,max_lvls)
    
    X,y = scrapper.get_data()
    
    #(Optional) apply a stemming algorithm to the mined text:
    #X = stem_data(X) #Slow, and did not improve resuts

    #split the data into train and test sets
    X_train, X_test, y_train, y_test = create_train_test_sets(X,y)

    #text feature extractor
    vectorizer = create_vectorizer(X_train)
    X_train, X_test = vectorizer.transform(X_train),vectorizer.transform(X_test)

    #Fit a linear model with stochastic gradient descent:
    wiki_clf = create_logistic_regression_model()
    wiki_clf.fit(X_train, y_train)
    
    #save model to the disk
    joblib.dump(wiki_clf, 'models/'+argv[2]+'.pkl') 
    joblib.dump(vectorizer, 'models/'+argv[3]+'.pkl') 
    
    joblib.dump(X, 'models/'+'X.pkl') 
    joblib.dump(y, 'models/'+'y.pkl') 
    
    #Now let's try the model and feed the test batch:
    predicted = wiki_clf.predict(X_test)
    
    #Generate report:
    print "\n ****Most relevant features for each category:****"
    print_most_relevant_features(vectorizer, wiki_clf, categories)
    print "***Total hit rate is %f ***\n" % np.mean(predicted == y_test) 
    
    print "***Classification report: ***"
    print(metrics.classification_report(y_test, predicted,target_names = categories))
    
    print "***Confusion Matrix:***"
    print metrics.confusion_matrix(y_test, predicted)


if __name__ == "__main__":
   main(sys.argv)
