#!/usr/local/bin/python
#title           :category_predicter.py
#description     :Scraps specified url and predicts category based on specified model.
#author          :louis
#date            :03/01/16
#version         :0.1
#usage           :python category_predicter.py  <model_name>  <vectorizer_name> <url>
#python_version  :2.7.10
#==============================================================================

from bs4 import BeautifulSoup
import urllib2
import re
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
import numpy as np
from sklearn.externals import joblib
import sys
from wikipedia_scrapper import wiki_scrapper


def main(argv):
    
    try:
        print "\n url to classify is: %s" % argv[4]
        print "\n Model and vectorizer are %s and %s" % (argv[2]+'.pkl', argv[3]+'.pkl')
    except:
        print "\n Usage: python category_predicter.py  <categories_file_name> <model_name>  <vectorizer_name> <url>"
        print "example: python category_predicter.py categories.txt model vectorizer https://en.wikipedia.org/wiki/Mycotic_aneurysm"
        return
    
    
    #read in categories:
    text_file = open(argv[1], 'r')
    lines = text_file.read().split(',')
    
    categories = list()
    
    for l in lines:
        #wikipedia categories to classify
        categories.append(l.split(' ')[0])
    
    
    model_name = argv[2]
    vectorizer_name = argv[3]
    url = argv[4]
    
    wiki_clf = joblib.load('models/'+model_name+'.pkl') 
    vectorizer = joblib.load('models/'+vectorizer_name+'.pkl') 

    #build the wikipedia scrapper and get the data:
    scrapper = wiki_scrapper([],[]) 
    text = scrapper.fetch_url_text(url)

    text = vectorizer.transform([text])
    
    cat = wiki_clf.predict(text)
    probs = wiki_clf.predict_proba(text)
    
    print "\n*** probability of belonging to each category*** \n"

    for p,c in zip(probs[0],categories):
        print c+":","%.2f" % p
    
    print "\n ==> The predicted category is: %s \n" % (categories[cat-1])


# main("https://en.wikipedia.org/wiki/Data_mining")
if __name__ == "__main__":
   main(sys.argv)
