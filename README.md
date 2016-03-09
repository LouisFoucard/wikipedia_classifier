Wikipedia classifier Louis Foucard

Description

This python classifier can take in any number of Wikipedia category, scrap through the articles and subcategory to collect text in each category, construct the term document matrix and fit a stochastic gradient descent classifier. The machine learning python library sklearn is used to preprocess the mined text and to generate/fit the model.
There are 3 files:
- wikipedia_scrapper.py: python module that uses BeautifulSoup and urllib2 to recursively scrap through the specified categories. Takes as input a list of Wikipedia categories, and the maximum level of subcategories to search.
- build_wiki_classifier.py: main python script that collects the text using the scrapper, builds the model and creates a classifier report that describes the model’s performance. It takes as input a text file with any number of Wikipedia categories followed by the maximum level of subcategory, and the names under which to save the model.
- category_predicter.py: python scripts that loads the serialized model and predicts the category (and probability for each category) for a Wikipedia url. It takes as input the names of the model and the url to classify.
The text mined from the Wikipedia categories is first preprocessed by removing stop words, punctuation and by generating the term frequency–inverse document frequency matrix. Once the term frequency matrix is built, a classifier model is fitted.

Dependencies

BeautifulSoup, urllib2, re, sklearn, numpy, nltk

Usage

To build the classifier (5-15mn depending on internet connection):
python build_wiki_classifier.py categories_file_name model_save_name vectorizer_save_name
To run the classifier:
python category_predicter.py categories_file_name model_name> <vectorizer_name url
 


