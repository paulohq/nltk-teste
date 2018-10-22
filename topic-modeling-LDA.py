#https://towardsdatascience.com/topic-modeling-and-latent-dirichlet-allocation-in-python-9bf156893c24

#Topic modeling is a type of statistical modeling for discovering the abstract “topics” that occur in a collection of documents.
# Latent Dirichlet Allocation (LDA) is an example of topic model and is used to classify text in a document to a particular topic.
# It builds a topic per document model and words per topic model, modeled as Dirichlet distributions.

import pandas as pd

data = pd.read_csv('abcnews-date-text.csv', error_bad_lines=False);
data_text = data[['headline_text']]
data_text['index'] = data_text.index
documents = data_text

print(len(documents))
print(documents[:5])

import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
np.random.seed(2018)

import nltk
#nltk.download('wordnet')

#Data Pre-processing
def lemmatize_stemming(text):
    lmtzr = WordNetLemmatizer()
    porter_stemmer = PorterStemmer()
    return porter_stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result

doc_sample = documents[documents['index'] == 4310].values[0][0]

print('original document: ')
words = []
for word in doc_sample.split(' '):
    words.append(word)
print(words)
print('\n\n tokenized and lemmatized document: ')
print(preprocess(doc_sample))

#Preprocess the headline text, saving the results as ‘processed_docs’
processed_docs = documents['headline_text'].map(preprocess)
print(processed_docs[:10])