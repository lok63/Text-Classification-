import pandas as pd
import re
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
from nltk import pos_tag, word_tokenize

stop_words = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

class NLP_Preprocessor(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):

        self.variables = variables

    def fit(self, X, y=None):
        # we need the fit statement to accomodate the sklearn pipeline
        return self

    def transform(self, X):
        X = X.copy()
        X = X.apply(lambda x: x.lower()).\
                apply(lambda x: decontracted(x)).\
                apply(lambda x: " ".join([item for item in x.split() if item not in stop_words])).\
                apply(lambda x: replace_num(x)).\
                apply(lambda x: replace_orderID(x)).\
                apply(lambda x: lemmatize(x)).\
                apply(lambda x: remove_punc(x)).\
                apply(lambda x: remove_extra_space(x))
        print("--------------------")
        return X


def lemmatize(phraze):
    new_phraze = []
    for word, tag in pos_tag(word_tokenize(phraze)):
        wntag = tag[0].lower()
        wntag = wntag if wntag in ['a', 'r', 'n', 'v'] else None
        if not wntag:
            new_phraze.append(word)
        else:
            new_phraze.append(lemmatizer.lemmatize(word, wntag))
        
    return " ".join(new_phraze)

def replace_num(phrase):
    return re.sub(" \d+", " _number_", phrase)

def replace_orderID(phrase):
    return re.sub("([A-Za-z]+[\d@]+[\w@]*|[\d@]+[A-Za-z]+[\w@]*)", "_orderID_", phrase)

def remove_punc(phraze):
    return re.sub(r'[^\w\s]',"",phraze)

def remove_extra_space(phraze):
    return re.sub(' +', ' ', phraze)
    
def decontracted(phrase):
    # specific
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"hasn\’t", "has not", phrase)
    phrase = re.sub(r"haven\’t", "has not", phrase)
    phrase = re.sub(r"\’d", " would", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase