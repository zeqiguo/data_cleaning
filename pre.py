from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from nltk.stem import PorterStemmer
from gensim import utils
from gensim.parsing.preprocessing import remove_stopwords
from gensim.parsing.preprocessing import strip_punctuation
from gensim.parsing.preprocessing import strip_multiple_whitespaces
import pandas as pd
import nltk
import spacy
import re
from unidecode import unidecode
import multiprocessing as mp
from math import log
# basic
import os
import warnings
import json
warnings.filterwarnings("ignore")

# data wrangling & models
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('stopwords')

sp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
porter_stemmer = PorterStemmer()


def string_processor(token):
#     str = str(token)
    str = unidecode(token)
    str = remove_stopwords(str)
    str = strip_punctuation(str)
#    str = strip_non_alphanum(str) # will rm all puncs
    tokens = sp(str)
    tokens = [token.lemma_ for token in tokens]  # lemma_ replace all 'I' to '-PRON-', sorce code bug
#    tokens = [porter_stemmer.stem(token) for token in tokens]
    str = " ".join(tokens)
    str = strip_multiple_whitespaces(str)
    str = str.strip(' ')
    return str

