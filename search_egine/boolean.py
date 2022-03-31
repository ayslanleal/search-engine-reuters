import pandas as pd
from nltk.stem import RSLPStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
import numpy as np
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 
def clean_stopwords(instance):
    words = instance.lower().split()
    stops = set(stopwords.words('english'))
    join = [] 
    for i in words:
        if i not in stops:
            join.append(i)
    return ' '.join(join)

def stemming(instance):
    stemmer = SnowballStemmer("english")
    words = instance.lower().split()
    stems = []
    for i in words:
        stems.append(stemmer.stem(i))
    return ' '.join(stems) 

def remove_alpha(instance):
    words = instance.split()
    alpha = []
    for i in words:
        if i.isalnum():
            alpha.append(i)
    return ' '.join(alpha)

def tokenize(text):
    words = word_tokenize(text)
    words = [word.lower() for word in words]
    return words

def search(query, df):
    query = tokenize(query)
    lista_retorno = []
    for i in query:
        if i in df.columns:
            lista_retorno.append(df[df[i]>=1].index)
    return lista_retorno


if __name__ == "__main__":
    df = pd.read_json('./teste.json')
    df['body'] = df['body'].apply(clean_stopwords).apply(stemming).apply(remove_alpha)

    vec = CountVectorizer()
    x = vec.fit_transform(df.body)
    novo_df = pd.DataFrame(x.toarray(), columns=vec.get_feature_names())
    novo_df["title"] = df['title']
    novo_df = novo_df.set_index('title')
    print(search('brazil', novo_df))