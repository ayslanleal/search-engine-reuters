import pandas as pd
from nltk.stem import RSLPStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
import numpy as np
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
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

def cosine_sim(a, b):
    cos_sim = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
    return cos_sim

def gen_vector(tokens,vocabulary, tfidf):    
    q = np.zeros((len(vocabulary)))
    x = tfidf.transform(tokens)
        
    for token in tokens[0].split(','):
        try:
            ind = vocabulary.index(token) 
            q[ind] = x[0, tfidf.vocabulary_[token]]   
        except:
            pass
        
    return q

def string_preprocess(string):
    text = clean_stopwords(string)
    text = stemming(text)
    return text


def search(df,k, query,vocabulary, tfidf,tfidf_transform):
    preprocessing = string_preprocess(query)
    tokens = ",".join(word_tokenize(str(preprocessing)))
    query_df = pd.DataFrame(columns=['query_clean'])
    query_df.loc[0, 'query_clean'] = str(tokens)
    d_cosines = []

    
    query_vector = np.array(gen_vector(query_df['query_clean'],vocabulary,tfidf))
    for i in tfidf_transform.A:
        d_cosines.append(cosine_sim(query_vector, i))

    out = np.array(d_cosines).argsort()[-k:][::-1]
    d_cosines.sort()
    a = pd.DataFrame()
    for i,index in enumerate(out):
        a.loc[i, 'index'] = str(index)
        a.loc[i, 'Title'] = df['title'][index]
    return a


if __name__ == "__main__":
    df = pd.read_json('./teste.json')
    df['body'] = df['body'].apply(clean_stopwords).apply(stemming).apply(remove_alpha)
    df["tokenize"] = [','.join((word_tokenize(entry))) for entry in df.body]
    
    list_english = set()
    for i in df.tokenize:
        list_english.update(i.split(','))
    
    list_english = list(list_english)

    tfidf = TfidfVectorizer(vocabulary=list_english)
    tfidf.fit(df.tokenize)
    tfidf_transform  = tfidf.transform(df.tokenize)
    
    a = search(df,10, "Brazil JP Morgan",list_english,tfidf, tfidf_transform)
    print(a)



    


