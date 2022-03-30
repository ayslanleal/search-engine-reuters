from curses.ascii import isalnum
from math import fabs
from mimetypes import init
from re import search
import pandas as pd
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
from _curses import *
from curses.ascii import isalnum
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import warnings


class SearchEngine:

    def __init__(self,dataframe_path):
        if dataframe_path is None:
            raise Exception("Favor colocar caminho valido")

        nltk.download('punkt')
        nltk.download('stopwords')

      
        self.stopwords = set(stopwords.words('english'))
        self.stemmer = SnowballStemmer("english")
        self.__dataframe_path = dataframe_path
        self.files_name = []
        self.vocabulary = set()
        self.tfidf = None
        self.tfidf_tran = None
        self.df = None
        

    def __clean_stopwords(self, instance):
        words = instance.lower().split()
        stops = []
        for i in words:
            if i not in self.stopwords:
                stops.append(i)
        return ' '.join(stops)     

    def  __stemming(self, instance):
        words = instance.lower().split()
        stems = []
        for i in words:
            stems.append(self.stemmer.stem(i))
        return ' '.join(stems) 

    def __remove_alpha(self,instance):
        words = instance.split()
        alpha = []
        for i in words:
            if i.isalnum():
                alpha.append(i)
        return ' '.join(alpha)


    def preprocessing(self):
        df = pd.read_json(self.__dataframe_path)
        df["body"] = df["body"].apply(self.__clean_stopwords).apply(self.__stemming).apply(self.__remove_alpha)
        return df
    

    def text_processing(self,text):
        word = self.__remove_alpha(text)
        word = self.__clean_stopwords(word)
        word = self.__stemming(word)
        return word
    
    
    def tf_idf_process(self):
       
        
        self.df = self.preprocessing()
        for doc in self.df.to_numpy():
            name_file = doc[0]
            self.files_name.append(name_file)
            self.vocabulary.update(doc[1].split())
        

        #ininicialização do tf-idf
        self.tfidf = TfidfVectorizer(vocabulary=list(self.vocabulary))
        self.tfidf = self.tfidf.fit(self.df.body)
        self.tfidf_tran = self.tfidf.transform(self.df.body)
        return self.tfidf
        
    

    def gen_vector(self, tokens):
        
        x = self.tf_idf_process().transform(tokens)
        q = np.zeros((len(self.vocabulary)))
        
        for token in tokens[0].split():
            try:
                ind = self.vocabulary.index(token) 
                q[ind] = x[0, self.tfidf.vocabulary_[token]]   
            except:
                pass
        
        return q

    def cosine_sim(self,a, b):
        cos_sim = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
        return cos_sim

    def search(self, k, query):
        preprocessing = self.text_processing(query)
        query_df = pd.DataFrame(columns=['query_clean'])
        query_df.loc[0,'query_clean'] = preprocessing
        
        d_cosines = []
        
        query_vector = np.array(self.gen_vector(query_df['query_clean']))
        
        for i in self.tfidf_tran.A:
            d_cosines.append(self.cosine_sim(query_vector,i))
        
        out = np.array(d_cosines).argsort()[-k:][::-1]
        d_cosines.sort()
        """
        
        a = pd.DataFrame()
        for i,index in enumerate(out):
            a.loc[i,'index'] = str(index)
            a.loc[i,'Subject'] = self.preprocessing()['title'][index]
        for j,simScore in enumerate(d_cosines[-k:][::-1]):
            a.loc[j,'Score'] = simScore
        """
        return query_vector
        

if __name__ == '__main__':
    caminho = './teste.json'
    df = SearchEngine(caminho)
    warnings.filterwarnings("ignore", category=RuntimeWarning) 
    print(df.search(10, 'oil price'))
    
    
    
    
        

    


