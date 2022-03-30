from turtle import title
from xml.dom.minidom import Document
import pandas
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from contextlib import redirect_stdout
from nltk.tokenize import word_tokenize

 
terms = []
keys = []
vec_Dic = {} 
dicti = {}
dummy_list = []

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
 
def filter(documents, rows, cols):
 
    for i in range(rows):
        for j in range(cols):
            if(j == 0):
                keys.append(documents.loc[i].iat[j])
            else:
                dummy_list.append(documents.loc[i].iat[j])
 
                if documents.loc[i].iat[j] not in terms:
                    terms.append(documents.loc[i].iat[j])
 
        copy = dummy_list.copy()
        dicti.update({documents.loc[i].iat[0]: copy})
        dummy_list.clear()
 
 
def bool_representation(dicti, rows, cols):
 
    terms.sort()
 
    for i in (dicti):
        for j in terms: 
            if j in dicti[i]:
                dummy_list.append(1)
            else:
                dummy_list.append(0)
 
        copy = dummy_list.copy()
        vec_Dic.update({i: copy})
        dummy_list.clear()
 
 
def query_Vector(query):
 
    qvect = []
    for i in terms:
        if i in query:
            qvect.append(1)
        else:
            qvect.append(0)
    return qvect
 
 
def prediction(q_Vect):
 
    dictionary = {}
    listi = []
    count = 0

    term_Len = len(terms)
 
    for i in vec_Dic: 
        for t in range(term_Len):
            if(q_Vect[t] == vec_Dic[i][t]):
                count += 1
 
        dictionary.update({i: count})

        count = 0
 
    for i in dictionary:
        listi.append(dictionary[i])
 
    listi = sorted(listi, reverse=True)
 
    ans = ' '
    for count, i in enumerate(listi):
        key = check(dictionary,i)

        if count == 0:
            ans = key
        print(key, count+1)
        dictionary.pop(key)
    print("\n")
        
 
def check(dictionary, val): 
    for key, value in dictionary.items():
        if(val == value):
 
            return key

def remove_duplicates(lista):
    l = []
    for i in lista:
        if i not in l:
            l.append(i)
    l.sort()
    return l

def tokenize(term):
    return word_tokenize(term) 
 
if __name__ == "__main__":
    documents = pandas.read_json('./copy.json')   
    documents['title'] =documents['title'].apply(clean_stopwords).apply(stemming).apply(remove_alpha)
    documents['tokenize'] = documents['title'].apply(tokenize)
    tokenize= []
    for doc in documents.tokenize:
        tokenize = tokenize + doc
    columns = ['title','body', 'tokenize'] + tokenize
    
    df = pandas.DataFrame(columns=columns)
    for i in documents.columns:
        df[i] = documents[i].values
    
    for index,x in enumerate(df.tokenize):
        for y in x:
            if y in df.columns:
                df.loc[index,y] = y
    
    df.fillna("vazio",inplace=True)
    novo_df = df.loc[:,['title'] + tokenize]
    rows = len(novo_df)
    cols = len(novo_df.columns)
 
    filter(novo_df, rows, cols)
 
    bool_representation(dicti, rows, cols) 
    
    query = "dividend"
 
    query = query.split(' ')
    q_Vect = query_Vector(query)
 
    prediction(q_Vect)