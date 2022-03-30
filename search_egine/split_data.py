import pandas as pd
import re
import json

def title_body(text):
    ''' Split \ n to divide title and body'''
    title = re.split(r'\n', text)[0]
    body =  re.split(r'\n', text)[1:]
    return {"title":title,"body":''.join(body)}

def df_to_dict(df):
    lista = [title_body(i) for i in df['text']] 
    return lista

df = pd.read_excel('reutersNLTK.xlsx')
with open('reuters.json', 'w') as f:
    json.dump(df_to_dict(df), f)