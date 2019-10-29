import re
import numpy as np
from tqdm import tqdm_notebook as tqdm
import pathlib
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import MinMaxScaler

def to_ascii(string):
    latin_dict = \
        {'\\u00e1':'a', '\\u00e0':'a', '\\u00e2':'a', '\\u00e3':'a', '\\u00e4':'a', '\\u00c1':'A', '\\u00c0':'A', 
         '\\u00c2':'A', '\\u00c3':'A', '\\u00c4':'A', '\\u00e9':'e', '\\u00e8':'e', '\\u00ea':'e', '\\u00ea':'e', 
         '\\u00c9':'E', '\\u00c8':'E', '\\u00ca':'E', '\\u00cb':'E', '\\u00ed':'i', '\\u00ec':'i', '\\u00ee':'i', 
         '\\u00ef':'i', '\\u00cd':'I', '\\u00cc':'I', '\\u00ce':'I', '\\u00cf':'I', '\\u00f3':'o', '\\u00f2':'o', 
         '\\u00f4':'o', '\\u00f5':'o', '\\u00f6':'o', '\\u00d3':'O', '\\u00d2':'O', '\\u00d4':'O', '\\u00d5':'O', 
         '\\u00d6':'O', '\\u00fa':'u', '\\u00f9':'u', '\\u00fb':'u', '\\u00fc':'u', '\\u00da':'U', '\\u00d9':'U', 
         '\\u00db':'U', '\\u00e7':'c', '\\u00c7':'C', '\\u00f1':'n', '\\u00d1':'N', '\\u0026':'E', '\\u0027':'\'',
         'á':'a', 'à':'a', 'â':'a', 'ã':'a', 'ä':'a', 'Á':'A', 'À':'A', 'Â':'A', 'Ã':'A', 'Ä':'A', 'é':'e', 'è':'e', 
         'ê':'e', 'ê':'e', 'É':'E', 'È':'E', 'Ê':'E', 'Ë':'E', 'í':'i', 'ì':'i', 'î':'i', 'ï':'i', 'Í':'I', 'Ì':'I', 
         'Î':'I', 'Ï':'I', 'ó':'o', 'ò':'o', 'ô':'o', 'õ':'o', 'ö':'o', 'Ó':'O', 'Ò':'O', 'Ô':'O', 'Õ':'O', 'Ö':'O', 
         'ú':'u', 'ù':'u', 'û':'u', 'ü':'u', 'Ú':'U', 'Ù':'U', 'Û':'U', 'ç':'c', 'Ç':'C', 'ñ':'n', 'Ñ':'N', '&':'E'}
    for key in latin_dict:
        string = string.replace(key, latin_dict[key])
    return string

def html_to_np(table):
    result = []
    all_rows = table.find_all('tr')
    for row in all_rows:
        result.append([])
        all_cols = row.find_all('td')
        for col in all_cols:
            col_text = [s for s in col.find_all(text=True)]
            col_text  = ''.join(col_text) # it should not change anything
            result[-1].append(col_text)
    return np.array(result)

def pages_to_matrix(pages):
    X_data, idx_array = [], []
    invalid_count = 0
    for idx, page in enumerate(tqdm(pages)):

        stem  = pathlib.Path(page).stem
        
        if not pathlib.Path(page).exists():
            invalid_count += 1
            continue
        with open(page,"r") as f:
            lines = f.readlines()
            doc   = '\n'.join(lines)

        s = BeautifulSoup(doc, "html.parser")

        head_text, att = '', ''
        if s.head is not None and s.head.title is not None:
            head_text = s.head.title.find_all(text=True)

        if len(s.find_all("table")) > 0:              
            table       = max(s.find_all("table"),key=len)
            np_table    = (html_to_np(table))

            if len(np_table.shape)>1 and  np_table.shape[1] >= 1:
                att = np_table.ravel()#, np_table[:,1]
        X = ' '.join(head_text) + ' '+ ' '.join(att)

        regex = re.compile('[^a-zA-Z ]')
        X     = regex.sub(' ', X)

        X_data.append(X)
        idx_array.append(idx)
    print(f"Invalid Links {invalid_count}")
    return np.array(X_data), np.array(idx_array)

def get_counts(X):
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(X)
    return X_train_counts

def counts_to_tfidf(X_train_counts):
    tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
    X_train_tf = tf_transformer.transform(X_train_counts)
    return X_train_tf

class DenseTransformer(MinMaxScaler):

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        return X.todense()