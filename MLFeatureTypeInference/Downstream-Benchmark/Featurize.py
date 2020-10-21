#Copyright 2020 Vraj Shah, Arun Kumar
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import pickle
import pandas as pd
import numpy as np
import os
from pandas.api.types import is_numeric_dtype
from collections import Counter,defaultdict
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
import re
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer


def URLProcessor(x):
    y = re.findall(r"[\w']+", x)
    z = ' '.join(y)
    return z

def Featurize(dataDownstream,attribute_names,y_cur):

    all_cols,numeric_cols,categ_cols,ngram_cols,url_cols = pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
    vectorizer = CountVectorizer(ngram_range=(2,2),analyzer='char')
    vec = TfidfVectorizer(encoding = "latin-1", strip_accents = "unicode", stop_words = "english")
    vectorizerWord = CountVectorizer(ngram_range=(2,2))

    for i in range(len(y_cur)):
        curcol = attribute_names[i]
        curdf = dataDownstream[curcol]
        if y_cur[i] == 0:
            numeric_cols = pd.concat([numeric_cols,curdf],axis=1)
        if y_cur[i] == 1:
            tempdf = pd.get_dummies(curdf, columns=[curcol])
            categ_cols = pd.concat([categ_cols,tempdf],axis=1)
    #     elif y_cur[i] == 2:
    #         temp = pd.DataFrame()
    #         temp['month'] = dataDownstream.apply(lambda row: pd.Timestamp(row[curcol]).month, axis=1)
    #         print(temp)
    #         tempdf = pd.get_dummies(temp, columns=['month'])
    #         date_cols = pd.concat([date_cols,tempdf], axis=1, sort=False)
        if y_cur[i] == 3:
            arr = curdf.astype(str).values
#             print(arr)
            X = vec.fit_transform(arr)
            tempdf = pd.DataFrame(X.toarray())
            ngram_cols = pd.concat([ngram_cols,tempdf], axis=1, sort=False)
        if y_cur[i] == 4:
            temp = curdf.apply(lambda x: URLProcessor(x))
#             print(temp)
            curdf1 = dataDownstream[curcol]
            arr = temp.astype(str).values  
#             print(len(arr))
            X = vectorizerWord.fit_transform(arr)
#             print(X)
            tempdf = pd.DataFrame(X.toarray())
            url_cols = pd.concat([url_cols,tempdf], axis=1, sort=False)
        if y_cur[i] in [2,3,5,6,8]:
            arr = curdf.astype(str).values
            X = vectorizer.fit_transform(arr)
            tempdf = pd.DataFrame(X.toarray())
            ngram_cols = pd.concat([ngram_cols,tempdf], axis=1, sort=False)

    all_cols = pd.concat([all_cols,numeric_cols,categ_cols,ngram_cols], axis=1, sort=False)        
#     print(all_cols)
    return all_cols