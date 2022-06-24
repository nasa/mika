# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 16:47:08 2022

@author: srandrad
"""

from module.NER_utils import read_doccano_annots
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn import feature_selection
import sys
import os
import numpy as np

#reading in annotation file
file = r"C:\Users\srandrad\Downloads\llis_risk_factors.jsonl"
df = read_doccano_annots(file,  encoding=True) #encoding=True is required when Sequoia is importing data from Hannah

#formating annotations and text
annotated_texts = []
annotated_risk_factors = []
for i in range(len(df)):
    text = df.at[i, 'text']
    labels = df.at[i, 'label']
    for l in labels:
        annotated_text = text[l[0]:l[1]]
        annotated_risk_factor = l[2]
        annotated_texts.append(annotated_text)
        annotated_risk_factors.append(annotated_risk_factor)
X = annotated_texts

#formatting labels into one-hot encoding
y_raw = annotated_risk_factors
y = pd.DataFrame(np.zeros((len(y_raw), len(set(y_raw)))), columns=set(y_raw)).sort_index(axis=1)
for i in range(len(y_raw)):
    y.at[i, y_raw[i]] = 1

#vectorize text
vectorizer = TfidfVectorizer(strip_accents='unicode', analyzer='word', ngram_range=(1,3), norm='l2', max_features=10000)
vectorizer.fit(X)
X_vec = vectorizer.transform(X)

#chi-squared for significant words
#i.e., find words with statistically significiant difference in frequency between categories
X_names = vectorizer.get_feature_names_out()
p_value_limit = 0.99
dtf_features = pd.DataFrame()
for cat in y.columns:
    chi2, p = feature_selection.chi2(X_vec, y[[cat]])
    dtf_features = dtf_features.append(pd.DataFrame(
                   {"feature":X_names, "score":1-p, "y":cat}))
    dtf_features = dtf_features.sort_values(["y","score"], 
                    ascending=[True,False])
    dtf_features = dtf_features[dtf_features["score"]>p_value_limit]
X_names = dtf_features["feature"].unique().tolist()

#print top words
for cat in y:
    print("# {}:".format(cat))
    print("  . selected features:",
          len(dtf_features[dtf_features["y"]==cat]))
    print("  . top features:", ",".join(dtf_features[dtf_features["y"]==cat]["feature"].values[:10]))
    print(" ")
    
#save lexicons
#reformat: each column is a list of words for the category
data_dict = {cat:[] for cat in y}
for cat in y:
    cat_words = dtf_features[dtf_features["y"]==cat]
    data_dict[cat] = cat_words["feature"].tolist()
# make lists of equal length
max_list_len = max([len(data_dict[cat]) for cat in data_dict])
for cat in data_dict:
    if len(data_dict[cat]) < max_list_len:
        for i in range(len(data_dict[cat]), max_list_len):
            data_dict[cat].append("")
#save all words per category
lexicon = pd.DataFrame(data_dict)
lexicon.to_csv("results/Risk_Factor_Lexicons.csv")    