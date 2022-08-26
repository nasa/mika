# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 16:47:08 2022
Updates: 
    - annotated only text gives the V1 results - chi-squared is calculated 
    from only risk factor annotated text. This gives words unique to each risk factor.
    - total text should give different results - chi-squared is caluclated from
    the total text (both annotated and unannotated). This should still give unique words.
    Note that this may decrease or increase words - not sure which
    - pval can be varied by changing pval variable. Default is 0.99, should try 0.95 and 0.9
    these values should result in more words per risk factor
@author: srandrad
"""
import sys
import os
sys.path.append(os.path.join("..", ".."))
from mika.kd.NER import read_doccano_annots
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn import feature_selection
from sklearn.utils.extmath import safe_sparse_dot

import numpy as np

#reading in annotation file
file = r"C:\Users\srandrad\Downloads\LLIS_150_no_extra.jsonl"
df = read_doccano_annots(file,  encoding=True) #encoding=True is required when Sequoia is importing data from Hannah

def format_y_labels(y_raw):
    y = pd.DataFrame(np.zeros((len(y_raw), len(set(y_raw)))), columns=set(y_raw)).sort_index(axis=1)
    for i in range(len(y_raw)):
        y.at[i, y_raw[i]] = 1
    return y

def vectorize_text(X):
    vectorizer = TfidfVectorizer(strip_accents='unicode', analyzer='word', ngram_range=(1,3), norm='l2', max_features=10000)
    vectorizer.fit(X)
    X_vec = vectorizer.transform(X)
    return X_vec, vectorizer

def get_significant_words(vectorizer, X_vec, y, p_value_limit=0.99):
    X_names = vectorizer.get_feature_names_out()
    y_array = y.to_numpy()
    p_value_limit = 0.99 #Change this to vary p-value, try 0.95 and 0.9
    dtf_features = pd.DataFrame()
    for cat in y.columns:
        chi2, p = feature_selection.chi2(X_vec, y[[cat]])
        y_array = y[[cat]].to_numpy()
        observed = safe_sparse_dot(y_array.T, X_vec)
        feature_count = X_vec.sum(axis=0).reshape(1, -1)
        class_prob = y_array.mean(axis=0).reshape(1, -1)
        expected = np.dot(class_prob.T, feature_count)
        expected = [expected[0,i] for i in range(expected.shape[1])]
        dtf_features = dtf_features.append(pd.DataFrame(
                       {"feature":X_names, "score":1-p, "y":cat,
                        "expected": expected, 
                        "observed": observed[0]}))
        dtf_features = dtf_features.sort_values(["y","score"], 
                        ascending=[True,False])
        # must have a signficant p-value and observed>expected 
        dtf_features = dtf_features[(dtf_features["score"]>p_value_limit) & (dtf_features["observed"]>dtf_features["expected"])]
    X_names = dtf_features["feature"].unique().tolist() #unnecessary?
    return dtf_features

def print_results(y, dtf_features):
    #print top words
    print("Using only annotated text")
    for cat in y:
        print("# {}:".format(cat))
        print("  . selected features:",
              len(dtf_features[dtf_features["y"]==cat]))
        print("  . top features:", ",".join(dtf_features[dtf_features["y"]==cat]["feature"].values[:10]))
        print(" ")
    
def save_lexicons(y, dtf_features, p_value_limit, total_or_annotated):
    #save lexicons
    #reformat: each column is a list of words for the category
    data_dict = {cat:[] for cat in y if cat != 'None'}
    for cat in y:
        if cat != 'None':
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
    lexicon.to_csv("Risk_Factor_Lexicons_"+total_or_annotated+"_"+str(np.round(1-p_value_limit,2))+".csv")    

#formating annotations and text
annotated_texts = []
annotated_risk_factors = []
non_annotated_texts = []
non_annotated_labels = []
for i in range(len(df)):
    text = df.at[i, 'text']
    labels = df.at[i, 'label']
    prev_label_loc = 0
    for l in labels:
        non_annotated_text = text[prev_label_loc:l[0]]
        if len(non_annotated_text) > 1:
            non_annotated_texts.append(non_annotated_text)
            non_annotated_labels.append("None")
        prev_label_loc = l[1]+1 #sets new start of nonannotated text
        annotated_text = text[l[0]:l[1]]
        annotated_risk_factor = l[2]
        annotated_texts.append(annotated_text)
        annotated_risk_factors.append(annotated_risk_factor)
    non_annotated_text = text[prev_label_loc:-1] #gets the remainder of unlabeled text
    non_annotated_texts.append(non_annotated_text)
    non_annotated_labels.append("None")

X_annotated_only = annotated_texts #for just annotated text - i.e. words unique between risk factors
X_total_text = annotated_texts + non_annotated_texts #for total text
#formatting labels into one-hot encoding
y_annotated_raw = annotated_risk_factors
y_annotated_only = format_y_labels(y_annotated_raw)
#for total text
y_total_raw = y_annotated_raw + non_annotated_labels
y_total = format_y_labels(y_total_raw)
#vectorize text
X_annotated_only_vec, vectorizer_annotated_only = vectorize_text(X_annotated_only)
#for total text
X_total_vec, vectorizer_total = vectorize_text(X_total_text)
#chi-squared for significant words
#i.e., find words with statistically significiant difference in frequency between categories
p_val = 0.99 #Change P-val here
dtf_features_annotated_only = get_significant_words(vectorizer_annotated_only, X_annotated_only_vec, y_annotated_only, p_value_limit=p_val)
dtf_features_total = get_significant_words(vectorizer_total, X_total_vec, y_total, p_value_limit=p_val)
#print results
print("Using only annotated text: \n")
print_results(y_annotated_only, dtf_features_annotated_only)
print("Using total text: \n")
print_results(y_total, dtf_features_total)
#save lexicons
save_lexicons(y_annotated_only, dtf_features_annotated_only, p_val, total_or_annotated="annotated_only")
save_lexicons(y_total, dtf_features_total, p_val, total_or_annotated="total")