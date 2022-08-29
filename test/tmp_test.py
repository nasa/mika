# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 13:40:07 2022

@author: srandrad
"""

import pandas as pd

df = pd.read_excel(r"C:\Users\srandrad\smart_nlp\results\ICS_bertopic_combined_topics_Jun-20-2022\BERTopic_results.xlsx", sheet_name='Combined Text')
df =  pd.read_excel(r"C:\Users\srandrad\smart_nlp\results\ICS_bertopic_combined_topics_Jun-20-2022\BERTopic_results.xlsx",  sheet_name='Combined Text')
df = pd.read_csv(r"C:\Users\srandrad\smart_nlp\results\ICS_bertopic_combined_topics_Jun-20-2022\Combined Text_BERT_topics_modified.csv")
print(df)

def clean_list(df, col):
    #print(col)
    df_col = df[col].tolist()#[1:]
    df_col = [[j.strip("'") for j in i.strip("[]").split(", ") if len(j)>0] for i in df_col]
    return df_col

#best_docs = clean_list(df, 'best documents')
total_docs = clean_list(df, 'documents')
number_of_docs = [int(i) for i in df['number of documents in topic'].tolist()]#[1:]]

for i in range(len(df)):
    #print(i, len(total_docs[i]), number_of_docs[i])
    #print(total_docs[i][-1])
    try:
        assert(len(total_docs[i])==number_of_docs[i])
    except:
        print(i, len(total_docs[i]), number_of_docs[i])
        print(total_docs[i][0])