# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 15:40:54 2022

@author: srandrad
"""

import os
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import time
from tqdm import tqdm
import numpy as np

# load data
llis_filename = os.path.join('data','LLIS','lessons_learned_2021-12-10.xlsx')
llis_df = pd.read_excel(llis_filename)
requirements_filename = os.path.join('data','functional_performance_requirements_docs.csv')
requirements_df = pd.read_csv(requirements_filename)

# remove nans
llis_df = llis_df.replace(np.nan, "")

# combine sections to create documents
llis_df['corpus'] = llis_df['Abstract'] + llis_df['Lesson(s) Learned'] + llis_df['Recommendation(s)'] + llis_df['Driving Event'] + llis_df['Evidence']
llis_df['section'] = ["n/a" for i in range(len(llis_df))]

sections = ['Abstract','Lesson(s) Learned','Recommendation(s)','Driving Event','Evidence']
cols_from_llis = ["Lesson ID", "Title", 'Topics','From what phase of the program or project was this lesson learned captured?']
llis_paragraphs_df = pd.DataFrame(columns = ['corpus', 'section'] + cols_from_llis)

for i in range(len(llis_df)):
    for section in sections:
        row = llis_df.iloc[i][cols_from_llis]
        row['corpus'] = llis_df.at[i, section]
        row['section'] = section
        llis_paragraphs_df = llis_paragraphs_df.append(row, ignore_index=True)
        
llis_dfs = {'section paragraphs':llis_paragraphs_df, 'full lesson':llis_df}

# select query
print('Relevant Requirement: ', requirements_df['Requirement'][0])
query = 'cyber security measures for data and systems' # do not search full requirement - use short phrase
print('Query: ', query)

start_time = time.time() # start timer


results = {"Result #":[], "Score":[], "Lesson ID":[],
           "Title":[], "Section":[], "LLIS-Designated Topics":[], "Phase":[],
           "LLIS Text":[],}

doc_types = ['section paragraphs','full lesson']
index = []
sbert = 'msmarco-roberta-base-v3'
sbert_model = SentenceTransformer('msmarco-roberta-base-v3')
for doc_type in tqdm(doc_types, "Iterating document types..."):
    query_embedding = sbert_model.encode(query)
    print(llis_dfs[doc_type]['corpus'])
    corpus_embedding = sbert_model.encode(llis_dfs[doc_type]['corpus'], convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, corpus_embedding, top_k=5)
    hits = hits[0] # Get the hits for the first query
    for hit in hits:
        index.append((query, doc_type, sbert))
        results["Result #"].append(hits.index(hit)+1)
        results["Score"].append(hit['score'])
        results["Lesson ID"].append(llis_dfs[doc_type].at[hit['corpus_id'],"Lesson ID"])
        results["Title"].append(llis_dfs[doc_type].at[hit['corpus_id'],"Title"])
        results["LLIS-Designated Topics"].append(llis_dfs[doc_type].at[hit['corpus_id'],'Topics'])
        results["Phase"].append(llis_dfs[doc_type].at[hit['corpus_id'],'From what phase of the program or project was this lesson learned captured?'])
        results["LLIS Text"].append(llis_dfs[doc_type].at[hit['corpus_id'],'corpus'])
        results["Section"].append(llis_dfs[doc_type].at[hit['corpus_id'],'section'])
        
index = pd.MultiIndex.from_tuples(index, names=["Query", "Document Type", "sBERT model"])
results_df = pd.DataFrame(results, index=index)
results_df.to_csv(os.path.join('results','llis_hls_query_doctype_comparisons.csv'))
print('Total run time: ', (time.time()-start_time)/60," min")