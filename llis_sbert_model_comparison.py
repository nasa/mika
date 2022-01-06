# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 15:49:25 2022
Compares different non-tuned bert models performance
@author: srandrad
"""

import os
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import time
from tqdm import tqdm
import numpy as np

# load data
llis_filename = os.path.join('data','lessons_learned_2021-12-10.xlsx')
llis_df = pd.read_excel(llis_filename)
requirements_filename = os.path.join('data','functional_performance_requirements_docs.csv')
requirements_df = pd.read_csv(requirements_filename)

# remove nans
llis_df = llis_df.replace(np.nan, "")
# combine sections to create documents
llis_df['corpus'] = llis_df['Abstract'] + llis_df['Lesson(s) Learned'] + llis_df['Recommendation(s)'] + llis_df['Driving Event'] + llis_df['Evidence']

# select query
print('Relevant Requirement: ', requirements_df['Requirement'][0])
query = 'cyber security measures for data and systems' # do not search full requirement - use short phrase
print('Query: ', query)

start_time = time.time() # start timer

#sbert models

cosine_models = ['msmarco-roberta-base-v3', 'msmarco-distilbert-base-v4', 'msmarco-distilbert-base-v3', 
                'msmarco-MiniLM-L-12-v3', 'msmarco-MiniLM-L-6-v3']
dot_models = ['msmarco-distilbert-base-dot-prod-v3','msmarco-roberta-base-ance-firstp', 'msmarco-distilbert-base-tas-b']                 
sbert_models = cosine_models + dot_models
results = {"Result #":[], "Score":[], "Lesson ID":[],
           "Title":[], "LLIS-Designated Topics":[], "Phase":[],
           "LLIS Text":[] }
index = []
for sbert in tqdm(sbert_models, "Iterating sBERT models..."):
    if sbert in dot_models: model_type = "dot"
    else: model_type = "cosine"
    sbert_model =  SentenceTransformer(sbert)
    query_embedding = sbert_model.encode(query)
    corpus_embedding = sbert_model.encode(llis_df['corpus'], convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, corpus_embedding, top_k=5)
    hits = hits[0] # Get the hits for the first query
    for hit in hits:
        index.append((query, model_type, sbert))
        results["Result #"].append(hits.index(hit)+1)
        results["Score"].append(hit['score'])
        results["Lesson ID"].append(llis_df.at[hit['corpus_id'],"Lesson ID"])
        results["Title"].append(llis_df.at[hit['corpus_id'],"Title"])
        results["LLIS-Designated Topics"].append(llis_df.at[hit['corpus_id'],'Topics'])
        results["Phase"].append(llis_df.at[hit['corpus_id'],'From what phase of the program or project was this lesson learned captured?'])
        results["LLIS Text"].append(llis_df.at[hit['corpus_id'],'corpus'])
        
index = pd.MultiIndex.from_tuples(index, names=["Query", "Model Type", "sBERT model"])
results_df = pd.DataFrame(results, index=index)
results_df.to_csv(os.path.join('results','llis_hls_query_sBERT_comparisons.csv'))
print('Total run time: ', (time.time()-start_time)/60," min")