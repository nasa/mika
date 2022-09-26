# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 16:41:38 2022

@author: srandrad
"""
import numpy as np
import pandas as pd
import sys
import os
sys.path.append(os.path.join("..", "..", ".."))
from mika.kd.trend_analysis import get_results_info, get_topics_per_doc
from mika.utils import Data
#read in topic model results

#map topics to categories - for each category, find all the topics
#first get set of categories a
categories = ['Hazard', 'UAS', 'Accident', 'Airspace', 
              'Maintenance', 'Mishap Prevention']
safecom_data = Data()
safecom_data.load('topic_model_results/preprocessed_data.csv', preprocessed=True, id_col='Tracking #')
safecom_cats = safecom_data.data_df[categories]
total_cats = set([d for cat in safecom_cats for c in safecom_cats[cat].tolist() for d in str(c).split(", ")])
#get list of documents per category in form {cat:[doc1, doc2, ...]}
docs_per_cat = {cat:[] for cat in total_cats}
for i in range(len(safecom_data.data_df)):
    doc_cats = [c for l in safecom_data.data_df.iloc[i][categories].values for c in str(l).split(", ") if c != 'nan']
    for cat in doc_cats:
        docs_per_cat[cat].append(safecom_data.data_df.at[i, "Tracking #"])
#read in results
results_file = 'topic_model_resultsAug-26-2022/lda_results.xlsx'#'topic_model_results/lda_results.xlsx'
text_field = 'Narrative'
results, _, _, _ = get_results_info(results_file=results_file, results_text_field=None, text_field=text_field, doc_topic_dist_field=None)
#get topics per category
#first get topics per doc
docs = safecom_data.data_df['Tracking #'].tolist()
topics_per_doc, _ = get_topics_per_doc(docs=docs, results=results, results_text_field='Narrative', hazards=[])
topics_per_cat = {cat:[] for cat in total_cats}
for cat in docs_per_cat:
    docs = docs_per_cat[cat]
    topics = []
    topics = list(set([t for doc in docs_per_cat[cat] for t in topics_per_doc[doc]]))
    topics_per_cat[cat] = topics
print(topics_per_cat)
#conclusion -> no clear mapping to categories to topics
print(total_cats)
#df = pd.DataFrame({"Category": list(total_cats),
#                   "Hazard": ['' for c in total_cats]})
#df.to_csv("category_to_hazard.csv")
