# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 13:19:22 2021

@author: srandrad
"""
import pandas as pd
import os
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
import tomotopy as tp
import sys
import os
sys.path.append(os.path.join("..", "..", ".."))
from mika.kd import Topic_Model_plus
from mika.utils import Data


text_columns = ['Narrative']

id_col = 'Tracking #'
csv_file_name = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir, os.pardir)), 'data','SAFECOM','SAFECOM_data.csv')
name = os.path.join('safecom')
#""" #Preprocess data
safecom_data = Data()
safecom_data.load(csv_file_name, preprocessed=False, id_col=id_col, text_columns=text_columns, name='safecom')

num_topics ={'Narrative': 96}
safecom_data.prepare_data(remove_incomplete_rows=True)
fire_missions = [mission for mission in list(safecom_data.data_df['Mission Type']) if type(mission) is str and 'fire' in mission.lower()]
safecom_data.data_df = safecom_data.data_df.loc[safecom_data.data_df['Mission Type'].isin(fire_missions)].reset_index(drop=True)
safecom_data.doc_ids = safecom_data.data_df[id_col].tolist()
raw_text = safecom_data.data_df[safecom_data.text_columns] 
raw_attrs = ['Raw_'+attr for attr in safecom_data.text_columns]
safecom_data.data_df[raw_attrs] = raw_text
safecom_data.preprocess_data()
safecom_data.save("preprocessed_data.csv")
safecom_tm = Topic_Model_plus(text_columns=text_columns, data=safecom_data)
#"""
"""#Extract preprocessed data
file = os.path.join('topic_model_results','preprocessed_data.csv')
safecom_data = Data()
safecom_data.load(file, preprocessed=True, id_col=id_col, text_columns=text_columns, name='safecom')
safecom_tm = Topic_Model_plus(text_columns=text_columns, data=safecom_data)
#"""
"""#run hdp to get topic numbers
#safecom_tm.database_name = "SAFECOM_hazards_hdp"
safecom_tm.hdp()
for attr in text_columns:
    print(safecom_tm.hdp_models[attr].k)
safecom_tm.save_lda_results()
safecom_tm.save_lda_models()
for attr in text_columns:
    safecom_tm.lda_visual(attr)
"""
#"""
num_topics ={'Narrative': 100}
#safecom_tm.database_name = "SAFECOM_hazards_lda"
safecom_tm.lda(min_cf=1, num_topics=num_topics)
safecom_tm.save_lda_results()
safecom_tm.save_lda_models()
for attr in text_columns:
    safecom_tm.lda_visual(attr)
#"""

"""#Run hlda
safecom_tm.hlda(levels=3, eta=0.50, min_cf=1, min_df=1)
safecom_tm.save_hlda_models()
safecom_tm.save_hlda_results()
for attr in text_columns:
    safecom_tm.hlda_visual(attr)
"""

#"""#Run Bertopic
BERTkwargs={"calculate_probabilities":True, "top_n_words": 20, 'min_topic_size':20}
safecom_tm.data_df[safecom_tm.text_columns] = raw_text
vectorizer_model = CountVectorizer(ngram_range=(1, 3), stop_words="english") #removes stopwords
safecom_tm.bert_topic(count_vectorizor=vectorizer_model, BERTkwargs=BERTkwargs, from_probs=True)
safecom_tm.save_bert_results()
safecom_tm.save_bert_topics_from_probs()
#get coherence
#coh = ICS.get_bert_coherence(coh_method='c_v')
safecom_tm.save_bert_vis()
safecom_tm.reduce_bert_topics(num=100)
safecom_tm.save_bert_results()
safecom_tm.save_bert_vis()
#"""