# -*- coding: utf-8 -*-
"""
Created on Tue May  3 15:11:12 2022

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

text_columns = ["Narrative"]
csv_file_name = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir, os.pardir)),"data","SAFENET","SAFENET_1999_2021.csv")
name = os.path.join('safenet')
extra_cols = ["Iteration", "Event Start Date", "Event Stop Date", "Incident Name",
              "Incident Number", "Event State", "Event Jurisdiction", "Event Local Unit",
              "Event Position Title", "Event Task", "Event Management Level", "Event Resources", 
              "Event Incident Type", "Event Incident Activity", "Event Incident Stage", "Contributing Factors",
              "Human Factors", "Other Factors", "Narrative", "Immediate Action Taken",  "SAFENET Create Date"
              ]
document_id_col = "ID"
safenet_data = Data()
safenet_data.load(csv_file_name, text_columns=text_columns, id_col=document_id_col,name=name)
safenet_data.prepare_data()
raw_text = safenet_data.data_df[safenet_data.text_columns] 
raw_attrs = ['Raw_'+attr for attr in safenet_data.text_columns]
safenet_data.data_df[raw_attrs] = raw_text
safenet_data.preprocess_data()
safenet_data.save()
safenet_tm = Topic_Model_plus(text_columns=text_columns, data=safenet_data)
#run lda
num_topics ={'Narrative': 50}
safenet_tm.lda(min_cf=1, num_topics=num_topics)
safenet_tm.save_lda_results()
safenet_tm.save_lda_models()
for attr in text_columns:
    safenet_tm.lda_visual(attr)

#"""#Run Bertopic
safenet_tm.data_df[safenet_tm.text_columns] = raw_text
vectorizer_model = CountVectorizer(ngram_range=(1, 2), stop_words="english") #removes stopwords
#hdbscan_model = HDBSCAN(min_cluster_size=3, min_samples=3) #allows for smaller topic sizes/prevents docs with no topics
safenet_tm.bert_topic(count_vectorizor=vectorizer_model)#, hdbscan=hdbscan_model)
safenet_tm.save_bert_results()
#get coherence
#coh = ICS.get_bert_coherence(coh_method='c_v')
safenet_tm.save_bert_vis()
safenet_tm.reduce_bert_topics(num=100)
safenet_tm.save_bert_results()
safenet_tm.save_bert_vis()
#"""