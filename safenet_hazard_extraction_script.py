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
from module.topic_model_plus_class import Topic_Model_plus

list_of_attributes = ["Narrative"]
csv_file_name = os.path.join("data","SAFENET","SAFENET_1999_2021.csv")
name = os.path.join('safenet')
extra_cols = ["Iteration", "Event Start Date", "Event Stop Date", "Incident Name",
              "Incident Number", "Event State", "Event Jurisdiction", "Event Local Unit",
              "Event Position Title", "Event Task", "Event Management Level", "Event Resources", 
              "Event Incident Type", "Event Incident Activity", "Event Incident Stage", "Contributing Factors",
              "Human Factors", "Other Factors", "Narrative", "Immediate Action Taken",  "SAFENET Create Date"
              ]
document_id_col = "ID"
safenet = Topic_Model_plus(list_of_attributes=list_of_attributes, document_id_col=document_id_col, 
                        csv_file=csv_file_name, database_name=name, extra_cols=extra_cols)
safenet.prepare_data()
raw_text = safenet.data_df[safenet.list_of_attributes] 
raw_attrs = ['Raw_'+attr for attr in safenet.list_of_attributes]
safenet.data_df[raw_attrs] = raw_text
safenet.preprocess_data()
safenet.save_preprocessed_data()
#run lda
num_topics ={'Narrative': 50}
safenet.lda(min_cf=1, num_topics=num_topics)
safenet.save_lda_results()
safenet.save_lda_models()
for attr in list_of_attributes:
    safenet.lda_visual(attr)

#"""#Run Bertopic
safenet.data_df[safenet.list_of_attributes] = raw_text
vectorizer_model = CountVectorizer(ngram_range=(1, 2), stop_words="english") #removes stopwords
#hdbscan_model = HDBSCAN(min_cluster_size=3, min_samples=3) #allows for smaller topic sizes/prevents docs with no topics
safenet.bert_topic(count_vectorizor=vectorizer_model)#, hdbscan=hdbscan_model)
safenet.save_bert_results()
#get coherence
#coh = ICS.get_bert_coherence(coh_method='c_v')
safenet.save_bert_vis()
safenet.reduce_bert_topics(num=100)
safenet.save_bert_results()
safenet.save_bert_vis()
#"""