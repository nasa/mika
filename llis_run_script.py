# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 16:40:39 2021

@author: srandrad
"""
import sys
import os
from module.topic_model_plus_class import Topic_Model_plus

# these variables must be defined to create the object
list_of_attributes = ['Lesson(s) Learned','Driving Event','Recommendation(s)']
document_id_col = 'Lesson ID'
csv_file_name = os.path.join('data','train_set_expanded_H.csv')
name = os.path.join('results','test_lda') # optional, used at beginning of folder for identification
# optional, can use optimize instead
num_topics ={'Lesson(s) Learned':5, 'Driving Event':5, 'Recommendation(s)':5}

# creating object
tm = Topic_Model_plus(list_of_attributes=list_of_attributes, document_id_col=document_id_col, csv_file=csv_file_name, name=name)

# preparing the data: loading, dropping columns and rows
# parameters: none required, any kwargs for pd.read_csv can be passed
tm.prepare_data()

# parameters: domain_stopwords, ngrams=True (used for custom ngrams), ngram_range=3, threshold=15, min_count=5
tm.preprocess_data()

# perform lda: can pass in any parameter used in tp model
# parameters: optional
tm.lda(min_cf=1, num_topics=num_topics)
tm.save_lda_results()

tm.hlda()
tm.save_hlda_results()
