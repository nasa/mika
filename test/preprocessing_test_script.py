# -*- coding: utf-8 -*-
"""
@author: hswalsh
This script allows you to run only the data preparation and preprocessing portions of the topic modeling code for testing purposes.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")

from topic_model_plus_class import Topic_Model_plus

list_of_attributes = ['Lesson(s) Learned','Driving Event','Recommendation(s)']
document_id_col = 'Lesson ID'
csv_file_name = "input data/train_set_expanded_H.csv" 
name = "output data/test"
num_topics ={'Lesson(s) Learned':5, 'Driving Event':5, 'Recommendation(s)':5}

test = Topic_Model_plus(list_of_attributes=list_of_attributes, document_id_col=document_id_col, csv_file=csv_file_name, name=name, combine_cols=True,quot_correction=True,spellcheck=False,segmentation=False)
test.prepare_data()
test.preprocess_data()

print(test.data_df)
