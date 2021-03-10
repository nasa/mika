# -*- coding: utf-8 -*-
"""
@author: hswalsh
This script allows you to assess the spellchecker effect.
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

print('SPELLCHECKER TESTS')

print('Generating topic model object without spellchecker...')
no_spellcheck = Topic_Model_plus(list_of_attributes=list_of_attributes, document_id_col=document_id_col, csv_file=csv_file_name, name=name, combine_cols=True,LLIS=1,spellcheck=0)
no_spellcheck.prepare_data()
no_spellcheck.preprocess_data()
no_spellcheck.hlda()

print('Generating topic model object with spellchecker...')
spellcheck = Topic_Model_plus(list_of_attributes=list_of_attributes, document_id_col=document_id_col, csv_file=csv_file_name, name=name, combine_cols=True,LLIS=1,spellcheck=1)
spellcheck.prepare_data()
spellcheck.preprocess_data()
spellcheck.hlda()

print('Testing difference in coherence...')

no_spellcheck_coherence = no_spellcheck.hlda_coherence['Combined Text']['average']
spellcheck_coherence = spellcheck.hlda_coherence['Combined Text']['average']

print('Coherence without spellcheck: ',no_spellcheck_coherence)
print('Coherence with spellcheck: ',spellcheck_coherence)

# prints a sample of corrected words
