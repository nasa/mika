# -*- coding: utf-8 -*-
"""
@author: hswalsh
This script allows you to assess the spellchecker and segmentation effects.
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

print('Generating topic model object without spellchecker or segmentation...')
no_spellcheck = Topic_Model_plus(list_of_attributes=list_of_attributes, document_id_col=document_id_col, csv_file=csv_file_name, name=name, combine_cols=True)
no_spellcheck.prepare_data()
no_spellcheck.preprocess_data(quot_correction=True,spellcheck=False,segmentation=False)
no_spellcheck.hlda()

print('Generating topic model object with spellchecker...')
spellcheck = Topic_Model_plus(list_of_attributes=list_of_attributes, document_id_col=document_id_col, csv_file=csv_file_name, name=name, combine_cols=True)
spellcheck.prepare_data()
spellcheck.preprocess_data(quot_correction=True,spellcheck=True,segmentation=False)
spellcheck.hlda()

print('Generating topic model object with segmentation...')
segmentation = Topic_Model_plus(list_of_attributes=list_of_attributes, document_id_col=document_id_col, csv_file=csv_file_name, name=name, combine_cols=True)
segmentation.prepare_data()
segmentation.preprocess_data(quot_correction=True,spellcheck=False,segmentation=True)
segmentation.hlda()

print('Generating topic model object with both...')
both = Topic_Model_plus(list_of_attributes=list_of_attributes, document_id_col=document_id_col, csv_file=csv_file_name, name=name, combine_cols=True)
both.prepare_data()
both.preprocess_data(quot_correction=True,spellcheck=True,segmentation=True)
both.hlda()

print('Testing difference in coherence...')

no_spellcheck_coherence = no_spellcheck.hlda_coherence['Combined Text']['average']
spellcheck_coherence = spellcheck.hlda_coherence['Combined Text']['average']
segmentation_coherence = segmentation.hlda_coherence['Combined Text']['average']
both_coherence = both.hlda_coherence['Combined Text']['average']

print('Coherence without spellcheck or segmentation: ',no_spellcheck_coherence)
print('Coherence with spellcheck only: ',spellcheck_coherence)
print('Coherence with segmentation only: ',segmentation_coherence)
print('Coherence with spellcheck and segmentation: ',both_coherence)

print('Sample of spellcheck corrected words...')
print(spellcheck.correction_list[0:10])

print('Sample of segmentation corrected words...')
print(segmentation.correction_list[0:10])
