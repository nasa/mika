# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 16:40:39 2021

@author: srandrad
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
from sys import platform

from module.topic_model_plus_class import Topic_Model_plus

if platform == "darwin":
    sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
    smart_nlp_path = ''
elif platform == "win32":
    sys.path.append('../')
    smart_nlp_path = os.getcwd()
    smart_nlp_path = "\\".join([smart_nlp_path.split("\\")[i] for i in range(0,len(smart_nlp_path.split("\\"))-1)]+["/"])

#print(smart_nlp_path)
#these variables must be defined to create the object
list_of_attributes = ['Lesson(s) Learned','Driving Event','Recommendation(s)']
document_id_col = 'Lesson ID'
csv_file_name = smart_nlp_path+"input data/train_set_expanded_H.csv" 
name = smart_nlp_path+"output data/test_lda" #optional, used at beginning of folder for identification
#optional, can use optimize instead
num_topics ={'Lesson(s) Learned':5, 'Driving Event':5, 'Recommendation(s)':5}

#creating object
test = Topic_Model_plus(list_of_attributes=list_of_attributes, document_id_col=document_id_col, csv_file=csv_file_name, name=name)
#preparing the data: loading, dropping columns and rows
#parameters: none required, any kwargs for pd.read_csv can be passed
#test.prepare_data()
#preprocessing the data: cleaning, lemmatizing, bigrams (optional, pass ngrams=False to skip)
#parameters: domain_stopwords, ngrams=True (used for custom ngrams), ngram_range=3, threshold=15, min_count=5
#test.preprocess_data()

#print(test.data_df)
#optimize lda: needs work, generally outputs the max_topics, can use coherence or loglikelihood
#parameters: optional, can pass max_topics and any kwargs for tp.lda model
#outpus: saves the optimized num of topics in a member variable, need to optimize hyper params alpha and beta as well
#test.lda_optimization(min_cf=1, max_topics = 100)
##perform lda: can pass in any parameter used in tp model
#parameters: optional

#test.lda(min_cf=1, num_topics=num_topics)
#saving various lda results

#test.save_lda_taxonomy()

#test.save_lda_document_topic_distribution()
#test.save_lda_models()
#test.save_lda_coherence()
#LDA visualization using pyLDAvis, saves html link to folder
#for attr in list_of_attributes:
#    test.lda_visual(attr)
#test.hlda(levels=4)
#test.save_hlda_models()
#display_options = {"level 1": 1,
#                   "level 2": 3,
#                   "level 3": 8}
#file_root = r"C:\Users\srandrad\smart_nlp\output data\testtopics-Mar-17-2021"
#for attr in list_of_attributes:
#    test.hlda_display(attr=attr, display_options=display_options, filename=file_root+"\\"+attr+"_hlda_model_object.bin")
r"""
#perform hlda: can pass in any parameter used in tp model
test.hlda()
#saving various hlda results
test.save_hlda_models()
test.save_hlda_taxonomy()
test.save_hlda_coherence()
test.save_hlda_level_n_taxonomy()


test.hlda_extract_models(file_path=r"C:\Users\srandrad\smart_nlp\output data\test_ldatopics-Mar-18-2021")
test.save_hlda_taxonomy()
test.save_hlda_topics()
test.save_hlda_coherence()
test.save_hlda_level_n_taxonomy()
test.save_hlda_document_topic_distribution()

test.lda_extract_models(file_path=r"C:\Users\srandrad\smart_nlp\output data\test_ldatopics-Mar-18-2021")
test.save_lda_document_topic_distribution()
test.save_lda_taxonomy()
test.save_lda_coherence()
"""
