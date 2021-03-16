# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 16:40:39 2021

@author: srandrad
"""
from topic_model_plus_class import Topic_Model_plus

#these variables must be defined to create the object
list_of_attributes = ['Lesson(s) Learned','Driving Event','Recommendation(s)']
document_id_col = 'Lesson ID'
csv_file_name = "input data/train_set_expanded_H.csv" 
name = "output data/test" #optional, used at beginning of folder for identification
#optional, can use optimize instead
num_topics ={'Lesson(s) Learned':5, 'Driving Event':5, 'Recommendation(s)':5}

#creating object
test = Topic_Model_plus(list_of_attributes=list_of_attributes, document_id_col=document_id_col, csv_file=csv_file_name, name=name)
#preparing the data: loading, dropping columns and rows
#parameters: none required, any kwargs for pd.read_csv can be passed
test.prepare_data()
#preprocessing the data: cleaning, lemmatizing, bigrams (optional, pass ngrams=False to skip)
#parameters: domain_stopwords, ngrams=True (used for custom ngrams), ngram_range=3, threshold=15, min_count=5
test.preprocess_data()

print(test.data_df)
#optimize lda: needs work, generally outputs the max_topics, can use coherence or loglikelihood
#parameters: optional, can pass max_topics and any kwargs for tp.lda model
#outpus: saves the optimized num of topics in a member variable, need to optimize hyper params alpha and beta as well
#test.lda_optimization(min_cf=1, max_topics = 100)
##perform lda: can pass in any parameter used in tp model
#parameters: optional

test.lda(min_cf=1, num_topics=num_topics)
#saving various lda results

#test.save_lda_taxonomy()

#test.save_lda_document_topic_distribution()
#test.save_lda_models()
test.save_lda_coherence()
#LDA visualization using pyLDAvis, saves html link to folder
#for attr in list_of_attributes:
#    test.lda_visual(attr)
"""
#perform hlda: can pass in any parameter used in tp model
test.hlda()
#saving various hlda results
test.save_hlda_models()
test.save_hlda_taxonomy()
test.save_hlda_coherence()
test.save_hlda_level_n_taxonomy()
"""
