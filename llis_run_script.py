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
database_name = 'LLIS' # optional, used at beginning of folder for identification
# optional, can use optimize instead
num_topics ={'Lesson(s) Learned':75, 'Driving Event':75, 'Recommendation(s)':75} # for lda only

# data file input
filename = os.path.join('data','useable_LL_combined.csv')

# creating object
tm = Topic_Model_plus(list_of_attributes=list_of_attributes, document_id_col=document_id_col, csv_file=filename, database_name=database_name)

# preparing the data: loading, dropping columns and rows
# parameters: none required, any kwargs for pd.read_csv can be passed
#tm.prepare_data()

#stopwords = ['Ames','CalTech','Goddard','JPL','Laboratory','Langley','Glenn','Armstrong','Johnson','Kennedy','Marshall','Michoud','Headquarters','Plum','Brook','Stennis','Wallops','Appendix','January','February','March','April','May','June','July','August','September','October','November','December','however']

# parameters: domain_stopwords, ngrams=True (used for custom ngrams), ngram_range=3, threshold=15, min_count=5
#tm.preprocess_data(percent=.3, domain_stopwords=stopwords, ngrams=False, quot_correction=True, min_count=3) # min_count should be equivalent to min_cf in tm.lda
#tm.save_preprocessed_data()

filepath = os.path.join('results','LLIS_topics_Dec-03-2021')
tm.extract_preprocessed_data(os.path.join(filepath,'preprocessed_data.csv'))

# perform lda: can pass in any parameter used in tp model
# parameters: optional
#tm.lda(min_df=2, min_cf=4, num_topics=num_topics, training_iterations=1000)
#tm.save_lda_models()
#tm.save_lda_results()

# perform hlda
#tm.hlda(min_df=2, min_cf=4, training_iterations=1000, alpha = .1, eta = 3, gamma = .5) # increasing eta decreases number of topics
#tm.save_hlda_models()
#tm.save_hlda_results()

tm.lda_extract_models(filepath)
tm.hlda_extract_models(filepath)

# label - need to add a way to save these
tm.label_lda_topics(extractor_min_cf=3, extractor_min_df=2, extractor_max_len=5, extractor_max_cand=10000, labeler_min_df=1, labeler_smoothing=1e-2, labeler_mu=0.3, label_top_n=1)
tm.label_hlda_topics(extractor_min_cf=3, extractor_min_df=2, extractor_max_len=5, extractor_max_cand=10000, labeler_min_df=1, labeler_smoothing=1e-2, labeler_mu=0.3, label_top_n=1) # these settings worked at least once: extractor_min_cf=3, extractor_min_df=2, extractor_max_len=5, extractor_max_cand=10000, labeler_min_df=1, labeler_smoothing=1e-2, labeler_mu=0.3, label_top_n=1

# save desired taxonomy
tm.save_mixed_taxonomy(use_labels=True)

# save visuals
#tm.lda_visual('Lesson(s) Learned')
#tm.hlda_visual('Driving Event')
#tm.lda_visual('Recommendation(s)')
