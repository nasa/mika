# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 16:40:39 2021

@author: srandrad
"""
import sys
import os
from mika.kd import Topic_Model_plus
from mika.utils import Data

# these variables must be defined to create the object
text_columns = ['Lesson(s) Learned','Driving Event','Recommendation(s)']
document_id_col = 'Lesson ID'
database_name = 'LLIS' # optional, used at beginning of folder for identification
# optional, can use optimize instead
num_topics ={'Lesson(s) Learned':75, 'Driving Event':75, 'Recommendation(s)':75} # for lda only

# data file input
filename = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir, os.pardir)), 'data','LLIS','useable_LL_combined.csv')
#for raw data
#llis_data = Data()
#llis_data.load(filename, preprocessed=False, text_columns=text_columns, id_col=document_id_col, name=database_name)

# preparing the data: loading, dropping columns and rows
# parameters: none required, any kwargs for pd.read_csv can be passed
#tm.prepare_data()

#stopwords = ['Ames','CalTech','Goddard','JPL','Laboratory','Langley','Glenn','Armstrong','Johnson','Kennedy','Marshall','Michoud','Headquarters','Plum','Brook','Stennis','Wallops','Appendix','January','February','March','April','May','June','July','August','September','October','November','December','however']

# parameters: domain_stopwords, ngrams=True (used for custom ngrams), ngram_range=3, threshold=15, min_count=5
#llis_.preprocess_data(percent=.3, domain_stopwords=stopwords, ngrams=False, quot_correction=True, min_count=3) # min_count should be equivalent to min_cf in tm.lda
#llis_data.save()
llis_data = Data()
filepath = os.path.join('LLIS_topics_Dec-03-2021')
llis_data.load(os.path.join(filepath, 'preprocessed_data.csv'), preprocessed=True, text_columns=text_columns, id_col=document_id_col, name=database_name)

# creating object
tm = Topic_Model_plus(text_columns=text_columns, data=llis_data)
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
