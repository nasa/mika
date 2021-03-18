# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 10:04:44 2021
Functions for ICS-209-PLUS
@author: srandrad
"""


"""Analyze fire trends: see notebook
    - frequency (count incidents 1 per year)
    - days burning (FOD contain doy-fod discovery doy)
    -FSR??
    - acres burned (final acres)
    -average structures destroyed per year"""


"""Analyze operational trends: see notebook
    - number of assets (total areial sum)-average or total per year
    - number of crews (total personnel sum) - average or total per year
    - number of organizations involved -does not seem to be available
    -cost?
    -injury average per year?
    -complex??
    -num of situation reports?? -total or average per incident per year"""
    
"""Extract Hazards:
    - preprocessing from lda3
    - apply lda or hlda
    - words from topics = hazards
    """
import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")

from module.topic_model_plus_class import Topic_Model_plus
from module.stopwords.ICS_stop_words import stop_words

ICS_stop_words = stop_words

list_of_attributes = ["REMARKS", "SIGNIF_EVENTS_SUMMARY", "MAJOR_PROBLEMS"]
document_id_col = "INCIDENT_ID"
extra_cols = ["CY","DISCOVERY_DATE", "START_YEAR", "REPORT_DOY", "DISCOVERY_DOY",
              "TOTAL_PERSONNEL", "TOTAL_AERIAL", "PCT_CONTAINED_COMPLETED"]
file_name = r"C:\Users\srandrad\smart_nlp\input data\209-PLUS\ics209-plus-wildfire\ics209-plus-wildfire\ics209-plus-wf_sitreps_1999to2014.csv"
name = r"\output data\ICS_full"
#list_of_attributes = ["Combined Text"]

#ICS = Topic_Model_plus(document_id_col=document_id_col, extra_cols=extra_cols, csv_file=file_name, list_of_attributes=list_of_attributes, name=name, combine_cols=True)
#ICS.prepare_data(dtype=str)
#print(ICS.data_df)
#ICS.data_df = ICS.data_df.loc[ICS.data_df["CY"]>="2006"]
#ICS.data_df = ICS.data_df.reset_index(drop=True)
#ICS.data_df = ICS.data_df[:5000].reset_index(drop=True)
#print(ICS.data_df)
#ICS.preprocess_data(domain_stopwords = ICS_stop_words)
#print(ICS.data_df)
#ICS.save_preprocessed_data()

r"""

file = r"C:\Users\srandrad\smart_nlp\output data\ICS_fullset_combinedtopics-Mar-05-2021\preprocessed_data.csv"
ICS = Topic_Model_plus(document_id_col=document_id_col, extra_cols=extra_cols, list_of_attributes=list_of_attributes, name=name, combine_cols=True)
ICS.extract_preprocessed_data(file)
ICS.ngrams = "custom"
ICS.data_df = ICS.data_df[:5000].reset_index(drop=True)
#ICS.extract_preprocessed_data(r"C:\Users\srandrad\smart_stereo\ICS_TEST_full_topics-Feb-25-2021-\preprocessed_data.csv")

#ICS.lda_optimization(min_cf=5, max_topics = 200)
num_topics = {attr:40 for attr in list_of_attributes}
print(ICS.data_df)
ICS.lda(min_cf=5, num_topics=num_topics)
ICS.save_lda_coherence()
ICS.save_lda_taxonomy()
#for attr in list_of_attributes:
#    ICS.lda_visual(attr)
ICS.hlda(levels=4, eta=1.0)
ICS.save_hlda_models()
ICS.save_hlda_coherence()
ICS.save_hlda_taxonomy()
ICS.save_hlda_level_n_taxonomy()
"""
"""Analyze hazard trends for each hazard:
    -for each document: 
        -find its most representative topics. for each word, if it is in the topic,
        identify MTTF, store in dictionary with years as keys and lists of MTTF as values. 
        Two versions, one in raw time, one in percent containment
        - go through the data set, if word present, add +1 to frequency dictionary. Frequency
        dictionary has years as keys, frequency as value. Monthly frequecy has month-year as keys,
        frequency as value.
    - what time during operation it occurs (MTTF): report doy - discovery doy
    - frequency -> use this to derive how often it occurs (MTBF)"""
