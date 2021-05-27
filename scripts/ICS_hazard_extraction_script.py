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

import sys
import os

from sys import platform
if platform == "darwin":
    #sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
    smart_nlp_path = ''
elif platform == "win32":
    sys.path.append('../')
    smart_nlp_path = os.getcwd()
    smart_nlp_path = "\\".join([smart_nlp_path.split("\\")[i] for i in range(0,len(smart_nlp_path.split("\\"))-1)])
    
from module.topic_model_plus_class import Topic_Model_plus
ICS_stop_words = stop_words

import pandas as pd 

extra_cols = ["CY","DISCOVERY_DATE", "START_YEAR", "REPORT_DOY", "DISCOVERY_DOY",
              "TOTAL_PERSONNEL", "TOTAL_AERIAL", "PCT_CONTAINED_COMPLETED"]

"""#Preprocess data
list_of_attributes = ["REMARKS", "SIGNIF_EVENTS_SUMMARY", "MAJOR_PROBLEMS"]
document_id_col = "INCIDENT_ID"
extra_cols = ["CY","DISCOVERY_DATE", "START_YEAR", "REPORT_DOY", "DISCOVERY_DOY",
              "TOTAL_PERSONNEL", "TOTAL_AERIAL", "PCT_CONTAINED_COMPLETED"]
file_name = smart_nlp_path+r"\input data\209-PLUS\ics209-plus-wildfire\ics209-plus-wildfire\ics209-plus-wf_sitreps_1999to2014.csv"
name = smart_nlp_path+r"\output data\ICS_"
ICS = Topic_Model_plus(document_id_col=document_id_col, extra_cols=extra_cols, csv_file=file_name, list_of_attributes=list_of_attributes, name=name, combine_cols=True, create_ids=True)
ICS.prepare_data(dtype=str)
#use filtered reports
file = smart_nlp_path+r"\input data\ICS_filtered_preprocessed_combined_data.csv"
filtered_df = pd.read_csv(file)
filtered_ids = filtered_df['INCIDENT_ID'].unique()
ICS.data_df = ICS.data_df.loc[ICS.data_df['INCIDENT_ID'].isin(filtered_ids)].reset_index(drop=True)
ICS.preprocess_data(domain_stopwords = ICS_stop_words, percent=0.5, ngrams=False, min_count=1)
ICS.save_preprocessed_data()
"""

#"""#Extract preprocessed data
list_of_attributes = ["Combined Text"]
name =  smart_nlp_path+r"\output data\ICS_"
file = smart_nlp_path+r"\input data\ICS_filtered_preprocessed_combined_data.csv"
filtered_df = pd.read_csv(file)
filtered_ids = filtered_df['INCIDENT_ID'].unique()
document_id_col = "Unique IDs"
ICS = Topic_Model_plus(document_id_col=document_id_col, extra_cols=extra_cols, list_of_attributes=list_of_attributes, name=name, combine_cols=False)
ICS.extract_preprocessed_data(file)
#"""

#"""#Run topic modeling
list_of_attributes = ["Combined Text"]
#ICS.lda_optimization(min_cf=5, max_topics = 200)
num_topics = {attr:160 for attr in list_of_attributes}
ICS.lda(min_cf=1, min_df=1, num_topics=num_topics, alpha=1, eta=0.0001)
ICS.save_lda_results()
ICS.save_lda_models()
for attr in list_of_attributes:
    ICS.lda_visual(attr)
ICS.hlda(levels=3, eta=0.50, min_cf=1, min_df=1)
ICS.save_hlda_models()
ICS.save_hlda_results()
for attr in list_of_attributes:
    ICS.hlda_visual(attr)
#"""

"""#Run Results from extracted models
list_of_attributes = ["Combined Text"]
document_id_col = "Unique IDs"
ICS = Topic_Model_plus(document_id_col=document_id_col, extra_cols=extra_cols, list_of_attributes=list_of_attributes, combine_cols=False)
ICS.combine_cols = True
filepath = smart_nlp_path+r"\output data\ICS__combined_topics-May-26-2021"
ICS.hlda_extract_models(filepath)
ICS.save_hlda_results()
"""
