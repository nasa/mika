# -*- coding: utf-8 -*-
"""
Created on Tue May 25 10:44:54 2021

@author: srandrad
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

document_id_col = "INCIDENT_ID"
extra_cols = ["CY","DISCOVERY_DATE", "START_YEAR", "REPORT_DOY", "DISCOVERY_DOY",
              "TOTAL_PERSONNEL", "TOTAL_AERIAL", "PCT_CONTAINED_COMPLETED"]
file_name = smart_nlp_path+r"\input data\209-PLUS\ics209-plus-wildfire\ics209-plus-wildfire\ics209-plus-wf_sitreps_1999to2014.csv"
name = smart_nlp_path+r"\output data\ICS_"
list_of_attributes = ["Combined Text"]

file = file = smart_nlp_path+r"\input data\ICS_filtered_preprocessed_combined_data.csv"

ICS = Topic_Model_plus(document_id_col=document_id_col, extra_cols=extra_cols, list_of_attributes=list_of_attributes, name=name, combine_cols=False)
ICS.extract_preprocessed_data(file)
preprocessed = ICS.data_df
filtered_text = [word for text in preprocessed['Combined Text'].tolist() for word in text]
print(filtered_text)

list_of_attributes = ["REMARKS", "SIGNIF_EVENTS_SUMMARY", "MAJOR_PROBLEMS"]
no_stopwords = Topic_Model_plus(document_id_col=document_id_col, extra_cols=extra_cols, csv_file=file_name, list_of_attributes=list_of_attributes, name=name, combine_cols=True)
no_stopwords.prepare_data(dtype=str)
ids = preprocessed["INCIDENT_ID"].unique()
no_stopwords.data_df.loc[no_stopwords.data_df['INCIDENT_ID'].isin(ids)].reset_index(drop=True)
no_stopwords.preprocess_data(percent=0.5, ngrams=False, min_count=1)

total_text = [word for text in no_stopwords.data_df['Combined Text'].tolist() for word in text]
print(len(total_text))
filtered_text = [word for text in preprocessed['Combined Text'].tolist() for word in text]
removed_words = [word for word in total_text if word not in filtered_text]
print(len(removed_words))
print(removed_words)