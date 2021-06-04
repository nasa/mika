# -*- coding: utf-8 -*-
"""
Created on Tue May 25 10:44:54 2021

@author: srandrad
""" 
import pandas as pd 

import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../..")

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
    smart_nlp_path = "\\".join([smart_nlp_path.split("\\")[i] for i in range(0,len(smart_nlp_path.split("\\"))-2)])

document_id_col = "INCIDENT_ID"
extra_cols = ["CY","DISCOVERY_DATE", "START_YEAR", "REPORT_DOY", "DISCOVERY_DOY",
              "TOTAL_PERSONNEL", "TOTAL_AERIAL", "PCT_CONTAINED_COMPLETED"]
file_name = smart_nlp_path+r"\input data\209-PLUS\ics209-plus-wildfire\ics209-plus-wildfire\ics209-plus-wf_sitreps_1999to2014.csv"
name = smart_nlp_path+r"\output data\ICS_"
list_of_attributes = ["Combined Text"]

#file = smart_nlp_path+r"\input data\ICS_filtered_preprocessed_combined_data1.csv"
file = smart_nlp_path+r"\output data\ICS_0_combined_topics-Jun-03-2021/preprocessed_data_combined_text.csv"

ICS = Topic_Model_plus(document_id_col=document_id_col, extra_cols=extra_cols, list_of_attributes=list_of_attributes, name=name, combine_cols=False)
ICS.extract_preprocessed_data(file)
preprocessed = ICS.data_df
filtered_text = [word for text in preprocessed['Combined Text'].tolist() for word in text]
"""
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
"""
hazard_file = smart_nlp_path+r"\output data\hazard_interpretation_v2.xlsx"
hazard_info = pd.read_excel(hazard_file, sheet_name=['Hazard-focused'])
hazard_words = []
for i in range(len(hazard_info['Hazard-focused'])):
    hazard_subject_words = hazard_info['Hazard-focused'].iloc[i]['Hazard Noun/Subject']
    hazard_subject_words = hazard_subject_words.split(", ")
    hazard_action_words = hazard_info['Hazard-focused'].iloc[i]['Action/Descriptor']
    hazard_action_words = hazard_action_words.split(", ")
    words = hazard_subject_words + hazard_action_words
    hazard_words = hazard_words + words
#print(hazard_words)
hazard_stopwords = [word for word in hazard_words if word in stop_words]
print(hazard_stopwords)

unused_hazard_words = [word for word in hazard_words if word not in filtered_text]
print(unused_hazard_words)