# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 10:04:44 2021
Functions for ICS-209-PLUS
@author: srandrad
"""


"""Analyze fire trends:
    - frequency (count incidents 1 per year)
    - days burning (FOD contain doy-fod discovery doy)
    -FSR??
    - acres burned (final acres)
    -average structures destroyed per year"""
    

"""Analyze operational trends:
    - number of assets (total areial sum)-average per year
    - number of crews (total personnel sum) - average per year
    - number of organizations involved -does not seem to be available
    -cost?
    -injury average per year?
    -complex??"""
    
"""Extract Hazards:
    - preprocessing from lda3
    - apply lda or hlda
    - words from topics = hazards
    """
import pandas as pd
from topic_model_plus_class import Topic_Model_plus

list_of_attributes = ["REMARKS", "SIGNIF_EVENTS_SUMMARY", "MAJOR_PROBLEMS"]#, "CY",
                    #"DISCOVERY_DATE", "START_YEAR", "REPORT_DOY", "DISCOVERY_DOY",
                    #"TOTAL_PERSONNEL", "TOTAL_AERIAL"]
document_id_col = "INCIDENT_ID"
file_name = r"C:\Users\srandrad\OneDrive - NASA\Desktop\ICS NLP\209-PLUS\ics209-plus-wildfire\ics209-plus-wildfire\ics209-plus-wf_sitreps_1999to2014.csv"
name = "ICS_TEST_5000_"

ICS = Topic_Model_plus(document_id_col=document_id_col, csv_file=file_name, list_of_attributes=list_of_attributes, name=name)
ICS.prepare_data(dtype=str)
ICS.data_df=ICS.data_df.loc[:5000, :]
ICS.preprocess_data()
#print(ICS.data_df)
#ICS.save_preprocessed_data()
#ICS.extract_preprocessed_data(r"C:\Users\srandrad\smart_stereo\ICS_TEST_full_topics-Feb-25-2021-\preprocessed_data.csv")
ICS.hlda()
ICS.save_hlda_models()
ICS.save_hlda_taxonomy()
ICS.save_hlda_level_n_taxonomy()
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