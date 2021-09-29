# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 13:19:22 2021

@author: srandrad
"""
import pandas as pd
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
    smart_nlp_path = "\\".join([smart_nlp_path.split("\\")[i] for i in range(0,len(smart_nlp_path.split("\\"))-1)])
    smart_nlp_path = smart_nlp_path+"\\"

list_of_attributes = ['narr_public']#, 'corrective_public', 'notes']
extra_cols = ['region', 'agency', 'duplicate_yn', 'completed_yn', 'rep_by_org',
                   'air_number', 'air_type', 'air_model', 'air_manufacturer',
                   'air_owner', 'mission_destination', 'mission_depart', 'mission_hazmat',
                   'mission_special_use', 'mission_pax', 'mission_procurement_other',
                   'mission_procurement', 'mission_type_other', 'mission_type',
                   'event_damage', 'event_injuries', 'event_org', 'event_org_other',
                   'event_state', 'event_location', 'event_time', 'event_date',
                   'public_yn', 'sequence_number', 'fiscal_year', 'unitid', 'safecomid',
                   'id']
document_id_col = 'id'
csv_file_name = smart_nlp_path+"input data/safecom-2011-present-NASA.xlsx" 
name = smart_nlp_path+"output data/test_safecom"
"""
test = Topic_Model_plus(list_of_attributes=list_of_attributes, document_id_col=document_id_col, 
                        csv_file=csv_file_name, name=name, extra_cols=extra_cols)
num_topics ={'narr_public': 96}

test.prepare_data(csv=False, sheet_name='safecom-2011-present-NASA')
print(test.data_df)
test.preprocess_data()
test.save_preprocessed_data()
test.lda(min_cf=1, num_topics=num_topics)
test.save_lda_results()
test.save_lda_models()
for attr in list_of_attributes:
    test.lda_visual(attr)
"""

#"""#Extract preprocessed data

file = smart_nlp_path+r"/output data/test_safecom_topics-Sep-21-2021/preprocessed_data.csv"
safecom = Topic_Model_plus(document_id_col=document_id_col, extra_cols=extra_cols, list_of_attributes=list_of_attributes, name=name, combine_cols=False)
safecom.extract_preprocessed_data(file)
#"""


#"""#Run hlda
safecom.hlda(levels=3, eta=0.50, min_cf=1, min_df=1)
safecom.save_hlda_models()
safecom.save_hlda_results()
for attr in list_of_attributes:
    safecom.hlda_visual(attr)
#"""
#filepath = smart_nlp_path+r"/output data/test_safecom_topics-Sep-21-2021"
#safecom.hlda_extract_models(filepath)
#safecom.save_hlda_results()