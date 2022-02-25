# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 13:19:22 2021

@author: srandrad
"""
import pandas as pd
import os

from module.topic_model_plus_class import Topic_Model_plus


list_of_attributes = ['Narrative']#'narr_public']#, 'corrective_public', 'notes']
extra_cols_doi = ['region', 'agency', 'duplicate_yn', 'completed_yn', 'rep_by_org',
                   'air_number', 'air_type', 'air_model', 'air_manufacturer',
                   'air_owner', 'mission_destination', 'mission_depart', 'mission_hazmat',
                   'mission_special_use', 'mission_pax', 'mission_procurement_other',
                   'mission_procurement', 'mission_type_other', 'mission_type',
                   'event_damage', 'event_injuries', 'event_org', 'event_org_other',
                   'event_state', 'event_location', 'event_time', 'event_date',
                   'public_yn', 'sequence_number', 'fiscal_year', 'unitid', 'safecomid',
                   'id']
extra_cols = ['Agency', 'Region', 'Location', 'Date', 'Date Submitted', 'Tracking #',
              'Mission Type', 'Persons Onboard', 'Departure Point', 'Destination',
              'Special Use', 'Damages', 'Injuries', 'Hazardous Materials', 'Other Mission Type',
              'Type', 'Manufacturer', 'Model', 'Hazard', 'Incident	Management',
              'UAS', 'Accident', 'Airspace', 'Maintenance', 'Mishap Prevention'
              ]
document_id_col = 'Tracking #'#id'
csv_file_name = os.path.join('data','SAFECOM_data.csv')
name = os.path.join('safecom')
#"""
safecom = Topic_Model_plus(list_of_attributes=list_of_attributes, document_id_col=document_id_col, 
                        csv_file=csv_file_name, database_name=name, extra_cols=extra_cols)
num_topics ={'Narrative': 96}
#"""
safecom.prepare_data()
fire_missions = [mission for mission in list(safecom.data_df['Mission Type']) if type(mission) is str and 'fire' in mission.lower()]
safecom.data_df = safecom.data_df.loc[safecom.data_df['Mission Type'].isin(fire_missions)].reset_index(drop=True)
raw_text = safecom.data_df[safecom.list_of_attributes] 
raw_attrs = ['Raw_'+attr for attr in safecom.list_of_attributes]
safecom.data_df[raw_attrs] = raw_text
safecom.preprocess_data()
safecom.save_preprocessed_data()
#"""
"""#Extract preprocessed data
file = os.path.join('results','safecom_topics-Feb-17-2022','preprocessed_data.csv')
safecom = Topic_Model_plus(document_id_col=document_id_col, extra_cols=extra_cols, list_of_attributes=list_of_attributes, database_name=name, combine_cols=False)
safecom.extract_preprocessed_data(file)
"""
safecom.lda(min_cf=1, num_topics=num_topics)
safecom.save_lda_results()
safecom.save_lda_models()
for attr in list_of_attributes:
    safecom.lda_visual(attr)
#"""




#"""#Run hlda
safecom.hlda(levels=3, eta=0.50, min_cf=1, min_df=1)
safecom.save_hlda_models()
safecom.save_hlda_results()
for attr in list_of_attributes:
    safecom.hlda_visual(attr)
#"""

#Run Bertopic
safecom.data_df[safecom.list_of_attributes] = raw_text
safecom.bert_topic()
safecom.save_bert_results()
#get coherence
#coh = ICS.get_bert_coherence(coh_method='c_v')
safecom.save_bert_vis()
safecom.reduce_bert_topics(num=100)
safecom.save_bert_results()
safecom.save_bert_vis()

#filepath = smart_nlp_path+r"/output data/test_safecom_topics-Sep-21-2021"
#safecom.hlda_extract_models(filepath)
#safecom.save_hlda_results()
