# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 09:10:24 2021

@author: srandrad
"""
import pandas as pd
import os
from module.In_time_risk_matrix_class import  In_Time_Risk_Matrix

hazards = ['Traffic','Command_Transitions', 'Evacuations', 'Inaccurate_Mapping',
                   'Aerial_Grounding', 'Resource_Issues', 'Injuries', 'Cultural_Resources',
                   'Livestock', 'Law_Violations', 'Military_Base', 'Infrastructure',
                   'Extreme_Weather', 'Ecological', 'Hazardous_Terrain', 'Floods','Dry_Weather']
rm = In_Time_Risk_Matrix(num_severity_models=1, num_likelihood_models=2, hazards=hazards)
severity_x_cols = ['TOTAL_AERIAL', 'TOTAL_PERSONNEL', 'WF_FSR', 'DAYS_BURING', 'ACRES','PCT_CONTAINED_COMPLETED', 'Current_total_Injuries',
       'Current_total_Structures_Damages',
       'Current_total_Structures_Destroyed', 'Current_total_Fatalities','Traffic',
       'Command_Transitions', 'Evacuations', 'Inaccurate_Mapping',
       'Aerial_Grounding', 'Resource_Issues', 'Injuries', 'Cultural_Resources',
       'Livestock', 'Law_Violations', 'Military_Base', 'Infrastructure',
       'Extreme_Weather', 'Ecological', 'Hazardous_Terrain', 'Floods',
       'Dry_Weather', 'Incident_region_AICC', 'Incident_region_CA', 'Incident_region_EACC',
       'Incident_region_GBCC', 'Incident_region_HICC', 'Incident_region_NRCC',
       'Incident_region_NWCC', 'Incident_region_RMCC', 'Incident_region_SACC',
       'Incident_region_SWCC']

severity_model = os.path.join(os.path.dirname(os.getcwd()),'smart_nlp','models','severity_model_test.sav')
rm.load_model(model_location=[severity_model], model_type="severity", model_inputs=[severity_x_cols])
text_input = ["Combined_Text"]
meta_inputs = ["TOTAL_PERSONNEL", "TOTAL_AERIAL", "PCT_CONTAINED_COMPLETED",
              "ACRES",  "WF_FSR", "INJURIES", "FATALITIES", "EST_IM_COST_TO_DATE", "STR_DAMAGED",
              "STR_DESTROYED", "NEW_ACRES", "EVACUATION_IN_PROGRESS", 
              "NUM_REPORTS", "DAYS_BURING", 'Incident_region_AICC', 
              'Incident_region_CA', 'Incident_region_EACC','Incident_region_GBCC', 'Incident_region_HICC', 
              'Incident_region_NRCC','Incident_region_NWCC', 'Incident_region_RMCC', 'Incident_region_SACC',
              'Incident_region_SWCC', 'INC_MGMT_ORG_ABBREV_1', 'INC_MGMT_ORG_ABBREV_2','INC_MGMT_ORG_ABBREV_3', 
              'INC_MGMT_ORG_ABBREV_4','INC_MGMT_ORG_ABBREV_5', 'INC_MGMT_ORG_ABBREV_B','INC_MGMT_ORG_ABBREV_C', 
              'INC_MGMT_ORG_ABBREV_D','INC_MGMT_ORG_ABBREV_E', 'INC_MGMT_ORG_ABBREV_F']
"""
#tfidf_model
text_model = os.path.join(os.path.dirname(os.getcwd()),'smart_nlp','models','likelihood_model_1_test.sav')
meta_model = os.path.join(os.path.dirname(os.getcwd()),'smart_nlp','models','likelihood_model_2_test.sav')
rm.load_model(model_location=[text_model, meta_model], model_type="likelihood", model_inputs=[text_input, meta_inputs])
#vectorized
tfidf_model = os.path.join(os.path.dirname(os.getcwd()),'smart_nlp','models','likelihood_tfidf_model_1_test.sav')
rm.load_nlp_model(model_location=[tfidf_model], model_type="likelihood", model_number=[0], model_input=[["Combined_Text"]])
input_reports = pd.read_csv(os.path.join(os.path.dirname(os.getcwd()),'smart_nlp','data','ICS_predictive_sitreps_test.csv')).drop(["Unnamed: 0"], axis=1)
input_reports = input_reports.iloc[40:42][:].reset_index(drop=True)
rm.build_risk_matrix(input_reports, clean=True, vectorize=True, target="Combined_Text", model_type="likelihood", nlp_model_type="tfidf", nlp_model_number=0)
"""
#word2vec_model
"""
text_model = os.path.join(os.path.dirname(os.getcwd()),'smart_nlp','models','word2vec_LSTM_NN')
meta_model = os.path.join(os.path.dirname(os.getcwd()),'smart_nlp','models','likelihood_model_word2vec_2_test.sav')
rm.load_model(model_location=[text_model, meta_model], model_type="likelihood", NN=True, model_inputs=[text_input, meta_inputs])

bigram_model = os.path.join(os.path.dirname(os.getcwd()),'smart_nlp','models','word2vec_bigram_model.sav')
trigram_model = os.path.join(os.path.dirname(os.getcwd()),'smart_nlp','models','word2vec_trigram_model.sav')
vectorization_model = os.path.join(os.path.dirname(os.getcwd()),'smart_nlp','models','word2vec_tokenizer.sav')
model_numbers = ["bigram_0", "trigram_0", "tokenizer_0"]
rm.load_nlp_model(model_location=[bigram_model, trigram_model, vectorization_model], model_type="likelihood", 
                  model_number=model_numbers, model_input=[["Combined_Text"],["Combined_Text"],["Combined_Text"]])
input_reports = pd.read_csv(os.path.join(os.path.dirname(os.getcwd()),'smart_nlp','data','ICS_predictive_sitreps_test.csv')).drop(["Unnamed: 0"], axis=1)
input_reports = input_reports.iloc[40:42][:].reset_index(drop=True)
rm.build_risk_matrix(input_reports, clean=True, vectorize=True, target="Combined_Text", model_type="likelihood", nlp_model_type="word2vec", nlp_model_number=0)
"""
#BERT model
#"""
text_model = os.path.join(os.path.dirname(os.getcwd()),'smart_nlp','models','sbert_likelihood_model1.sav')
meta_model = os.path.join(os.path.dirname(os.getcwd()),'smart_nlp','models','sbert_likelihood_model2.sav')
rm.load_model(model_location=[text_model, meta_model], model_type="likelihood", model_inputs=[text_input, meta_inputs])
#vectorized
bert_model = os.path.join(os.path.dirname(os.getcwd()),'smart_nlp','models','sbert_model.sav')
rm.load_nlp_model(model_location=[bert_model], model_type="likelihood", model_number=[0], model_input=[["Combined_Text"]])
input_reports = pd.read_csv(os.path.join(os.path.dirname(os.getcwd()),'smart_nlp','data','ICS_predictive_sitreps_test.csv')).drop(["Unnamed: 0"], axis=1)
input_reports = input_reports.iloc[40:42][:].reset_index(drop=True)
rm.build_risk_matrix(input_reports, clean=True, vectorize=True, target="Combined_Text", model_type="likelihood", nlp_model_type="sBERT", nlp_model_number=0)
#"""
