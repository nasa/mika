# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 09:10:24 2021

@author: srandrad
"""
import pandas as pd
import os
from module.In_time_risk_matrix_class import  In_Time_Risk_Matrix
from module.trend_analysis_functions import calc_severity, calc_metrics
from module.topic_model_plus_class import Topic_Model_plus

hazards = ['Traffic','Command_Transitions', 'Evacuations', 'Inaccurate_Mapping',
                   'Aerial_Grounding', 'Resource_Issues', 'Injuries', 'Cultural_Resources',
                   'Livestock', 'Law_Violations', 'Military_Base', 'Infrastructure',
                   'Extreme_Weather', 'Ecological', 'Hazardous_Terrain', 'Floods','Dry_Weather']
rm = In_Time_Risk_Matrix(num_severity_models=1, num_likelihood_models=1, hazards=hazards)
severity_x_cols = ['TOTAL_AERIAL', 'TOTAL_PERSONNEL', 'WF_FSR', 'DAYS_BURING', 'ACRES','PCT_CONTAINED_COMPLETED',
      'INJURIES','FATALITIES', 'STR_DESTROYED','STR_DAMAGED','Traffic',
       'Command_Transitions', 'Evacuations', 'Inaccurate_Mapping',
       'Aerial_Grounding', 'Resource_Issues', 'Injuries', 'Cultural_Resources',
       'Livestock', 'Law_Violations', 'Military_Base', 'Infrastructure',
       'Extreme_Weather', 'Ecological', 'Hazardous_Terrain', 'Floods',
       'Dry_Weather', 'Incident_region_AICC', 'Incident_region_CA', 'Incident_region_EACC',
       'Incident_region_GBCC', 'Incident_region_HICC', 'Incident_region_NRCC',
       'Incident_region_NWCC', 'Incident_region_RMCC', 'Incident_region_SACC',
       'Incident_region_SWCC']

severity_model = os.path.join(os.path.dirname(os.getcwd()),'smart_nlp','models','severity_model_xGB.sav')
rm.load_model(model_location=[severity_model], model_type="severity", model_inputs=[severity_x_cols])
text_input = ['Raw_Combined_Text']
meta_inputs = ["TOTAL_PERSONNEL", "TOTAL_AERIAL", "PCT_CONTAINED_COMPLETED",
              "ACRES",  "WF_FSR", "INJURIES", "FATALITIES", "EST_IM_COST_TO_DATE", "STR_DAMAGED",
              "STR_DESTROYED", "NEW_ACRES", "EVACUATION_IN_PROGRESS", 
              "NUM_REPORTS", "DAYS_BURING", 'Incident_region_AICC', 
              'Incident_region_CA', 'Incident_region_EACC','Incident_region_GBCC', 'Incident_region_HICC', 
              'Incident_region_NRCC','Incident_region_NWCC', 'Incident_region_RMCC', 'Incident_region_SACC',
              'Incident_region_SWCC', 'INC_MGMT_ORG_ABBREV_1', 'INC_MGMT_ORG_ABBREV_2','INC_MGMT_ORG_ABBREV_3', 
              'INC_MGMT_ORG_ABBREV_4','INC_MGMT_ORG_ABBREV_5', 'INC_MGMT_ORG_ABBREV_B','INC_MGMT_ORG_ABBREV_C', 
              'INC_MGMT_ORG_ABBREV_D','INC_MGMT_ORG_ABBREV_E', 'INC_MGMT_ORG_ABBREV_F']
input_reports = pd.read_csv(os.path.join(os.path.dirname(os.getcwd()),'smart_nlp','data','ICS_val_sitreps_preprocessed.csv')).drop(["Unnamed: 0"], axis=1)
input_reports = input_reports.loc[input_reports['INCIDENT_ID']=='2012_CA-NEU-15060_ROBBERS'].reset_index(drop=True)
#print(input_reports)
#input_reports = input_reports.drop([i for i in range(len(input_reports)) if i not in [0,9]]).reset_index(drop=True) #only the 1st and 10th reports
#CB model
model_location = os.path.join(os.path.dirname(os.getcwd()),'smart_nlp','models','hazard_model_CB.sav')
model_inputs = [col for col in input_reports.columns if col not in hazards+['INCIDENT_ID', 'Unique_IDs']]
rm.load_model(model_location=[model_location], model_type="likelihood", model_inputs=[model_inputs],NN=False)
rm.build_risk_matrix(input_reports, figsize=(10,5), clean=False, vectorize=False, model_type="likelihood", nlp_model_number=0, save=True, id_col='Unique_IDs')
dynamic_kwargs = {"clean":False, "vectorize":False, "model_type":"likelihood", 
                   "nlp_model_number":0}

#Static RM
"""
document_id_col = "INCIDENT_ID"
extra_cols = severity_x_cols + meta_inputs
list_of_attributes = ["Combined_Text"]
file = os.path.join(os.path.dirname(os.getcwd()),'smart_nlp','data', 'ICS_predictive_sitreps_full.csv')
ICS = Topic_Model_plus(document_id_col=document_id_col, extra_cols=extra_cols, list_of_attributes=list_of_attributes, combine_cols=False)
ICS.extract_preprocessed_data(file)
preprocessed_df = ICS.data_df.drop("Unnamed: 0", axis=1)
hazard_file = os.path.join(os.path.dirname(os.getcwd()),'smart_nlp','results','hazard_interpretation_v2.xlsx')
_, _, _, fires, frequency_fires, _, hazards, _, _ = calc_metrics(hazard_file, preprocessed_df,  target='Combined_Text', rm_outliers=False,unique_ids_col='Unique_IDs')
hazards = [hazard.replace(" ", "_") for hazard in hazards]
summary_reports = pd.read_csv(os.path.join(os.path.dirname(os.getcwd()),'smart_nlp','data','ICS_predictive_summaryreps_full.csv'))
severity_total, severity_table = calc_severity(fires, summary_reports, rm_outliers=False)

probs_df = rm.calc_static_likelihoods(frequency_fires, total_fires=len(summary_reports))
severity_df = rm.calc_static_severity(severity_table)
#rm.build_static_risk_matrix(severity_df, probs_df, figsize=(10,5))

rates_df = rm.calc_static_likelihoods_rates(frequency_fires)
rm.build_static_risk_matrix(severity_df, rates_df, rates=True, save=True,  figsize=(10,5))
"""

#Compare solution efficacy
"""
val =  pd.read_csv(os.path.join(os.path.dirname(os.getcwd()),'smart_nlp','data','ICS_val_sitreps_preprocessed.csv')).drop(["Unnamed: 0"], axis=1)
train =  pd.read_csv(os.path.join(os.path.dirname(os.getcwd()),'smart_nlp','data','ICS_test_sitreps_preprocessed.csv')).drop(["Unnamed: 0"], axis=1)
test =  pd.read_csv(os.path.join(os.path.dirname(os.getcwd()),'smart_nlp','data','ICS_train_sitreps_preprocessed.csv')).drop(["Unnamed: 0"], axis=1)
preprocessed_df = pd.concat([val, train, test]).reset_index(drop=True)
dynamic_kwargs['clean'] = False #preprocessed_df is already cleaned 
percent_same, results_df = rm.compare_results(preprocessed_df, dynamic_kwargs=dynamic_kwargs, rate=False)
print("For %", 100*percent_same, " of situation reports the dyanmic risk matrix is the static risk matrix")
print(results_df)
results_df.to_csv("dynamic_static_comparison.csv")
#with rates
rates_df = rm.calc_static_likelihoods_rates(frequency_fires)
rm.build_static_risk_matrix(severity_df, rates_df, rates=True,figsize=(10,5))
percent_same, results_df = rm.compare_results(preprocessed_df, dynamic_kwargs=dynamic_kwargs, rate=False)
print("For %", 100*percent_same, " of situation reports the dyanmic risk matrix is the static risk matrix build using rates")
print(results_df)
results_df.to_csv("dynamic_static_rates_comparison.csv")
"""