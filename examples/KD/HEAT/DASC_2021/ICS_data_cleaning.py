# -*- coding: utf-8 -*-
"""
Created on Mon May 24 12:14:25 2021

@author: srandrad
"""

import pandas as pd

import math

import sys
import os
sys.path.append(os.path.join('..', '..', '..', '..'))

from mika.kd.trend_analysis import *

incident_file = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir, os.pardir, os.pardir)),'data','ICS','ics209-plus-wf_incidents_1999to2014.csv')
incident_summary_df = pd.read_csv(incident_file)
incident_summary_df = incident_summary_df.drop("Unnamed: 0", axis=1)
incident_summary_df = incident_summary_df.loc[incident_summary_df["START_YEAR"]>=2006].reset_index(drop=True)
print(len(incident_summary_df))

preprocessed_file = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir, os.pardir)),'data','ICS',"ics209-plus-wf_sitreps_1999to2014.csv")#"ICS_sitreps.csv")
sitrep_df = pd.read_csv(preprocessed_file)
sitrep_df = sitrep_df.loc[sitrep_df["START_YEAR"]>2005].reset_index(drop=True)
print(len(sitrep_df))

fire_ids = incident_summary_df['INCIDENT_ID'].unique()
sitrep_ids = sitrep_df['INCIDENT_ID'].unique()
sitrep_df = sitrep_df[sitrep_df['INCIDENT_ID'].isin(fire_ids)].reset_index(drop=True)
incident_summary_df =incident_summary_df[incident_summary_df['INCIDENT_ID'].isin(sitrep_ids)].reset_index(drop=True)
print(sitrep_df)
print(len(sitrep_df))
print(incident_summary_df)
print(len(incident_summary_df))

##NaN handling: containment doy and discovery_doy
def convert_date_to_doy(date):
    days_per_month = {'1':31, '2':28, '3':31,'4':30, '5':31, '6':30, '7':31, '8':31, '9':30, '10':31, '11':30, '12':31}
    date = date.split(" ")
    if len(date)>0:
        date = date[0]
        date = date.split("-")
        month = date[1]
        day = date[2]
        doy = sum([days_per_month[str(month)] for month in range(1,int(month))])+ int(day)
        return (doy)
    else: 
        return 

fire_ids_to_drop = []
for i in range(len(incident_summary_df)):
    
    if math.isnan(incident_summary_df.iloc[i]["FOD_DISCOVERY_DOY"]):
        fire_id =  incident_summary_df.iloc[i]['INCIDENT_ID']
        sit_reps_for_fire = sitrep_df.loc[sitrep_df['INCIDENT_ID']==fire_id]
        possible_dates = sit_reps_for_fire["DISCOVERY_DOY"].unique() 
        #handles multiple start dates and fixes the summary reports and sitreps
        if len(possible_dates)>1:
            date = min(possible_dates)
            #changes the date in the sitrep
            for j in range(len(sitrep_df)):
                if sitrep_df.iloc[j]["INCIDENT_ID"]==fire_id:
                    sitrep_df.at[j,'DISCOVERY_DOY']=date
        elif len(possible_dates)==0 or math.isnan(possible_dates[0]):
            fire_ids_to_drop.append(fire_id)
        else:
            date = possible_dates[0]
        incident_summary_df.at[i,"FOD_DISCOVERY_DOY"] = date
        
    if math.isnan(incident_summary_df.iloc[i]["FOD_CONTAIN_DOY"]):
        #look at final report date, expected containment date
        fire_id =  incident_summary_df.iloc[i]['INCIDENT_ID']
        sit_reps_for_fire = sitrep_df.loc[(sitrep_df['INCIDENT_ID']==fire_id) & (sitrep_df["PCT_CONTAINED_COMPLETED"]==100.0)]
        possible_dates = sit_reps_for_fire["REPORT_DOY"].unique() 
        if len(possible_dates)>1:
            date = min(possible_dates)
        elif len(possible_dates) ==0 or math.isnan(possible_dates[0]):
            #check final report date, expected containment date
            final_report_date = convert_date_to_doy(incident_summary_df.iloc[i]['FINAL_REPORT_DATE'])
            if not math.isnan(final_report_date):
                date = final_report_date
            else: 
                expected_containment_date = convert_date_to_doy(incident_summary_df.iloc[i]['EXPECTED_CONTAINMENT_DATE'])
                if math.isnan(expected_containment_date):
                    fire_ids_to_drop.append(fire_id)
                else:
                    date = expected_containment_date
        else:
            date = possible_dates[0]
        incident_summary_df.at[i,"FOD_CONTAIN_DOY"] = date
        
print(len(set(fire_ids_to_drop)))

#Nan Handling: NaN handeling: personnel and aerial numbers
for i in range(len(incident_summary_df)):
    
    if math.isnan(incident_summary_df.iloc[i]["TOTAL_PERSONNEL_SUM"]):
        fire_id =  incident_summary_df.iloc[i]['INCIDENT_ID']
        sitrep_for_fire = sitrep_df.loc[sitrep_df['INCIDENT_ID']==fire_id]
        calculated_personnel_sum = sum([personnel for personnel in sitrep_for_fire["TOTAL_PERSONNEL"]])
        incident_summary_df.at[i,"TOTAL_PERSONNEL_SUM"] = calculated_personnel_sum
        #removes fires with no reported personnel
        if calculated_personnel_sum == 0:
            fire_ids_to_drop.append(fire_id)
        
    if math.isnan(incident_summary_df.iloc[i]["TOTAL_AERIAL_SUM"]):
        #confirms summaries reporting no aerial support had no aerial support, corrects if needed
        fire_id =  incident_summary_df.iloc[i]['INCIDENT_ID']
        sitrep_for_fire = sitrep_df.loc[sitrep_df['INCIDENT_ID']==fire_id]
        calculated_aerial_sum = sum([personnel for personnel in sitrep_for_fire["TOTAL_AERIAL"]])
        incident_summary_df.at[i,"TOTAL_AERIAL_SUM"] = calculated_aerial_sum
        
print(len(set(fire_ids_to_drop)))

#Finalizing data
total_ids = [id_ for id_ in sitrep_ids if id_ not in fire_ids_to_drop]
incident_summary_df = incident_summary_df.loc[incident_summary_df['INCIDENT_ID'].isin(total_ids)].reset_index(drop=True)
sitrep_df = sitrep_df.loc[sitrep_df['INCIDENT_ID'].isin(total_ids)].reset_index(drop=True)

#sitrep_df.to_csv(os.path.join('data','ICS',"ICS_sitreps_clean.csv"))
#incident_summary_df.to_csv(os.path.join('data','ICS','summary_reports_cleaned.csv'))

print(len(incident_summary_df),len(sitrep_df))
print(len([id_ for id_ in sitrep_df["INCIDENT_ID"].unique() if id_ in incident_summary_df["INCIDENT_ID"].unique()]))
sitrep_df
