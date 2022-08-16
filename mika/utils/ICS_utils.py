# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 12:03:38 2022

@author: srandrad
"""

from module.trend_analysis_functions import remove_outliers, plot_metric_time_series, plot_frequency_time_series
import numpy as np
import pandas as pd
import scipy.stats as stats
from tqdm import tqdm

def check_anamolies(time_of_occurence_days, time_of_occurence_pct_contained, frequency, fires, hazards):
    anomolous_hazards = {'OTTO days':{},
                        'OTTO pct': {},
                        'frequency days': {},
                        'frequency pct': {}}
    anoms = False
    for hazard in hazards:
        years_anom_days = []
        years_anom_pct = []
        years_anom_days_missing_nums = []
        years_anom_pct_missing_nums = []
        for year in time_of_occurence_days[hazard]:
            if time_of_occurence_days[hazard][year] == []:
                years_anom_days.append(year)
            if time_of_occurence_pct_contained[hazard][year] == []:
                years_anom_pct.append(year)
            if len(time_of_occurence_days[hazard][year]) != frequency[hazard][year]:
                years_anom_days_missing_nums.append(year)
            if len(time_of_occurence_pct_contained[hazard][year]) != frequency[hazard][year]:
                years_anom_pct_missing_nums.append(year)
        if years_anom_days != []:
            anomolous_hazards['OTTO days'][hazard] = years_anom_days
            anoms = True
        if years_anom_pct != []:
            anomolous_hazards['OTTO pct'][hazard] = years_anom_pct
            anoms = True
        if years_anom_days_missing_nums != []:
            anomolous_hazards['frequency days'][hazard] = years_anom_days_missing_nums
            anoms = True
        if years_anom_pct_missing_nums != []:
            anomolous_hazards['frequency pct'][hazard] = years_anom_pct_missing_nums
            anoms = True
    return anomolous_hazards, anoms

def calc_metrics(hazard_file, preprocessed_df, rm_outliers=True, distance=3, target='Combined Text', ids="INCIDENT_ID", unique_ids_col="INCIDENT_ID"):
     """
    uses the hazard focused sheet in the hazard interpretation results to calculate metrics.
    hazards are identified based on subject-action pairs 
    ARGUMENTS
     ---------
     hazard_file: string
         the location of the interpretation xlsx file
     preprocessed_df: string
         the location of preprocessed data that was fed into the model
     rm_outliers: boolean 
         used to remove outliers or not
     """
     years = preprocessed_df["START_YEAR"].unique()
     years.sort()
     hazard_info = pd.read_excel(hazard_file, sheet_name=['Hazard-focused'])
     hazards = hazard_info['Hazard-focused']['Hazard name'].tolist()
     categories = hazard_info['Hazard-focused']['Hazard Category'].tolist()
    
     time_of_occurence_days = {name:{year:[] for year in years} for name in hazards}
     time_of_occurence_pct_contained = {name:{year:[] for year in years} for name in hazards}
     frequency = {name:{year:0 for year in years} for name in hazards}
     fires = {name:{year:[] for year in years} for name in hazards}
     unique_ids = {name:{year:[] for year in years} for name in hazards}
     frequency_fires ={name:{year:0 for year in years} for name in hazards}

     for year in tqdm(years):
        temp_df = preprocessed_df.loc[preprocessed_df["START_YEAR"]==year].reset_index(drop=True)
        fire_ids = temp_df[ids].unique()
        for id_ in fire_ids:
            temp_fire_df = temp_df.loc[temp_df[ids]==id_].reset_index(drop=True)
            #date corrections
            start_date = temp_fire_df["DISCOVERY_DOY"].unique() #should only have one start date
            if len(start_date) != 1: 
                start_date = min(start_date)
            else: 
                start_date = start_date[0]
            if start_date == 365:
                    start_date = 0
        
            for j in range(len(temp_fire_df)):
                text = temp_fire_df.iloc[j][target]
                #check for hazard
                for i in range(len(hazard_info['Hazard-focused'])):
                    hazard_name = hazard_info['Hazard-focused'].iloc[i]['Hazard name']
                    hazard_subject_words = hazard_info['Hazard-focused'].iloc[i]['Hazard Noun/Subject']
                    hazard_subject_words = hazard_subject_words.split(", ")
                    hazard_action_words = hazard_info['Hazard-focused'].iloc[i]['Action/Descriptor']
                    hazard_action_words = hazard_action_words.split(", ")
                    negation_words = hazard_info['Hazard-focused'].iloc[i]['Negation words']
                    #check if a word in text is in hazard words, for each list in hazard words, no words in negation words
                    hazard_found = False
                    for word in hazard_subject_words:
                        if word in text:
                            hazard_found = True
                            subject_index = text.index(word)
                            break
                    if hazard_found == True:
                        hazard_found = False
                        for word in hazard_action_words:
                            if word in text:
                                hazard_index = text.index(word)
                                if abs(hazard_index-subject_index)<=distance:
                                    hazard_found = True
                                    break
                                else:
                                    hazard_found = False
                    """
                    additional_action_words = hazard_info['Hazard-focused'].iloc[i]['Action']
                    if hazard_found == True and not isinstance(additional_action_words,float):
                        hazard_found = False
                        for word in additional_action_words.split(", "):
                            if word in text:
                                hazard_index = text.index(word)
                                if abs(hazard_index-subject_index)<=distance:
                                    hazard_found = True
                                    break
                                else:
                                    hazard_found = False
                                    """
                    """
                    hazard_words = [hazard_subject_words, hazard_action_words]
                    for word_list in hazard_words:
                        hazard_found = False #ensures a word from each list must show up
                        for word in word_list:
                            if word in text:
                                hazard_found = True
                        if hazard_found == False: #ensures a word from each list must show up
                            break
                    """
                    
                    if isinstance(negation_words,str):
                        for word in negation_words.split(", "): #removes texts that have negation words
                            if word in text:
                                hazard_found = False 
                    
                    if hazard_found == True:
                        time_of_hazard = int(temp_fire_df.iloc[j]["REPORT_DOY"])
                    
                        #correct dates
                        if time_of_hazard<start_date: 
                            #print("dates corrected")
                            if time_of_hazard<30 and start_date<330: #report day is days since start, not doy 
                                time_of_hazard+=start_date
                            elif time_of_hazard<30 and start_date>=330:
                                start_date = start_date-365 #fire spans two years
                            else: #start and report day were incorrectly switched
                                temp_start = start_date
                                start_date = time_of_hazard
                                time_of_hazard = temp_start
                                
                        time_of_occurence_days[hazard_name][year].append(time_of_hazard-int(start_date))
                        time_of_occurence_pct_contained[hazard_name][year].append(temp_fire_df.iloc[j]["PCT_CONTAINED_COMPLETED"])
                        fires[hazard_name][year].append(id_)
                        unique_ids[hazard_name][year].append(temp_fire_df.iloc[j][unique_ids_col])
                        frequency[hazard_name][year] += 1
     for name in frequency_fires:
         for year in frequency_fires[name]:
             frequency_fires[name][year] = len(set(fires[name][year]))
     
     anomolous_hazards, anoms = check_anamolies(time_of_occurence_days, time_of_occurence_pct_contained, frequency, fires, hazards)
     if anoms == True:
         print("Error in calculation:")
         print(anomolous_hazards)
         
     if rm_outliers == True:
         for year in years:
             for hazard in hazards:
                 if len(time_of_occurence_pct_contained[hazard][year])>9 and hazard != 'Law Violations':
                    time_of_occurence_days[hazard][year] = remove_outliers(time_of_occurence_days[hazard][year])
                    time_of_occurence_pct_contained[hazard][year] = remove_outliers(time_of_occurence_pct_contained[hazard][year])
     return time_of_occurence_days, time_of_occurence_pct_contained, frequency, fires, frequency_fires, categories, hazards, years, unique_ids

def calc_severity(fires, summary_reports, rm_all_outliers=False, rm_severity_outliers=True):
    severity_total = {}; injuries_total = {}; fatalities_total = {}; str_dam_total = {}; str_des_total = {}
    for hazard in fires:
        severity_total[hazard] = {}; injuries_total[hazard] = {}; fatalities_total[hazard] = {}
        str_dam_total[hazard] = {}; str_des_total[hazard] = {}
        for year in fires[hazard]:
            severity_total[hazard][year] = []; injuries_total[hazard][year] = []; fatalities_total[hazard][year] = []
            str_dam_total[hazard][year] = []; str_des_total[hazard][year] = []
            ids = list(set(fires[hazard][year]))
            for id_ in ids:
                id_df = summary_reports.loc[summary_reports['INCIDENT_ID'] == id_].reset_index(drop=True)
                severity = int(id_df.iloc[0]["STR_DESTROYED_TOTAL"]) + int(id_df.iloc[0]["STR_DAMAGED_TOTAL"])+ int(id_df.iloc[0]["INJURIES_TOTAL"])+ int(id_df.iloc[0]["FATALITIES"])
                severity_total[hazard][year].append(severity)
                injuries_total[hazard][year].append(int(id_df.iloc[0]["INJURIES_TOTAL"])); fatalities_total[hazard][year].append(int(id_df.iloc[0]["FATALITIES"]))
                str_dam_total[hazard][year].append(int(id_df.iloc[0]["STR_DAMAGED_TOTAL"])); str_des_total[hazard][year].append(int(id_df.iloc[0]["STR_DESTROYED_TOTAL"]))
    severity_table = pd.DataFrame({"Hazard": [hazard for hazard in severity_total],
                                    "Average Severity": [round(np.average(remove_outliers([val for year in severity_total[hazard] for val in severity_total[hazard][year]],rm_outliers=rm_severity_outliers)),3) for hazard in severity_total],
                                    "std dev Severity": [round(np.std(remove_outliers([val for year in severity_total[hazard] for val in severity_total[hazard][year]],rm_outliers=rm_severity_outliers)),3) for hazard in severity_total],
                                    "Average Injuries": [round(np.average(remove_outliers([val for year in injuries_total[hazard] for val in injuries_total[hazard][year]],rm_outliers=rm_all_outliers)),3) for hazard in injuries_total],
                                    "std dev Injuries": [round(np.std(remove_outliers([val for year in injuries_total[hazard] for val in injuries_total[hazard][year]],rm_outliers=rm_all_outliers)),3) for hazard in injuries_total],
                                    "Average Fatalities": [round(np.average(remove_outliers([val for year in fatalities_total[hazard] for val in fatalities_total[hazard][year]],rm_outliers=rm_all_outliers)),3) for hazard in fatalities_total],
                                    "std dev Fatalities": [round(np.std(remove_outliers([val for year in fatalities_total[hazard] for val in fatalities_total[hazard][year]],rm_outliers=rm_all_outliers)),3) for hazard in fatalities_total],
                                    "Average Structures Damaged": [round(np.average(remove_outliers([val for year in str_dam_total[hazard] for val in str_dam_total[hazard][year]],rm_outliers=rm_all_outliers)),3) for hazard in str_dam_total],
                                    "std dev Structures Damaged": [round(np.std(remove_outliers([val for year in str_dam_total[hazard] for val in str_dam_total[hazard][year]],rm_outliers=rm_all_outliers)),3) for hazard in str_dam_total],
                                    "Average Structures Destroyed": [round(np.average(remove_outliers([val for year in str_des_total[hazard] for val in str_des_total[hazard][year]],rm_outliers=rm_all_outliers)),3) for hazard in str_des_total],
                                    "std dev Structures Destroyed": [round(np.std(remove_outliers([val for year in str_des_total[hazard] for val in str_des_total[hazard][year]],rm_outliers=rm_all_outliers)),3) for hazard in str_des_total],
                                    "n total": [len([val for year in severity_total[hazard] for val in severity_total[hazard][year]]) for hazard in severity_total],
                                    "n after outliers": [len(remove_outliers([val for year in severity_total[hazard] for val in severity_total[hazard][year]],rm_outliers=rm_all_outliers)) for hazard in severity_total],
                                    "formatted": [str(round(np.average(remove_outliers([val for year in severity_total[hazard] for val in severity_total[hazard][year]],rm_outliers=rm_severity_outliers)),3))+"+-"+str(round(np.std(remove_outliers([val for year in severity_total[hazard] for val in severity_total[hazard][year]],rm_outliers=rm_severity_outliers)),3)) for hazard in severity_total]
                                    }
                                    )
    return severity_total, severity_table

def correct_dates(preprocessed_df, temp_hazard_df, i, id_col):
    fire_df = preprocessed_df.loc[preprocessed_df[id_col]==temp_hazard_df.iloc[i][id_col]]
    start_date = fire_df["DISCOVERY_DOY"].unique() #should only have one start date
    if len(start_date) != 1: 
        start_date = min(start_date)
    else: 
        start_date = start_date[0]
    if start_date == 365:
            start_date = 0
    
    time_of_hazard = int(temp_hazard_df.iloc[i]["REPORT_DOY"])
    #correct dates
    if time_of_hazard<start_date: 
        #print(time_of_hazard, start_date)
        if time_of_hazard<30 and start_date<330: #report day is days since start, not doy 
            time_of_hazard+=start_date
        elif time_of_hazard<30 and start_date>=330:
            start_date = start_date-365 #fire spans two years
        else: #start and report day were incorrectly switched
            temp_start = start_date
            start_date = time_of_hazard
            time_of_hazard = temp_start
    return start_date, time_of_hazard

def calc_ICS_metrics(docs_per_hazard, preprocessed_df, id_col, unique_ids_col, rm_outliers=True):
    time_of_occurence_days = {name:{year:[] for year in docs_per_hazard[name]} for name in docs_per_hazard}
    time_of_occurence_pct_contained = {name:{year:[] for year in docs_per_hazard[name]} for name in docs_per_hazard}
    frequency = {name:{year:0 for year in docs_per_hazard[name]} for name in docs_per_hazard}
    fires = {name:{year:[] for year in docs_per_hazard[name]} for name in docs_per_hazard}
    unique_ids = {name:{year:[] for year in docs_per_hazard[name]} for name in docs_per_hazard}
    frequency_fires ={name:{year:0 for year in docs_per_hazard[name]} for name in docs_per_hazard}
    for hazard in tqdm(docs_per_hazard):
        for year in docs_per_hazard[hazard]:
            ids = docs_per_hazard[hazard][year]
            temp_hazard_df = preprocessed_df.loc[preprocessed_df[unique_ids_col].isin(ids)].reset_index(drop=True)
            for j in range(len(temp_hazard_df)):
                start_date, time_of_hazard = correct_dates(preprocessed_df, temp_hazard_df, j, id_col)
                time_of_occurence_days[hazard][year].append(time_of_hazard-int(start_date))
                #time_of_occurence_pct_contained[hazard][year].append(temp_hazard_df.iloc[j]["PCT_CONTAINED_COMPLETED"])
                if temp_hazard_df.iloc[j]["PCT_CONTAINED_COMPLETED"] <= 100: time_of_occurence_pct_contained[hazard][year].append(temp_hazard_df.iloc[j]["PCT_CONTAINED_COMPLETED"])
                fires[hazard][year].append(temp_hazard_df.iloc[j][id_col])
                unique_ids[hazard][year].append(temp_hazard_df.iloc[j][unique_ids_col])
                frequency[hazard][year] += 1
    for name in frequency_fires:
        for year in frequency_fires[name]:
            frequency_fires[name][year] = len(set(fires[name][year]))
    
    hazards = [h for h in docs_per_hazard]
    years = list(set([y for h in docs_per_hazard for y in docs_per_hazard[h]]))
    anomolous_hazards, anoms = check_anamolies(time_of_occurence_days, time_of_occurence_pct_contained, frequency, fires, hazards)
    if anoms == True:
        print("Error in calculation:")
        print(anomolous_hazards)
        
    if rm_outliers == True:
        for year in years:
            for hazard in hazards:
                if len(time_of_occurence_pct_contained[hazard][year])>9 and hazard != 'Law Violations':
                   time_of_occurence_days[hazard][year] = remove_outliers(time_of_occurence_days[hazard][year])
                   time_of_occurence_pct_contained[hazard][year] = remove_outliers(time_of_occurence_pct_contained[hazard][year])
    
    return time_of_occurence_days, time_of_occurence_pct_contained, frequency, fires, frequency_fires

def create_primary_results_table(time_of_occurence_days, time_of_occurence_pct_contained, frequency, fire_freq, preprocessed_df, categories, hazards, years, interval=False):
    """
    creates the primary results table consisting of average OTTO, frequency, and rate for each hazard
    all arguments are outputs from calc_metrics
    ARGUMENTS
    ---------
    time_of_occurence_days: nested dict
        dict with keys as hazard names, values as dict.
        inner dict has keys as years and values as list of OTTO in days.
    time_of_occurence_pct_contained: nested dict
        dict with keys as hazard names, values as dict.
        inner dict has keys as years and values as list of OTTO in pct containment.
    frequency: nested dict
        dict with keys as hazard names, values as dict.
        inner dict has keys as years and values as integer frequency
    fires: nested dict
        dict with keys as hazard names, values as dict.
        inner dict has keys as years and values as list of fire ids with that hazard
    preprocessed_df: dataframe
        the preprocessed data df
    categories: list
        list of hazard categories
    hazards: list 
        list of hazard names
    years: list
        list of years the data spans
    
    """
    data_df = {"Hazard Category": categories,
               "Hazard Name": hazards}
    #OTTO days average += std dev; interval
    days_total_data = {}
    for hazard in hazards:
        days_total_data[hazard] = []
        for year in time_of_occurence_days[hazard]:
            for val in time_of_occurence_days[hazard][year]:
                days_total_data[hazard].append(val)
    days_av = {hazard: np.average(days_total_data[hazard]) for hazard in hazards}
    days_std = {hazard: np.std(days_total_data[hazard]) for hazard in hazards}
    data_df["OTTO days"] = [str(round(days_av[hazard],3))+"+-"+str(round(days_std[hazard],3)) for hazard in hazards]
    if interval == True:
        data_df["OTTO days interval"] = [stats.t.interval(alpha=0.95, df=len(days_total_data[hazard])-1, loc=np.mean(days_total_data[hazard]), scale=stats.sem(days_total_data[hazard])) for hazard in hazards]
    #OTTO percent average += std dev; interval
    pct_total_data = {}
    for hazard in hazards:
        pct_total_data[hazard] = []
        for year in time_of_occurence_pct_contained[hazard]:
            for val in time_of_occurence_pct_contained[hazard][year]:
                pct_total_data[hazard].append(val)
    pct_av = {hazard: np.average(pct_total_data[hazard]) for hazard in hazards}
    pct_std = {hazard: np.std(pct_total_data[hazard]) for hazard in hazards}
    data_df["OTTO %"] = [str(round(pct_av[hazard],3))+"+-"+str(round(pct_std[hazard],3)) for hazard in hazards]
    if interval == True:
        data_df["OTTO % interval"] = [stats.t.interval(alpha=0.95, df=len(pct_total_data[hazard])-1, loc=np.mean(pct_total_data[hazard]), scale=stats.sem(pct_total_data[hazard])) for hazard in hazards]
    #rate per year
    sums_per_hazard = [np.sum([fire_freq[hazard][year] for year in fire_freq[hazard]]) for hazard in hazards]
    data_df["Average Occurrences per year"] = [round(val/len(years),3) for val in sums_per_hazard]
    #rate interval
    data_df["Rate"] = [str(round(np.average([fire_freq[hazard][year] for year in fire_freq[hazard]]),3))+"+-"+ str(round(np.std([fire_freq[hazard][year] for year in fire_freq[hazard]]),3)) for hazard in hazards]
    if interval == True:
        data_df["Rate Interval (stats)"] = [stats.t.interval(alpha=0.95, df=len([fire_freq[hazard][year] for year in fire_freq[hazard]])-1, loc=np.mean([fire_freq[hazard][year] for year in fire_freq[hazard]]), scale=stats.sem([fire_freq[hazard][year] for year in fire_freq[hazard]])) for hazard in hazards]
        data_df["Rate Interval (percentile)"] = [str(round(np.average([fire_freq[hazard][year] for year in fire_freq[hazard]]),3))+" ("+str(round(np.percentile([fire_freq[hazard][year] for year in fire_freq[hazard]], 2.5),3))+","+str(round(np.percentile([fire_freq[hazard][year] for year in fire_freq[hazard]], 97.5),3))+")" for hazard in hazards]
    #fires per rate
    total_fires = len(preprocessed_df["INCIDENT_ID"].unique())
    data_df["Average fires per occurrence"] = [round(total_fires/val,3) for val in sums_per_hazard]

    #total frequency
    data_df["Total Frequency"] = [np.sum([frequency[hazard][year] for year in frequency[hazard]]) for hazard in hazards]
    #fire frequency
    data_df["Total Fire Frequency"] = sums_per_hazard
    return data_df

def graph_ICS_time_series(time_of_occurence_days, time_of_occurence_pct_contained, frequency, frequency_fires, hazards, categories):
    line_styles = []
    line_style_options = ['--', ':','-']
    line_style_dict = {list(set(categories))[i]:line_style_options[i] for i in range(len(list(set(categories))))}
    category_counter = {list(set(categories))[i]:0 for i in range(len(list(set(categories))))}
    marker_options = ['.', 'v', '^', 's', 'D', 'X', '+','*', '<', '>']
    markers = []
    for i in range(len(hazards)):
        category = categories[i]
        marker = marker_options[category_counter[category]]
        category_counter[category] += 1
        markers.append(marker)
        line_styles.append(line_style_dict[category])
    #plot days
    plot_metric_time_series(metric_data=time_of_occurence_days, metric_name='Days', line_styles=line_styles, markers=markers, title="Operational Time to Occurrence", time_name="Year", scaled=False, xtick_freq=1, show_std=True, save=False, dataset_name="", yscale=None)
    #plot pct contained
    plot_metric_time_series(metric_data=time_of_occurence_pct_contained, metric_name='% Containment', line_styles=line_styles, markers=markers, title="Operational Time to Occurrence", time_name="Year", scaled=False, xtick_freq=1, show_std=True, save=False, dataset_name="", yscale=None)
    #plot total frequency
    plot_frequency_time_series(frequency, metric_name='Frequency', line_styles=line_styles, markers=markers, title="Total Hazard Frequency", time_name="Year", xtick_freq=1, scale=False, save=False, dataset_name="")
    #plot fire frequency
    plot_frequency_time_series(frequency_fires, metric_name='Frequency', line_styles=line_styles, markers=markers, title="Hazard Frequency", time_name="Year", xtick_freq=1, scale=False, save=False, dataset_name="")
    #plot severity
    