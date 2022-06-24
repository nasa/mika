# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 12:14:51 2021

@author: srandrad
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
from nltk.corpus import words
from nltk.stem.porter import PorterStemmer
import scipy.stats as stats
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score
import seaborn as sn
from tqdm import tqdm
import random

def minmax_scale(data_list):
    """
    performs minmax scaling on a single data list in order to normalize the data.
    normalization is required prior to regression and ML.
    Also it is used for graphing multiple time series on the same axes.
    
    ARGUMENTS
    ---------
    data_list: list
        list of numerical data that will be scaled using minmax scaling
    """
    max_ = max(data_list)
    min_ = min(data_list)
    scaled_list = []
    for data in data_list:
        scaled_data = (data-min_)/(max_-min_)
        scaled_list.append(scaled_data)
    return scaled_list

def corr_sig(df=None):
    """returns the probability matrix for a correlation matrix
    
    ARGUMENTS
    ---------
    df: dataframe
        df storing the data that is used to create the correlation matrix.
        rows are years, columns are predictors + hazard frequencies
    """
    p_matrix = np.zeros(shape=(df.shape[1],df.shape[1]))
    for col in df.columns:
        for col2 in df.drop(col,axis=1).columns:
            _ , p = stats.pearsonr(df[col],df[col2])
            p_matrix[df.columns.to_list().index(col),df.columns.to_list().index(col2)] = p
    return p_matrix

def regression_feature_importance(predictors, hazards, correlation_mat_total):
    data = correlation_mat_total
    predictors = predictors
    hazards = ["total "+ h for h in hazards]
    full_model_score = []
    full_model_MSE = []
    xi_score = {predictor+" score":[] for predictor in predictors}
    xi_MSE = {predictor+" MSE":[] for predictor in predictors}

    importance_data = {}
    for hazard in hazards:
        X = data[predictors]
        y = data[[hazard]]
        regr = linear_model.LinearRegression()
        regr.fit(X,y)
        y_pred = regr.predict(X)
        model_score = r2_score(y, y_pred)#regr.score(X,y)
        full_model_score.append(round(model_score,3))
        model_MSE = mean_squared_error(y_pred,y)
        full_model_MSE.append(round(model_MSE,3))
        importance = regr.coef_
        importance_data[hazard] = importance[0]
        #run regression for each predictor individually
        for predictor in predictors:
            X = data[[predictor]]
            y = data[[hazard]]
            regr = linear_model.LinearRegression()
            regr.fit(X,y)
            y_pred = regr.predict(X)
            model_score = r2_score(y, y_pred)
            xi_score[predictor+" score"].append(round(model_score,3))
            model_MSE = mean_squared_error(y_pred,y)
            xi_MSE[predictor+" MSE"].append(round(model_MSE,3))
    results_data = {"hazard":hazards, "R2 for full model":full_model_score, "MSE for full model": full_model_MSE}
    results_data.update(xi_score)
    results_data.update(xi_MSE)
    results_df = pd.DataFrame(results_data)
    
    #graph feature importance
    X_axis = np.arange(len(predictors))
    num_bars = len(hazards)
    width = 1/(num_bars+2)
    i=1
    colors = cm.tab20(np.linspace(0, 1, num_bars))
    plt.figure(figsize=(len(hazards),4))
    for hazard in importance_data:
        plt.bar(X_axis+(width*i), importance_data[hazard], width, label=hazard.replace("total ",""), color=colors[i-1])
        i+=1
    plt.xticks(X_axis+(width*np.ceil(num_bars/2)), [pred.replace("total ","") for pred in predictors], rotation=70)
    plt.tick_params(labelsize=14)
    plt.xlabel("Predictors", fontsize=14)
    plt.ylabel("Importance", fontsize=14)
    plt.legend(bbox_to_anchor=(1, 1.1), loc='upper left', fontsize=14)
    plt.show()
    
    return results_df

def multiple_reg_feature_importance(predictors, hazards, correlation_mat_total):
    """
    builds multiple regression model for hazrd frequency given the predictors.
    also performs predictor importance to identify which predictors are most relevant to the hazards frequency
    
    ARGUMENTS
    ---------
    predictors: list
        list of predictor names, used to identify inputs to multiple regression
    hazards: list
        list of hazard names, used to identify targets for multiple regression
    correlation_mat_total: dataframe
        stores the time series values that were used for correlation matrix. 
        rows are years, columns are predictors + hazard frequencies
    """
    #predictors and hazards must be values from the correlation_mat_total
    data = correlation_mat_total
    predictors = predictors
    hazards = ["total "+ h for h in hazards]
    full_model_score = []
    full_model_MSE = []
    xi_score = {predictor+" removed score":[] for predictor in predictors}
    xi_MSE = {predictor+" removed MSE":[] for predictor in predictors}
    xi_score_delta = {predictor+" removed score":[] for predictor in predictors}
    xi_MSE_delta = {predictor+" removed MSE":[] for predictor in predictors}
    importance_data = {}
    for hazard in hazards:
        X = data[predictors]
        y = data[[hazard]]
        regr = linear_model.LinearRegression()
        regr.fit(X,y)
        y_pred = regr.predict(X)
        model_score = r2_score(y, y_pred)#regr.score(X,y)
        full_model_score.append(round(model_score,3))
        model_MSE = mean_squared_error(y_pred,y)
        full_model_MSE.append(round(model_MSE,3))
        importance = regr.coef_
        importance_data[hazard] = importance[0]
        
        for predictor in predictors:
            #shuffling each predictor
            index_list = np.random.RandomState(seed=42).permutation(len(data[predictors]))
            X = {}
            y = data[hazard]
            for i in range(len(predictors)):
                if predictors[i] == predictor:
                    shuffled = [data.iloc[i][predictor] for i in index_list]
                    X[predictor] = shuffled
                else:
                    X[predictors[i]] = data.loc[:, predictors[i]]
            regr = linear_model.LinearRegression()
            X_new = pd.DataFrame(X)[predictors]
            X = data[predictors]
            regr.fit(X,y)
            y_pred = regr.predict(X_new)
            predictor_score = r2_score(y, y_pred)
            
            xi_score[predictor+" removed score"].append(round(predictor_score,3))
            predictor_MSE = mean_squared_error(y_pred,y)
            xi_MSE[predictor+" removed MSE"].append(round(predictor_MSE,3))
            
            xi_score_delta[predictor+" removed score"].append(round(abs(model_score-predictor_score),3))
            xi_MSE_delta[predictor+" removed MSE"].append(round(model_MSE-predictor_MSE,3))
    
    results_data = {"hazard":hazards, "R2 for full model":full_model_score, "MSE for full model": full_model_MSE}
    results_data.update(xi_score)
    results_data.update(xi_MSE)
    results_df = pd.DataFrame(results_data)
    
    delta_data = {"hazard":hazards, "R2 for full model":full_model_score, "MSE for full model": full_model_MSE}
    delta_data.update(xi_score_delta)
    delta_data.update(xi_MSE_delta)
    delta_df = pd.DataFrame(delta_data)
    
    #graph feature importance
    X_axis = np.arange(len(predictors))
    num_bars = len(hazards)
    width = 1/(num_bars+2)
    i=1
    colors = cm.tab20(np.linspace(0, 1, num_bars))
    plt.figure(figsize=(len(hazards),4))
    for hazard in importance_data:
        plt.bar(X_axis+(width*i), importance_data[hazard], width, label=hazard.replace("total ",""), color=colors[i-1])
        i+=1
    plt.xticks(X_axis+(width*np.ceil(num_bars/2)), [pred.replace("total ","") for pred in predictors], rotation=70)
    plt.tick_params(labelsize=14)
    plt.xlabel("Predictors", fontsize=14)
    plt.ylabel("Importance", fontsize=14)
    plt.legend(bbox_to_anchor=(1, 1.1), loc='upper left', fontsize=14)
    plt.show()
    
    return results_df, delta_df

def check_rates():
    return 

def remove_outliers(data, threshold=1.5, rm_outliers=True):
    #print(data)
    if data == [] or rm_outliers == False:
        return data
    Q1 = np.quantile(data,0.25)
    Q3 = np.quantile(data,0.75)
    IQR = Q3 - Q1
    new_data = [pt for pt in data if (pt>(Q1-1.5*IQR)) and (pt<(Q3+1.5*IQR))]
    #print(len(data), len(new_data))
    return new_data

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
    #TODO: Generalize this
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

def identify_docs_per_hazard(hazard_file, preprocessed_df, results_file, text_field, time_field, id_field, results_text_field=None, doc_topic_dist_field=None, topic_thresh=0.0, ids_to_drop=[]):
    hazard_info = pd.read_excel(hazard_file, sheet_name=['topic-focused'])
    hazards = list(set(hazard_info['topic-focused']['Hazard name'].tolist()))
    hazards = [hazard for hazard in hazards if isinstance(hazard,str)]
    docs = preprocessed_df[id_field].tolist()
    hazard_words_per_doc = {hazard:['none' for doc in docs] for hazard in hazards}
    time_period = preprocessed_df[time_field].unique()
    categories = hazard_info['topic-focused']['Hazard Category'].tolist()
    punctuation = ['.', ',', "'", '"', '?', '!']
    stemmer = PorterStemmer()
    english_words = [w.lower() for w in words.words()]
    if '.csv' in results_file:
        results = pd.read_csv(results_file)
        results = {results_text_field: results}
        doc_topic_distribution = None
    elif '.xlsx' in results_file:
        results = pd.read_excel(results_file, sheet_name=[text_field])
        if doc_topic_dist_field:
            doc_topic_distribution = pd.read_excel(results_file, sheet_name=[doc_topic_dist_field])[doc_topic_dist_field]
        else:
            doc_topic_distribution = None
    if results_text_field == None:
        results_text_field = text_field
    frequency = {name:{str(time_p):0 for time_p in time_period} for name in hazards}
    docs_per_hazard = {hazard:{str(time_p):[] for time_p in time_period} for hazard in hazards}
    if results[results_text_field].at[0,'topic number'] == -1:
        begin_nums = 1
    else:
        begin_nums = 0
    for i in tqdm(range(len(hazards))):
         num_df = hazard_info['topic-focused'].loc[hazard_info['topic-focused']['Hazard name'] == hazards[i]].reset_index(drop=True)
         nums = [int(i)+begin_nums for nums in num_df['Topic Number'] for i in str(nums).split(", ")]#num_df['Topic Number'].to_list() #identifies all topics related to this hazard
         nums = [int(i) for nums in num_df['Topic Number'] for i in str(nums).split(", ")]
         ids_df = results[results_text_field].loc[nums]
         ids_ = ids_df['documents'].to_list()
         ids_ = [id_ for k in range(len(ids_)) for id_ in ids_[k].strip("[]").split(", ")]
         ids_= [w.replace("'","") for w in ids_]
         ids_ = set([id_ for id_ in ids_ if id_ not in ids_to_drop])
         # ids_ = ids_ only if topic nums > thres
         if doc_topic_distribution is not None:
             new_ids = []
             for id_ in ids_:
                 #check that topic prob> thres for at least one num
                 id_df = doc_topic_distribution.loc[doc_topic_distribution['document number']==id_].reset_index(drop=True)
                 probs = [float(id_df.iloc[0][text_field].strip("[]").split(" ")[num].strip("\n")) for num in nums]
                 max_prob = max(probs)
                 if max_prob > topic_thresh:
                     new_ids.append(id_)
             ids_ = new_ids

         temp_df = preprocessed_df.loc[preprocessed_df[id_field].astype(str).isin(ids_)].reset_index(drop=True)
         fire_ids = temp_df[id_field].unique()
         for id_ in fire_ids:
            temp_fire_df = temp_df.loc[temp_df[id_field]==id_].reset_index(drop=True)
            for j in range(len(temp_fire_df)):
                text = temp_fire_df.iloc[j][text_field]
                text = " ".join(text)
                text = text.replace(".", " ")
                #check for hazard -- looks at the hazard relevant words from the topic
                hazard_name = hazards[i]
                hazard_words = num_df['Relevant hazard words'].to_list()
                hazard_words = list(set([word for words in hazard_words for word in words.split(", ")]))
                negation_words = num_df['Negation words'].to_list()
                negation_words = [word for word in negation_words if isinstance(word, str)]
                #need to check if a word in text is in hazard words
                hazard_found = False
                for h_word in hazard_words:
                    #end_ind = 0
                    if h_word in text:
                        hazard_found = True
                        break
                        """
                        if len(h_word.split(" ")) == 1:
                            word_ind = text[end_ind:].find(h_word)
                            start_ind = max([s_ind for s_ind in [0, text[:word_ind].rfind(" ")]+[text[:word_ind].rfind(p) for p in punctuation] if s_ind!=-1])
                            next_punctuation = [s_ind for s_ind in [text[start_ind:].find(" ")+start_ind]+[text[start_ind:].find(p)+start_ind for p in punctuation] if s_ind > start_ind]
                            print(word_ind, start_ind, next_punctuation)
                            if next_punctuation == []:
                                next_punctuation = -1
                            else:
                                next_punctuation = min(next_punctuation)
                                #print(word_ind, next_punctuation)
                                print(word_ind, start_ind, next_punctuation)
                            
                            end_ind = max(next_punctuation, word_ind)
                            full_word = text[start_ind:end_ind].strip(" .,'")
                            print(full_word, h_word)
                            #full_word = text[text[:word_ind].rfind(" "):text[word_ind:].find(" ")] #what about words ending in punctuation????? beginning after punctuation?
                            if len(full_word) > len(h_word):
                                #check to see if different words - stem both
                                if stemmer.stem(full_word) == stemmer.stem(h_word):
                                    hazard_found = True
                                elif full_word not in english_words: #likely two conjoined words, split them
                                    word_split_ind = full_word.find(h_word)
                                    word_split = [full_word[:word_split_ind], full_word[word_split_ind:]]
                                    if word_split[0] in english_words and word_split[1] in english_words:
                                        hazard_found = True
                            elif len(full_word) == len(h_word):
                                hazard_found = True
                        else:
                            hazard_found = True"""
                
                if negation_words!=[]:
                    for neg_words in negation_words:
                        for word in neg_words.split(", "):#removes texts that have negation words
                            if word in text:
                                hazard_found = False

                if hazard_found == True:
                    year = temp_fire_df.iloc[j][time_field]
                    docs_per_hazard[hazard_name][str(year)].append(id_)
                    frequency[hazard_name][str(year)] += 1
                    hazard_words_per_doc[hazard_name][docs.index(id_)] = h_word

    return frequency, docs_per_hazard, hazard_words_per_doc

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
     

def topic_based_calc_metrics(hazard_file, preprocessed_df, results_file, rm_outliers=True, unique_ids_col="Unique IDs"):
    ###TODO: change from metric calc to just identifying docs, then metric calc separate
     """
     Uses the topic-focused spread sheet, goes through each hazard relevant topic,
     for each document in that topic, if the hazard relevant words appear in th document,
     then metrics are calculated for that document.
     
     ARGUMENTS
     ---------
     hazard_file: string
         the location of the interpretation xlsx file
     years: list
         the years the data spans TODO: remove this 
     preprocessed_df: string
         the location of preprocessed data that was fed into the model
     results_file: string
         the location of the results xlxs
     rm_outliers: boolean 
         used to remove outliers or not
     """
     hazard_info = pd.read_excel(hazard_file, sheet_name=['topic-focused'])
     hazards = list(set(hazard_info['topic-focused']['Hazard name'].tolist()))
     hazards = [hazard for hazard in hazards if isinstance(hazard,str)]
     categories = hazard_info['topic-focused']['Hazard Category'].tolist()
     results = pd.read_excel(results_file, sheet_name=['Combined Text'])
     years = preprocessed_df["START_YEAR"].unique()
     years.sort()
     time_of_occurence_days = {name:{year:[] for year in years} for name in hazards}
     time_of_occurence_pct_contained = {name:{year:[] for year in years} for name in hazards}
     frequency = {name:{year:0 for year in years} for name in hazards}
     fires = {name:{year:[] for year in years} for name in hazards}
     unique_ids = {name:{year:[] for year in years} for name in hazards}
     frequency_fires ={name:{year:0 for year in years} for name in hazards}
     for i in range(len(hazards)):
         num_df = hazard_info['topic-focused'].loc[hazard_info['topic-focused']['Hazard name'] == hazards[i]]
         nums = nums = [int(i) for nums in num_df['Topic Number'] for i in str(nums).split(", ")]
         ids_df = results['Combined Text'].loc[nums]
         ids_ = ids_df['documents'].to_list()
         ids_ = ids_[0].strip("[]").split(", ")
         ids_= [w.replace("'","") for w in ids_]
         temp_df = preprocessed_df.loc[preprocessed_df["Unique IDs"].isin(ids_)].reset_index(drop=True)
         fire_ids = temp_df["INCIDENT_ID"].unique()
         for id_ in fire_ids:
            temp_fire_df = temp_df.loc[temp_df["INCIDENT_ID"]==id_].reset_index(drop=True)
            #date corrections
            start_date = temp_fire_df["DISCOVERY_DOY"].unique() #should only have one start date
            if len(start_date) != 1: 
                start_date = min(start_date)
            else: 
                start_date = start_date[0]
            if start_date == 365:
                    start_date = 0
        
            for j in range(len(temp_fire_df)):
                text = temp_fire_df.iloc[j]["Combined Text"]
                #check for hazard -- looks at the hazard relevant words from the topic
                hazard_name = hazards[i]
                hazard_words = num_df['Relevant hazard words'].to_list()
                #print(hazard_words, len(hazard_words), hazard_words[0])
                hazard_words = [word for words in hazard_words for word in words.split(", ")]
                #print(hazard_words, len(hazard_words), hazard_words[0])
                negation_words = num_df['Negation words'].to_list()
                negation_words = [word for word in negation_words if isinstance(word, str)]
                #print(negation_words)
                #need to check if a word in text is in hazard words
                hazard_found = False
                for word in hazard_words:
                    if word in text:
                        hazard_found = True
                
                if negation_words!=[]:
                    for neg_words in negation_words:
                        for word in neg_words.split(", "):#removes texts that have negation words
                            if word in text:
                                hazard_found = False 
                
                if hazard_found == True:
                    time_of_hazard = int(temp_fire_df.iloc[j]["REPORT_DOY"])
                
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
                    year = temp_fire_df.iloc[j]["START_YEAR"]
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
                 time_of_occurence_days[hazard][year] = remove_outliers(time_of_occurence_days[hazard][year])
                 time_of_occurence_pct_contained[hazard][year] = remove_outliers(time_of_occurence_pct_contained[hazard][year])
     return time_of_occurence_days, time_of_occurence_pct_contained, frequency, fires, frequency_fires, categories, hazards, years, unique_ids



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
    #fires per rate
    total_fires = len(preprocessed_df["INCIDENT_ID"].unique())
    data_df["Average fires per occurrence"] = [round(total_fires/val,3) for val in sums_per_hazard]

    #total frequency
    data_df["Total Frequency"] = [np.sum([frequency[hazard][year] for year in frequency[hazard]]) for hazard in hazards]
    #fire frequency
    data_df["Total Fire Frequency"] = sums_per_hazard
    return data_df

def create_metrics_time_series(time_of_occurence_days, time_of_occurence_pct_contained, frequency, frequency_fires, years, categories, combined=True, scale=False):
    """
    creates the time series graphs for OTTO, frequency, and rate for each hazard
    arguments are outputs from calc_metrics
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
    years: list
        list of years the data spans
    combined: bool
        combines all the time series into one graph if true, seperate graphs if false
    
    """
       

    days_averages = {hazard: [np.average(time_of_occurence_days[hazard][year]) for year in time_of_occurence_days[hazard]] for hazard in time_of_occurence_days}
    days_stddevs = {hazard: [np.std(time_of_occurence_days[hazard][year]) for year in time_of_occurence_days[hazard]] for hazard in time_of_occurence_days}
    pct_averages = {hazard: [np.average(time_of_occurence_pct_contained[hazard][year]) for year in time_of_occurence_pct_contained[hazard]] for hazard in time_of_occurence_pct_contained}
    pct_stddevs = {hazard: [np.std(time_of_occurence_pct_contained[hazard][year]) for year in time_of_occurence_pct_contained[hazard]] for hazard in time_of_occurence_pct_contained}
    
    colors = cm.tab20(np.linspace(0, 1, len(days_averages)))
    line_styles = ['--', ':','-']
    line_style_dict = {list(set(categories))[i]:line_styles[i] for i in range(len(list(set(categories))))}
    category_counter = {list(set(categories))[i]:0 for i in range(len(list(set(categories))))}
    marker_styles = ['.', 'v', '^', 's', 'D', 'X', '+','*', '<', '>', ]
    years_plot = years#[year.strip('0').strip('.') for year in years]
    #plot OTTO
    if combined == True:
        plt.figure()
        #plt.title("Change in Operational Time of Occurence for Hazards from 2006-2014")
        plt.xlabel("Year", fontsize=16)
        plt.ylabel("% Containment", fontsize=16)
        i=0
        for hazard in pct_averages:
            category = categories[i]
            plt.errorbar(years_plot, pct_averages[hazard], yerr=pct_stddevs[hazard], color=colors[i], marker=marker_styles[category_counter[category]], linestyle=line_style_dict[category], label=hazard, capsize=5, markeredgewidth=1)
            category_counter[category] += 1
            i += 1
        plt.legend(bbox_to_anchor=(1, 1.1), loc='upper left', fontsize=14)
        plt.tick_params(labelsize=16)
        plt.show()
        
        plt.figure()
        #plt.title("Change in Operational Time of Occurence for Hazards from 2006-2014")
        plt.xlabel("Year", fontsize=16)
        plt.ylabel("Days", fontsize=16)
        i=0
        line_counter = 0
        marker_counter = 0
        category_counter = {list(set(categories))[i]:0 for i in range(len(list(set(categories))))}
        for hazard in pct_averages:
            category = categories[i]
            plt.errorbar(years_plot, days_averages[hazard], yerr=days_stddevs[hazard], color=colors[i], marker=marker_styles[category_counter[category]], linestyle=line_style_dict[category], label=hazard, capsize=5, markeredgewidth=1)
            category_counter[category] += 1
            i += 1 
            
        plt.tick_params(labelsize=16)
        plt.legend(bbox_to_anchor=(1, 1.1), loc='upper left', fontsize=14)
        plt.show()
    elif combined == False:
        for hazard in pct_averages:
            plt.figure()
            plt.title("Change in Operational Time of Occurence for \n"+hazard+" from 2006-2014")
            plt.xlabel("Year", fontsize=16)
            plt.ylabel("% Containment", fontsize=16)
            plt.errorbar(years_plot, pct_averages[hazard], yerr=pct_stddevs[hazard], label=hazard,  capsize=5, markeredgewidth=5)
            plt.tick_params(labelsize=16)
            plt.legend(fontsize=16)
            plt.show()
        for hazard in pct_averages:
            plt.figure()
            plt.title("Change in Operational Time of Occurence for \n"+hazard+" from 2006-2014")
            plt.xlabel("Year", fontsize=16)
            plt.ylabel("Days", fontsize=16)
            plt.tick_params(labelsize=16)
            plt.errorbar(years_plot, days_averages[hazard], yerr=days_stddevs[hazard], label=hazard,  capsize=5, markeredgewidth=5)
            plt.legend(fontsize=16)
            plt.show()
    #plot frequency
    category_counter = {list(set(categories))[i]:0 for i in range(len(list(set(categories))))}
    frequencies = {hazard: [frequency[hazard][year] for year in frequency[hazard]] for hazard in frequency}
    frequencies_fire = {hazard: [frequency_fires[hazard][year] for year in frequency_fires[hazard]] for hazard in frequency_fires}
    hazard_freqs_scaled = {hazard: minmax_scale(frequencies[hazard]) for hazard in frequencies}
    fire_freqs_scaled = {hazard: minmax_scale(frequencies_fire[hazard]) for hazard in frequencies_fire}
    
    if combined == True:
        plt.figure()
        plt.xlabel("Year", fontsize=16)
        #plt.title("Change in Hazard Frequency from 2006-2014")
        i = 0
        if scale == True:
            plt.ylabel("Total Scaled", fontsize=16)
            for hazard in hazard_freqs_scaled:
                category = categories[i]
                plt.plot(years_plot, hazard_freqs_scaled[hazard], color=colors[i], label=hazard, marker=marker_styles[category_counter[category]], linestyle=line_style_dict[category])
                category_counter[category] += 1
                i += 1
        else:
            plt.ylabel("Total", fontsize=16)
            for hazard in frequencies:
                category = categories[i]
                plt.plot(years_plot, frequencies[hazard], color=colors[i], label=hazard,marker=marker_styles[category_counter[category]], linestyle=line_style_dict[category])
                category_counter[category] += 1
                i += 1
            
        plt.legend(bbox_to_anchor=(1, 1.1), loc='upper left', fontsize=14)
        plt.tick_params(labelsize=16)
        plt.show()
        
        plt.figure()
        plt.xlabel("Year", fontsize=16)
        #plt.title("Change in Hazard Frequency from 2006-2014")
        i = 0
        category_counter = {list(set(categories))[i]:0 for i in range(len(list(set(categories))))}
        if scale == True:
            plt.ylabel("Total Scaled", fontsize=16)
            for hazard in hazard_freqs_scaled:
                category = categories[i]
                plt.plot(years_plot, fire_freqs_scaled[hazard], color=colors[i], label=hazard,marker=marker_styles[category_counter[category]], linestyle=line_style_dict[category])
                category_counter[category] += 1
                i += 1
        else:
            plt.ylabel("Total", fontsize=16)
            for hazard in frequencies_fire:
                category = categories[i]
                plt.plot(years_plot, frequencies_fire[hazard], color=colors[i], label=hazard,marker=marker_styles[category_counter[category]], linestyle=line_style_dict[category])
                category_counter[category] += 1
                i += 1
            
        plt.legend(bbox_to_anchor=(1, 1.1), loc='upper left', fontsize=14)
        plt.tick_params(labelsize=16)
        plt.show()
    elif combined == False:
        for hazard in frequencies:
            plt.figure()
            plt.ylabel("Total Scaled", fontsize=16)
            plt.xlabel("Year", fontsize=16)
            #plt.title("Change in "+hazard+"\n Frequency from 2006-2014")
            plt.plot(years_plot, frequencies[hazard], label=hazard)
            plt.tick_params(labelsize=16)
            plt.legend()
            plt.show()
    return days_averages, days_stddevs, pct_averages, pct_stddevs, frequencies, hazard_freqs_scaled, frequencies_fire, fire_freqs_scaled

def create_correlation_matrix(predictors_scaled, frequencies_scaled, graph=True):
    """
    creates the correlation matrix between all predictors and all hazard frequencies
    all arguments are outputs from create_metrics_time_series
    
    ARGUMENTS
    ---------
    predictors_scaled: dict
        dictionary with keys as predictor names, values as a time series list of values scaled using minmax
    frequencies_scaled: dict
        dictionary with keys as hazard names, values as times series list of frequencies scaled using minmax
    """
    correlation_mat_data = predictors_scaled.copy()
    for hazard in frequencies_scaled:
        correlation_mat_data["total " +hazard] = frequencies_scaled[hazard]
    correlation_mat_total = pd.DataFrame(correlation_mat_data)
    corrMatrix =correlation_mat_total.corr()
    p_values = corr_sig(correlation_mat_total)                     # get p-Value
    mask = np.invert(np.tril(p_values<0.05)) 
    if graph == True:
        sn.heatmap(corrMatrix, annot=True, mask=mask)
        plt.title("Correlational Matrix for Trends in \n Fires, Operations, Intensity, and Hazard Frequency per year")
        plt.show()
        
        sn.heatmap(corrMatrix, annot=True)
        plt.title("Correlational Matrix for Trends in \n Fires, Operations, Intensity, and Hazard Frequency per year")
        plt.show()

    return corrMatrix, correlation_mat_total, p_values

def reshape_correlation_matrix(corrMatrix, p_values, predictors, hazards):
    """
    reshapes the correlation matrix between all predictors and all hazard frequencies
    columns are predictors and rows are hazards
    arguments are outputs from create_correlation_matrix
    
    ARGUMENTS
    ---------
    correlation_mat_total: dataframe
        correlation matrix df with columns/rows as predictor and hazard names, values as correlation coefficients
    correlation_mat_sig: dataframe
        df with columns/rows as predictor and hazard names, values as correlation coefficient p-values
    predictors: list
        list of predictor names
    hazards: list
        list of hazard names
    """
    new_corr_data = {predictor.replace("total ",""):[] for predictor in predictors}
    new_p_val_data = {predictor:[] for predictor in predictors}
    annotation_data = {predictor:[] for predictor in predictors}
    
    i = 0
    for hazard in hazards:
        j = 0
        for predictor in predictors: 
            r = round(corrMatrix[hazard][predictor],2)
            new_corr_data[predictor.replace("total ","")].append(r)
            p_val = p_values[i][j]
            new_p_val_data[predictor].append(p_val)
            notation = ""
            if p_val < 0.05:
                notation = "*"
            if p_val < 0.01:
                notation = "**"
            if p_val < 0.001:
                notation = "***"
            annotation_data[predictor].append(str(r)+notation)
            j += 1
        i += 1
    hazards = [h.replace("total ", "") for h in hazards]
    new_corr_df = pd.DataFrame(new_corr_data, index=hazards)
    annotation_df = pd.DataFrame(annotation_data, index=hazards)
   
    fig, (cax, ax) = plt.subplots(nrows=2, figsize=(8,8.025),  gridspec_kw={"height_ratios":[0.025, 1]})

    # Draw heatmap
    sn.heatmap(new_corr_df, annot=annotation_df,  fmt="s", annot_kws={'fontsize':16},# cbar_kws={"orientation": "horizontal"}, 
               vmin=-1, vmax=1, center= 0, cbar=False)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
    plt.tick_params(labelsize=16)
    plt.xlabel("Predictor", fontsize=16)
    plt.ylabel("Hazard", fontsize=16)

    # colorbar
    cbar = fig.colorbar(ax.get_children()[0], cax=cax, orientation="horizontal")
    cbar.ax.tick_params(labelsize=16) 
    plt.show()
    

def hazard_accuracy(ids, num, results_path, hazard_words_per_doc, preprocessed_df, text_col, id_col, seed=0):
    hazards = [hazard for hazard in ids]
    sampled_hazard_ids = {hazard:[] for hazard in ids}
    num_total_ids = {hazard:0 for hazard in ids}
    docs = preprocessed_df[id_col].tolist()
    document_text = preprocessed_df[text_col].tolist()
    data = {}
    for hazard in ids:
        total_ids = [id_ for year in ids[hazard] for id_ in ids[hazard][year]]
        sampled_ids = random.sample(total_ids, min(num, len(total_ids)))
        sampled_hazard_ids[hazard] = sampled_ids
        num_total_ids[hazard] = len(total_ids)
        
        hazard_words = [hazard_words_per_doc[hazard][docs.index(id_)] for id_ in sampled_ids]
        doc_text = [document_text[docs.index(id_)] for id_ in sampled_ids]
        data[hazard] = pd.DataFrame({"ID": sampled_ids,
                                     "Contains Hazard": [0 for id_ in sampled_ids],
                                     "Hazard Word": hazard_words,
                                     "Document Text": doc_text})
    data["Summary"] = pd.DataFrame({"Hazards": [hazard for hazard in ids],
                       "# Total IDs": [num_total_ids[hazard] for hazard in num_total_ids],
                       "# Sampled IDs": [len(sampled_hazard_ids[hazard]) for hazard in sampled_hazard_ids],
                       "% Sampled": [len(sampled_hazard_ids[hazard])/num_total_ids[hazard] for hazard in num_total_ids],
                       "# Correct Sampled IDs": [0 for hazard in ids],
                       "Accuracy": [0 for hazard in ids]
                       })
    #correct sampled IDs is in col E -> set this equal to the sum of Hazard sheet, B1:num for each hazard
    # SUM() =SUM(B2:B16)
    #'Aerial Grounding'!B12
    # accuracy = correct sampled/ total sampled = E/C Accuracy worksheet.write_formula('F','=QUOTIENT(E:E,C:C)')
    with pd.ExcelWriter(results_path+"/hazard_extraction_accuracy.xlsx", engine='xlsxwriter') as writer:
            for results in data:
                if len(results)>31:
                    sheet_name = results[:30]
                else: 
                    sheet_name = results
                data[results].to_excel(writer, sheet_name = sheet_name, index = False)
                if results == "Summary":
                    worksheet = writer.sheets['Summary']
                    for i in range(len(hazards)):
                        worksheet.write_formula('E'+str(i+2), '{='+"'"+hazards[i]+"'"+'!B'+str(num+2)+'}')
                        worksheet.write_formula('F'+str(i+2),'{=QUOTIENT(E'+str(i+2)+',C'+str(i+2)+')}')
                else:
                    worksheet = writer.sheets[sheet_name]
                    worksheet.write('B'+str(num+2),'{=SUM(B2:B'+str(num+1)+')}')
    return sampled_hazard_ids, total_ids

def get_likelihoods(rates):
    curr_likelihoods = {hazard:0 for hazard in rates}
    for hazard in rates:
        r = rates[hazard]
        if r>=100:
            likelihood = 'Frequent'
        elif r>=10 and r<100:
            likelihood = 'Probable'
        elif r>=1 and r<10:
            likelihood = 'Remote'
        elif r>=0.1 and r<1:
            likelihood = 'Extremely Remote'
        elif r<0.1:
            likelihood = 'Extremely Improbable'
        curr_likelihoods[hazard] = likelihood
    return curr_likelihoods

def get_severities(severities):
    curr_severities = {hazard:0 for hazard in severities}
    for hazard in severities:
        s = severities[hazard]
        if s<=0.1: #negligible impact
            severity = 'Minimal Impact'
        elif s>0.1 and s <= 0.5:
            severity = 'Minor Impact'
        elif s>0.5 and s<=1:
            severity = 'Major Impact'
        elif s>1 and s<=2:
            severity = 'Hazardous Impact'
        elif s>2:
            severity = 'Catastrophic Impact'
        curr_severities[hazard] = severity
    return curr_severities

def plot_risk_matrix(rates, severities, figsize=(9,5), save=False):
    hazards = [h for h in rates]
    curr_likelihoods = get_likelihoods(rates)
    curr_severities = get_severities(severities)
    annotation_df = pd.DataFrame([["" for i in range(5)] for j in range(5)],
                         columns=['Minimal Impact', 'Minor Impact', 'Major Impact', 'Hazardous Impact', 'Catastrophic Impact'],
                          index=['Frequent', 'Probable', 'Remote','Extremely Remote', 'Extremely Improbable'])
    #hazards_per_row_df = pd.DataFrame([[0 for i in range(5)] for j in range(5)],
    #                 columns=['Minimal Impact', 'Minor Impact', 'Major Impact', 'Hazardous Impact', 'Catastrophic Impact'],
    #                  index=['Frequent', 'Probable', 'Remote','Extremely Remote', 'Extremely Improbable'])
    #rows = pd.DataFrame([[0 for i in range(5)] for j in range(5)],
    #                 columns=['Minimal Impact', 'Minor Impact', 'Major Impact', 'Hazardous Impact', 'Catastrophic Impact'],
    #                  index=['Frequent', 'Probable', 'Remote','Extremely Remote', 'Extremely Improbable'])
    annot_font = 12
    hazard_likelihoods = {hazard:"" for hazard in hazards}; hazard_severities={hazard:"" for hazard in hazards}
    for hazard in hazards:
        hazard_likelihoods[hazard] = curr_likelihoods[hazard]
        hazard_severities[hazard] = curr_severities[hazard]
        new_annot = annotation_df.at[curr_likelihoods[hazard], curr_severities[hazard]]
        if new_annot != "": new_annot += ", "
        hazard_annot = hazard.split(" ")
        #if line>20 then new line
        if len(new_annot.split("\n")[-1]) + len(hazard_annot[0]) < 20:
            new_annot += hazard_annot[0]
            annot_ind = 1
        elif len(hazard_annot[1]) + len(hazard_annot[0]) < 20:
            new_annot += "\n" + hazard_annot[0] + " " + hazard_annot[1]
            annot_ind = 2
        else:
            new_annot += "\n" + hazard_annot[0]
            annot_ind = 1
        if len(hazard_annot)>1:
            new_annot += "\n"+" ".join(hazard_annot[annot_ind:])
        annotation_df.at[curr_likelihoods[hazard], curr_severities[hazard]] = new_annot #+= (str(hazard_annot))
    
    df = pd.DataFrame([[0, 5, 10, 10, 10], [0, 5, 5, 10, 10], [0, 0, 5, 5, 10],
            [0, 0, 0, 5, 5], [0, 0, 0, 0, 5]],
          columns=['Minimal Impact', 'Minor Impact', 'Major Impact', 'Hazardous Impact', 'Catastrophic Impact'],
           index=['Frequent', 'Probable', 'Remote','Extremely Remote', 'Extremely Improbable'])
    fig,ax = plt.subplots(figsize=figsize)
    #annot df has hazards in the cell they belong to #annot=annotation_df
    sn.heatmap(df, annot=annotation_df, fmt='s',annot_kws={'fontsize':annot_font},cbar=False,cmap='RdYlGn_r')
    plt.title("Risk Matrix", fontsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)#, ha="right")
    plt.tick_params(labelsize=12)
    plt.ylabel("Likelihood", fontsize=12)
    plt.xlabel("Severity", fontsize=12)
    minor_ticks = np.arange(1, 6, 1)
    ax.set_xticks(minor_ticks, minor=True)
    ax.set_yticks(minor_ticks, minor=True)
    ax.tick_params(which='minor',length=0, grid_color='black', grid_alpha=1)
    ax.grid(which='minor', alpha=1)
    if save: 
        file_path = os.path.join(os.path.dirname(os.getcwd()),'results','risk_matrices', "SAFECOM_static_rm")
        plt.savefig(file_path+".pdf", bbox_inches="tight")
    plt.show()
