# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 12:14:51 2021

@author: srandrad
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, accuracy_score
import seaborn as sn
from tqdm import tqdm

def normalize_data(df, features, years, flag='minmax'):  
    num_features = len(features)
    
    if flag == 'standard':
        mean_vec = np.zeros(num_features)
        std_vec = np.zeros(num_features)
        
        for i in range(num_features):
            mean_vec[i] = np.mean(df.iloc[:][features[i]])
            std_vec[i] = np.std(df.iloc[:][features[i]])

        df_norm = np.zeros(np.shape(df))
        
        for j in range(num_features):
            if std_vec[j] == 0:
                df_norm[:, j] = 0
            else:
                for i in range(np.shape(df)[0]):
                   df_norm[i, j] = (df.iloc[i][features[j]] - mean_vec[j]) / std_vec[j]
                
    elif flag == 'minmax':
        for year in years:
            max_vec = np.zeros(num_features)
            min_vec = np.zeros(num_features)
            
            for i in range(num_features):
                max_vec[i] = np.max(df.loc[df['START_YEAR']==year].iloc[:][features[i]]) #df.loc[df['START_YEAR']==year]
                min_vec[i] = np.min(df.loc[df['START_YEAR']==year].iloc[:][features[i]])
    
            #df_norm = np.zeros(np.shape(df))
            
            for j in range(num_features):
                if (max_vec[j] - min_vec[j]) == 0:
                    df.loc[:][j] = 0
                else:
                    for i in range(np.shape(df)[0]):
                        df.iloc[i][features[j]] = (df.iloc[i][features[j]] - min_vec[j]) / (max_vec[j] - min_vec[j])
                    
    return df

def minmax_scale(data_list):
    max_ = max(data_list)
    min_ = min(data_list)
    scaled_list = []
    for data in data_list:
        scaled_data = (data-min_)/max_
        scaled_list.append(scaled_data)
    return scaled_list

def corr_sig(df=None):
    p_matrix = np.zeros(shape=(df.shape[1],df.shape[1]))
    for col in df.columns:
        for col2 in df.drop(col,axis=1).columns:
            _ , p = stats.pearsonr(df[col],df[col2])
            p_matrix[df.columns.to_list().index(col),df.columns.to_list().index(col2)] = p
    return p_matrix

def multiple_reg_feature_importance(predictors, hazards, correlation_mat_total):
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
    for hazard in hazards:
        X = data[predictors]
        y = data[hazard]
        regr = linear_model.LinearRegression()
        regr.fit(X,y)
        y_pred = regr.predict(X)
        model_score = regr.score(X,y)
        full_model_score.append(model_score)
        model_MSE = mean_squared_error(y_pred,y)
        full_model_MSE.append(model_MSE)
        for predictor in predictors:
            other_predictors = [pre for pre in predictors if pre!=predictor]
            X = data[other_predictors]
            y = data[hazard]
            regr = linear_model.LinearRegression()
            regr.fit(X,y)
            y_pred = regr.predict(X)
            predictor_score = regr.score(X,y)
            xi_score[predictor+" removed score"].append(predictor_score)
            predictor_MSE = mean_squared_error(y_pred,y)
            xi_MSE[predictor+" removed MSE"].append(predictor_MSE)
            
            xi_score_delta[predictor+" removed score"].append(model_score-predictor_score)
            xi_MSE_delta[predictor+" removed MSE"].append(model_MSE-predictor_MSE)
    
    results_data = {"hazard":hazards, "R2 for full model":full_model_score, "MSE for full model": full_model_MSE}
    results_data.update(xi_score)
    results_data.update(xi_MSE)
    results_df = pd.DataFrame(results_data)
    
    delta_data = {"hazard":hazards, "R2 for full model":full_model_score, "MSE for full model": full_model_MSE}
    delta_data.update(xi_score_delta)
    delta_data.update(xi_MSE_delta)
    delta_df = pd.DataFrame(delta_data)
    return results_df, delta_df

def check_rates():
    return 

def remove_outliers(data, threshold=1.5):
    #print(data)
    if data == []:
        return data
    Q1 = np.quantile(data,0.25)
    Q3 = np.quantile(data,0.75)
    IQR = Q3 - Q1
    new_data = [pt for pt in data if (pt>(Q1-1.5*IQR)) and (pt<(Q3+1.5*IQR))]
    #print(len(data), len(new_data))
    return new_data

def calc_metrics(hazard_file, years, preprocessed_df, rm_outliers=True):
     hazard_info = pd.read_excel(hazard_file, sheet_name=['Hazard-focused'])
     hazards = hazard_info['Hazard-focused']['Hazard name'].tolist()
     categories = hazard_info['Hazard-focused']['Hazard Category'].tolist()
     time_of_occurence_days = {name:{str(year):[] for year in years} for name in hazards}
     time_of_occurence_pct_contained = {name:{str(year):[] for year in years} for name in hazards}
     frequency = {name:{str(year):0 for year in years} for name in hazards}
     fires = {name:{str(year):[] for year in years} for name in hazards}
     years = preprocessed_df["START_YEAR"].unique()
     years.sort()
     for year in tqdm(years):
        temp_df = preprocessed_df.loc[preprocessed_df["START_YEAR"]==year].reset_index(drop=True)
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
                #print(text, type(text))
                #check for hazard
                for i in range(len(hazard_info['Hazard-focused'])):
                    hazard_name = hazard_info['Hazard-focused'].iloc[i]['Hazard name']
                    hazard_subject_words = hazard_info['Hazard-focused'].iloc[i]['Hazard Noun/Subject']
                    #hazard_subject_words = list(hazard_subject_words.split("; "))
                    hazard_subject_words = hazard_subject_words.split(", ")#[list_.split(", ") for list_ in hazard_subject_words]
                    hazard_action_words = hazard_info['Hazard-focused'].iloc[i]['Action/Descriptor']
                    #hazard_action_words = list(hazard_action_words.split("; "))
                    hazard_action_words = hazard_action_words.split(", ")#[list_.split(", ") for list_ in hazard_action_words]
                    negation_words = hazard_info['Hazard-focused'].iloc[i]['Negation words']
                    #need to check if a word in text is in hazard words, for each list in hazard words, no words in negation words
                    #print(hazard_words)
                    hazard_words = [hazard_subject_words, hazard_action_words]
                    #print(hazard_words)
                    for word_list in hazard_words:
                        hazard_found = False #ensures a word from each list must show up
                        for word in word_list:
                            if word in text:
                                hazard_found = True
                        if hazard_found == False: #ensures a word from each list must show up
                            break 
                    
                    if not np.isnan(negation_words):
                        for word in negation_words.split(", "): #removes texts that have negation words
                            if word in text:
                                hazard_found = False 
                    
                    if hazard_found == True:
                        time_of_hazard = int(temp_fire_df.iloc[j]["REPORT_DOY"])
                    
                        #correct dates
                        if time_of_hazard<start_date: 
                            #print(time_of_hazard, start_date)
                            if time_of_hazard<30: #report day is days since start, not doy 
                                time_of_hazard+=start_date
                            else: #start and report day were incorrectly switched
                                #print(time_of_hazard, start_date)
                                temp_start = start_date
                                start_date = time_of_hazard
                                time_of_hazard = temp_start
                                #print(time_of_hazard, start_date)
                        time_of_occurence_days[hazard_name][str(year)].append(time_of_hazard-int(start_date))
                        time_of_occurence_pct_contained[hazard_name][str(year)].append(temp_fire_df.iloc[j]["PCT_CONTAINED_COMPLETED"])
                        fires[hazard_name][str(year)].append(id_)
                        frequency[hazard_name][str(year)] += 1
     if rm_outliers == True:
         for year in years:
             for hazard in hazards:
                 time_of_occurence_days[hazard][str(year)] = remove_outliers(time_of_occurence_days[hazard][str(year)])
                 time_of_occurence_pct_contained[hazard][str(year)] = remove_outliers(time_of_occurence_pct_contained[hazard][str(year)])
     return time_of_occurence_days, time_of_occurence_pct_contained, frequency, fires, categories, hazards


def create_primary_results_table(time_of_occurence_days, time_of_occurence_pct_contained, frequency, fires, hazard_df, categories, hazards, years, interval=False):
    ##NOTE: Be wary of frequency and rate-> total frequency is different than number of fires!!
    data_df = {"Hazard Category": categories,
               "Hazard Name": hazards}
    #OTTO days average += std dev; interval
    days_total_data = {}
    for hazard in hazards:
        days_total_data[hazard] = []
        for year in years:
            for val in time_of_occurence_days[hazard][year]:
                days_total_data[hazard].append(val)
    days_av = {hazard: np.average(days_total_data[hazard]) for hazard in hazards}
    #print(days_av)
    #print(days_total_data[hazards[0]])
    days_std = {hazard: np.std(days_total_data[hazard]) for hazard in hazards}
    data_df["OTTO days"] = [str(days_av[hazard])+"+-"+str(days_std[hazard]) for hazard in hazards]
    data_df["OTTO days interval"] = [stats.t.interval(alpha=0.95, df=len(days_total_data[hazard])-1, loc=np.mean(days_total_data[hazard]), scale=stats.sem(days_total_data[hazard])) for hazard in hazards]
    #OTTO percent average += std dev; interval
    pct_total_data = {}
    for hazard in hazards:
        pct_total_data[hazard] = []
        for year in years:
            for val in time_of_occurence_pct_contained[hazard][year]:
                pct_total_data[hazard].append(val)
    pct_av = {hazard: np.average(pct_total_data[hazard]) for hazard in hazards}
    pct_std = {hazard: np.std(pct_total_data[hazard]) for hazard in hazards}
    data_df["OTTO %"] = [str(pct_av[hazard])+"+-"+str(pct_std[hazard]) for hazard in hazards]
    data_df["OTTO % interval"] = [stats.t.interval(alpha=0.95, df=len(pct_total_data[hazard])-1, loc=np.mean(pct_total_data[hazard]), scale=stats.sem(pct_total_data[hazard])) for hazard in hazards]
    #total frequency
    data_df["Total Frequency"] = [np.sum([frequency[hazard][year] for year in years]) for hazard in hazards]
    #rate per year
    sums_per_hazard = [np.sum([len(set(fires[hazard][year])) for year in years]) for hazard in hazards]
    data_df["Total Fire Frequency"] = sums_per_hazard
    data_df["Average Occurrences per year"] = [val/len(years) for val in sums_per_hazard]
    #fires per rate
    total_fires = total_fires = len(hazard_df["INCIDENT_ID"].unique())
    data_df["Average fires per occurrence"] = [total_fires/val for val in sums_per_hazard]
    
    return data_df

def create_metrics_time_series(time_of_occurence_days, time_of_occurence_pct_contained, frequency, fire, years, combined=True):
    days_averages = {hazard: [np.average(time_of_occurence_days[hazard][year]) for year in time_of_occurence_days[hazard]] for hazard in time_of_occurence_days}
    days_stddevs = {hazard: [np.std(time_of_occurence_days[hazard][year]) for year in time_of_occurence_days[hazard]] for hazard in time_of_occurence_days}
    pct_averages = {hazard: [np.average(time_of_occurence_pct_contained[hazard][year]) for year in time_of_occurence_pct_contained[hazard]] for hazard in time_of_occurence_pct_contained}
    pct_stddevs = {hazard: [np.std(time_of_occurence_pct_contained[hazard][year]) for year in time_of_occurence_pct_contained[hazard]] for hazard in time_of_occurence_pct_contained}
    #plot OTTO
    if combined == True:
        plt.figure()
        plt.title("Change in Operational Time of Occurence for Hazards from 2006-2014")
        plt.xlabel("Year")
        plt.ylabel("% Containment")
        for hazard in pct_averages:
            plt.errorbar(years, pct_averages[hazard], yerr=pct_stddevs[hazard], label=hazard)
        plt.legend()
        plt.show()
        
        plt.figure()
        plt.title("Change in Operational Time of Occurence for Hazards from 2006-2014")
        plt.xlabel("Year")
        plt.ylabel("Days")
        for hazard in pct_averages:
            plt.errorbar(years, days_averages[hazard], yerr=days_stddevs[hazard], label=hazard)
        plt.legend()
        plt.show()
    elif combined == False:
        for hazard in pct_averages:
            plt.figure()
            plt.title("Change in Operational Time of Occurence for \n"+hazard+" from 2006-2014")
            plt.xlabel("Year")
            plt.ylabel("% Containment")
            plt.errorbar(years, pct_averages[hazard], yerr=pct_stddevs[hazard], label=hazard)
            plt.legend()
            plt.show()
        for hazard in pct_averages:
            plt.figure()
            plt.title("Change in Operational Time of Occurence for \n"+hazard+" from 2006-2014")
            plt.xlabel("Year")
            plt.ylabel("Days")
            plt.errorbar(years, days_averages[hazard], yerr=days_stddevs[hazard], label=hazard)
            plt.legend()
            plt.show()
    #plot frequency
    frequencies = {hazard: [frequency[hazard][year] for year in years] for hazard in frequency}
    hazard_freqs_scaled = {hazard: minmax_scale(frequencies[hazard]) for hazard in frequencies}
    if combined == True:
        plt.figure()
        plt.ylabel("Total Scaled")
        plt.xlabel("Year")
        plt.title("Change in Hazard Frequency from 2006-2014")
        for hazard in hazard_freqs_scaled:
            plt.plot(years, hazard_freqs_scaled[hazard], label=hazard)
        plt.legend()
        plt.show()
    elif combined == False:
        for hazard in frequencies:
            plt.figure()
            plt.ylabel("Total Scaled")
            plt.xlabel("Year")
            plt.title("Change in "+hazard+"\n Frequency from 2006-2014")
            plt.plot(years, frequencies[hazard], label=hazard)
            plt.legend()
            plt.show()
    return days_averages, days_stddevs, pct_averages, pct_stddevs, frequencies, hazard_freqs_scaled

def create_correlation_matrix(predictors_scaled, frequencies_scaled):
    correlation_mat_data = predictors_scaled
    for hazard in frequencies_scaled:
        correlation_mat_data["total " +hazard] = frequencies_scaled[hazard]
    correlation_mat_total = pd.DataFrame(correlation_mat_data)
    corrMatrix =correlation_mat_total.corr()
    p_values = corr_sig(correlation_mat_total)                     # get p-Value
    mask = np.invert(np.tril(p_values<0.05)) 
    sn.heatmap(corrMatrix, annot=True, mask=mask)
    plt.title("Correlational Matrix for Trends in \n Fires, Operations, Intensity, and Hazard Frequency per year")
    plt.show()
    
    sn.heatmap(corrMatrix, annot=True)
    plt.title("Correlational Matrix for Trends in \n Fires, Operations, Intensity, and Hazard Frequency per year")
    plt.show()
    
    return correlation_mat_total