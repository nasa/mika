# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 12:14:51 2021

@author: srandrad
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import scipy.stats as stats
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score
import seaborn as sn
from tqdm import tqdm

def normalize_data(df, features, years, flag='minmax'):  
    ##NOTE: THIS IS UNUSED
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
        #print(importance)
        # summarize feature importance
        for i,v in enumerate(importance[0]):
            #print(i)
            #print(v)
            print('Feature: %0d, Score: %.5f' % (i,v))
        # plot feature importance
        plt.bar([x for x in range(len(importance[0]))], importance[0])
        plt.show()
        
        for predictor in predictors:
            """other_predictors = [pre for pre in predictors if pre!=predictor]
            X = data[other_predictors]
            y = data[hazard]
            regr = linear_model.LinearRegression()
            regr.fit(X,y)
            y_pred = regr.predict(X)
            predictor_score = regr.score(X,y)
            """
            index_list = np.random.RandomState(seed=42).permutation(len(data[predictors]))
            X = {}
            y = data[hazard]
            for i in range(len(predictors)):
                if predictors[i] == predictor:
                    #print(data.loc[:,feature])
                    shuffled = [data.iloc[i][predictor] for i in index_list]
                    #print(shuffled)
                    X[predictor] = shuffled
                else:
                    X[predictors[i]] = data.loc[:, predictors[i]]
            regr = linear_model.LinearRegression()
            X_new = pd.DataFrame(X)[predictors]
            X = data[predictors]
            regr.fit(X,y)
            y_pred = regr.predict(X_new)
            predictor_score = r2_score(y, y_pred)
            if predictor_score > 1: print(predictor)
            
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
    #print(results_data)
    #print(delta_data)
    #print(results_df.to_dict())
    #print(delta_df.to_dict())
    
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
     """
    uses the hazard focused sheet in the hazard interpretation results to calculate metrics.
    hazards are identified based on subject-action pairs 
    ARGUMENTS
     ---------
     hazard_file: string
         the location of the interpretation xlsx file
     years: list
         the years the data spans TODO: remove this 
     preprocessed_df: string
         the location of preprocessed data that was fed into the model
     rm_outliers: boolean 
         used to remove outliers or not
     """
     hazard_info = pd.read_excel(hazard_file, sheet_name=['Hazard-focused'])
     hazards = hazard_info['Hazard-focused']['Hazard name'].tolist()
     categories = hazard_info['Hazard-focused']['Hazard Category'].tolist()
     time_of_occurence_days = {name:{str(year):[] for year in years} for name in hazards}
     time_of_occurence_pct_contained = {name:{str(year):[] for year in years} for name in hazards}
     frequency = {name:{str(year):0 for year in years} for name in hazards}
     fires = {name:{str(year):[] for year in years} for name in hazards}
     frequency_fires ={name:{str(year):0 for year in years} for name in hazards}
     years = preprocessed_df["START_YEAR"].unique()
     years.sort()
     #print(years, type(years[0]))
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
                #check for hazard
                for i in range(len(hazard_info['Hazard-focused'])):
                    hazard_name = hazard_info['Hazard-focused'].iloc[i]['Hazard name']
                    hazard_subject_words = hazard_info['Hazard-focused'].iloc[i]['Hazard Noun/Subject']
                    hazard_subject_words = hazard_subject_words.split(", ")
                    hazard_action_words = hazard_info['Hazard-focused'].iloc[i]['Action/Descriptor']
                    hazard_action_words = hazard_action_words.split(", ")
                    negation_words = hazard_info['Hazard-focused'].iloc[i]['Negation words']
                    #need to check if a word in text is in hazard words, for each list in hazard words, no words in negation words
                    #print(hazard_words)
                    hazard_words = [hazard_subject_words, hazard_action_words]
                    for word_list in hazard_words:
                        hazard_found = False #ensures a word from each list must show up
                        for word in word_list:
                            if word in text:
                                hazard_found = True
                        if hazard_found == False: #ensures a word from each list must show up
                            break 
                    
                    if isinstance(negation_words,str):
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
                        #if hazard_name == 'Law Violations' and year in [2008.0, 2009.0, 2010.0, 2011.0]:
                        #    print(year)
                        #    print(temp_fire_df.iloc[j]["PCT_CONTAINED_COMPLETED"], id_)
                        #    print(hazard_found)
                        time_of_occurence_days[hazard_name][str(float(year))].append(time_of_hazard-int(start_date))
                        time_of_occurence_pct_contained[hazard_name][str(float(year))].append(temp_fire_df.iloc[j]["PCT_CONTAINED_COMPLETED"])
                        fires[hazard_name][str(float(year))].append(id_)
                        frequency[hazard_name][str(float(year))] += 1
     for name in frequency_fires:
         for year in frequency_fires[name]:
             frequency_fires[name][year] = len(set(fires[name][year]))
     if rm_outliers == True:
         for year in years:
             for hazard in hazards:
                 if hazard != 'Law Violations':
                     time_of_occurence_days[hazard][str(float(year))] = remove_outliers(time_of_occurence_days[hazard][str(year)])
                     time_of_occurence_pct_contained[hazard][str(float(year))] = remove_outliers(time_of_occurence_pct_contained[hazard][str(year)])
     return time_of_occurence_days, time_of_occurence_pct_contained, frequency, fires, frequency_fires, categories, hazards



def topic_based_calc_metrics(hazard_file, years, preprocessed_df, results_file, rm_outliers=True):
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
     hazards = hazard_info['topic-focused']['Hazard name'].tolist()
     categories = hazard_info['topic-focused']['Hazard Category'].tolist()
     results = pd.read_excel(results_file, sheet_name=['Combined Text'])
     
     time_of_occurence_days = {name:{str(year):[] for year in years} for name in hazards}
     time_of_occurence_pct_contained = {name:{str(year):[] for year in years} for name in hazards}
     frequency = {name:{str(year):0 for year in years} for name in hazards}
     fires = {name:{str(year):[] for year in years} for name in hazards}
     years = preprocessed_df["START_YEAR"].unique()
     years.sort()
     
     for i in range(len(hazards)):
         num = hazard_info['topic-focused'].iloc[i]['Topic Number']
         ids_ = results['Combined Text'].iloc[num]['documents']
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
                for i in range(len(hazard_info['topic-focused'])):
                    hazard_name = hazard_info['topic-focused'].iloc[i]['Hazard name']
                    hazard_words = hazard_info['topic-focused'].iloc[i]['Relevant hazard words']
                    hazard_words = hazard_words.split(", ")
                    negation_words = hazard_info['topic-focused'].iloc[i]['Negation words']
                    #need to check if a word in text is in hazard words
                    hazard_found = False
                    for word in hazard_words:
                        if word in text:
                            hazard_found = True
                    
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
                        year = temp_fire_df.iloc[j]["START_YEAR"]
                        time_of_occurence_days[hazard_name][str(float(year))].append(time_of_hazard-int(start_date))
                        time_of_occurence_pct_contained[hazard_name][str(float(year))].append(temp_fire_df.iloc[j]["PCT_CONTAINED_COMPLETED"])
                        fires[hazard_name][str(float(year))].append(id_)
                        frequency[hazard_name][str(float(year))] += 1
     if rm_outliers == True:
         for year in years:
             for hazard in hazards:
                 time_of_occurence_days[hazard][str(float(year))] = remove_outliers(time_of_occurence_days[hazard][str(year)])
                 time_of_occurence_pct_contained[hazard][str(float(year))] = remove_outliers(time_of_occurence_pct_contained[hazard][str(year)])
     return time_of_occurence_days, time_of_occurence_pct_contained, frequency, fires, categories, hazards



def create_primary_results_table(time_of_occurence_days, time_of_occurence_pct_contained, frequency, fires, preprocessed_df, categories, hazards, years, interval=False):
    ##NOTE: Be wary of frequency and rate-> total frequency is different than number of fires!!
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
        for year in years:
            for val in time_of_occurence_days[hazard][year]:
                days_total_data[hazard].append(val)
    days_av = {hazard: np.average(days_total_data[hazard]) for hazard in hazards}
    #print(days_av)
    #print(days_total_data[hazards[0]])
    days_std = {hazard: np.std(days_total_data[hazard]) for hazard in hazards}
    data_df["OTTO days"] = [str(round(days_av[hazard],3))+"+-"+str(round(days_std[hazard],3)) for hazard in hazards]
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
    data_df["OTTO %"] = [str(round(pct_av[hazard],3))+"+-"+str(round(pct_std[hazard],3)) for hazard in hazards]
    data_df["OTTO % interval"] = [stats.t.interval(alpha=0.95, df=len(pct_total_data[hazard])-1, loc=np.mean(pct_total_data[hazard]), scale=stats.sem(pct_total_data[hazard])) for hazard in hazards]
    #total frequency
    data_df["Total Frequency"] = [np.sum([frequency[hazard][year] for year in years]) for hazard in hazards]
    #rate per year
    sums_per_hazard = [np.sum([len(set(fires[hazard][year])) for year in years]) for hazard in hazards]
    data_df["Total Fire Frequency"] = sums_per_hazard
    data_df["Average Occurrences per year"] = [round(val/len(years),3) for val in sums_per_hazard]
    #fires per rate
    total_fires = len(preprocessed_df["INCIDENT_ID"].unique())
    data_df["Average fires per occurrence"] = [round(total_fires/val,3) for val in sums_per_hazard]
    
    return data_df

def create_metrics_time_series(time_of_occurence_days, time_of_occurence_pct_contained, frequency, frequency_fires, fire, years, combined=True):
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
    marker_styles = ['.', 'v', '^', 's', 'D', 'X']
    #print(type(years[0]))
    years_plot = [year.strip('0').strip('.') for year in years]
   
    #plot OTTO
    if combined == True:
        plt.figure()
        #plt.title("Change in Operational Time of Occurence for Hazards from 2006-2014")
        plt.xlabel("Year", fontsize=16)
        plt.ylabel("% Containment", fontsize=16)
        i=0
        line_counter = 0
        marker_counter = 0
        for hazard in pct_averages:
            plt.errorbar(years_plot, pct_averages[hazard], yerr=pct_stddevs[hazard], color=colors[i], marker=marker_styles[marker_counter%len(marker_styles)], linestyle=line_styles[line_counter], label=hazard, capsize=5, markeredgewidth=1)
            i += 1
            marker_counter += 1
            if i%len(marker_styles) == 0:
                line_counter += 1
            
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
        for hazard in pct_averages:
            plt.errorbar(years_plot, days_averages[hazard], yerr=days_stddevs[hazard], color=colors[i], marker=marker_styles[marker_counter%len(marker_styles)], linestyle=line_styles[line_counter], label=hazard, capsize=5, markeredgewidth=1)
            i += 1 
            marker_counter += 1
            if i%len(marker_styles) == 0:
                line_counter += 1
            
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
    frequencies = {hazard: [frequency[hazard][year] for year in years] for hazard in frequency}
    hazard_freqs_scaled = {hazard: minmax_scale(frequencies[hazard]) for hazard in frequencies}
    frequencies_fire = {hazard: [frequency_fires[hazard][year] for year in years] for hazard in frequency_fires}
    fire_freqs_scaled = {hazard: minmax_scale(frequencies_fire[hazard]) for hazard in frequencies_fire}
    if combined == True:
        plt.figure()
        plt.ylabel("Total Scaled", fontsize=16)
        plt.xlabel("Year", fontsize=16)
        #plt.title("Change in Hazard Frequency from 2006-2014")
        i = 0
        line_counter = 0
        marker_counter = 0
        for hazard in hazard_freqs_scaled:
            plt.plot(years_plot, hazard_freqs_scaled[hazard], color=colors[i], label=hazard, marker=marker_styles[marker_counter%len(marker_styles)], linestyle=line_styles[line_counter])
            i += 1
            marker_counter += 1
            if i%len(marker_styles) == 0:
                line_counter += 1
            
        plt.legend(bbox_to_anchor=(1, 1.1), loc='upper left', fontsize=14)
        plt.tick_params(labelsize=16)
        plt.show()
        
        plt.figure()
        plt.ylabel("Total Scaled", fontsize=16)
        plt.xlabel("Year", fontsize=16)
        #plt.title("Change in Hazard Frequency from 2006-2014")
        i = 0
        line_counter = 0
        marker_counter = 0
        for hazard in hazard_freqs_scaled:
            plt.plot(years_plot, fire_freqs_scaled[hazard], color=colors[i], label=hazard, marker=marker_styles[marker_counter%len(marker_styles)], linestyle=line_styles[line_counter])
            i += 1
            marker_counter += 1
            if i%len(marker_styles) == 0:
                line_counter += 1
            
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

def create_correlation_matrix(predictors_scaled, frequencies_scaled):
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
    sn.heatmap(corrMatrix, annot=True, mask=mask)
    plt.title("Correlational Matrix for Trends in \n Fires, Operations, Intensity, and Hazard Frequency per year")
    plt.show()
    
    sn.heatmap(corrMatrix, annot=True)
    plt.title("Correlational Matrix for Trends in \n Fires, Operations, Intensity, and Hazard Frequency per year")
    plt.show()
    #print(correlation_mat_data)
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
    sn.set(font_scale=1.5)
    sn.heatmap(new_corr_df, annot=annotation_df, annot_kws={'fontsize':16}, fmt="s",# cbar_kws={"orientation": "horizontal"}, 
               vmin=-1, vmax=1, center= 0, cbar=False)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
    plt.xlabel("Predictor")
    plt.ylabel("Hazard")

    # colorbar
    fig.colorbar(ax.get_children()[0], cax=cax, orientation="horizontal")

    plt.show()
    
