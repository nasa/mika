# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 12:14:51 2021

@author: srandrad
"""
import numpy as np
import pandas as pd
import scipy.stats as stats

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

def multiple_reg(hazard_freq, totals):
    return

def check_rates():
    return 