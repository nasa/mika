# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 12:14:51 2021

@author: srandrad
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors
import os
from nltk.corpus import words
from nltk.stem.porter import PorterStemmer
import scipy.stats as stats
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score, recall_score, precision_score, f1_score
from wordcloud import WordCloud
import seaborn as sn
from tqdm import tqdm
import pingouin as pg
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
    """
    Performs feature importance for a set of linear regression analyses

    Parameters
    ----------
    predictors: list
        list of predictor names, used to identify inputs to single linear regression
    hazards: list
        list of hazard names, used to identify targets for single linear regression
    correlation_mat_total: dataframe
        stores the time series values that were used for correlation matrix. 
        rows are years, columns are predictors + hazard frequencies
    Returns
    -------
    results_df : TYPE
        DESCRIPTION.

    """
    data = correlation_mat_total
    predictors = predictors
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
        plt.bar(X_axis+(width*i), importance_data[hazard], width, label=hazard, color=colors[i-1])
        i+=1
    plt.xticks(X_axis+(width*np.ceil(num_bars/2)), [pred.replace("total ","") for pred in predictors], rotation=70)
    plt.tick_params(labelsize=14)
    plt.xlabel("Predictors", fontsize=14)
    plt.ylabel("Importance", fontsize=14)
    plt.legend(bbox_to_anchor=(1, 1.1), loc='upper left', fontsize=14)
    plt.show()
    
    return results_df

def multiple_reg_feature_importance(predictors, hazards, correlation_mat_total, save=False, results_path="", 
                                    r2_fontsize=10, r2_figsize=(3.5,4), predictor_import_fontsize=10, predictor_import_figsize=(7,4)):
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
    hazards = [h for h in hazards]
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
    #graph r2 for full model
    num_bars = len(hazards)
    width = 1/(num_bars+2)
    i=1
    colors = cm.tab20(np.linspace(0, 1, num_bars))
    plt.figure(figsize=r2_figsize)
    for hazard in importance_data:
        plt.bar((width*i), full_model_score[hazards.index(hazard)], width, label=hazard.replace("total ",""), color=colors[i-1])
        i+=1
    plt.xticks([width*i for i in range(1,num_bars+1)],hazards, rotation=90)#(width*np.ceil(num_bars/2)), ["Full Model"], rotation=70)
    plt.tick_params(labelsize=r2_fontsize)
    plt.xlabel("Hazards", fontsize=r2_fontsize)
    plt.ylabel("R2", fontsize=r2_fontsize)
    if save:
        plt.savefig(results_path+'multiple_regression_R2.pdf', bbox_inches="tight") 
    plt.show()
    #graph feature importance
    X_axis = np.arange(len(predictors))
    num_bars = len(hazards)
    width = 1/(num_bars+2)
    i=1
    colors = cm.tab20(np.linspace(0, 1, num_bars))
    plt.figure(figsize=predictor_import_figsize)#(len(hazards),4))
    for hazard in importance_data:
        plt.bar(X_axis+(width*i), importance_data[hazard], width, label=hazard.replace("total ",""), color=colors[i-1])
        i+=1
    plt.xticks(X_axis+(width*np.ceil(num_bars/2)), [pred.replace("total ","") for pred in predictors], rotation=70)
    plt.tick_params(labelsize=predictor_import_fontsize)
    plt.xlabel("Predictors", fontsize=predictor_import_fontsize)
    plt.ylabel("Coefficient", fontsize=predictor_import_fontsize)
    plt.legend(bbox_to_anchor=(1, 1.1), loc='upper left', fontsize=predictor_import_fontsize-2)
    if save:
        plt.savefig(results_path+'multiple_regression.pdf', bbox_inches="tight") 
    plt.show()
    
    coefficient_df = pd.DataFrame(importance_data, index= [pred.replace("total ","") for pred in predictors])
    return results_df, delta_df, coefficient_df

def remove_outliers(data, threshold=1.5, rm_outliers=True):
    """
    removes outliers from the dataset using inter quartile range

    Parameters
    ----------
    data : list
        list of data points
    threshold : float, optional
        Threshold for the distance outside of the interquartile range that defines an outlier. The default is 1.5.
    rm_outliers : Booleam, optional
        True to remove outliers, false to return original data. The default is True.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    if data == [] or rm_outliers == False:
        return data
    Q1 = np.quantile(data,0.25)
    Q3 = np.quantile(data,0.75)
    IQR = Q3 - Q1
    new_data = [pt for pt in data if (pt>(Q1-1.5*IQR)) and (pt<(Q3+1.5*IQR))]
    return new_data

def check_for_hazard_words(h_word, text):
    """
    checks to see if a section of text contains a hazard word

    Parameters
    ----------
    h_word : string
        hazard word
    text :string
        section of text being searched for hazard word.

    Returns
    -------
    hazard_found : boolean
        true if hazard word appears in text, false if not.

    """
    hazard_found = False
    if h_word in text:
            hazard_found = True
    return hazard_found

def check_for_negation_words(negation_words, text, h_word):
    """
    checks to see if any negation words appear within 3 words of a hazard word in a 
    specified section of text

    Parameters
    ----------
    negation_words : list
        list of negation words.
    text :string
        section of text being searched for hazard and negation words.
    h_word : string
        hazard word

    Returns
    -------
    hazard_found : boolean
        true if hazard word appears in text with no negation words, false if not
        i.e., a negation word is present.

    """
    hazard_found = True
    for word in negation_words:#.split(", "):#removes texts that have negation words
        if word in text:
            #must be within 3 of hazard word, no punctuation
            hazard_word_inds = [h_i for h_i in range(len(text)) if text.startswith(h_word, h_i)]
            negation_word_inds = [n_i for n_i in range(len(text)) if text.startswith(word, n_i)]
            if len(hazard_word_inds) > len(negation_word_inds):
                continue
            else:
                for neg_ind in negation_word_inds:
                    paired_h_ind = hazard_word_inds[np.argmin([abs(neg_ind-h_i) for h_i in hazard_word_inds])]
                    sub_text = text[min(neg_ind, paired_h_ind):max(neg_ind, paired_h_ind)]
                    if sub_text.count(" ") <= 3 and sub_text.count(".") == 0 and sub_text.count("  ") == 0:
                        hazard_found = False
                        break
    return hazard_found

def get_topics_per_doc(docs, results, results_text_field, hazards): 
    """
    finds the topics accosiated with each document    
    
    Parameters
    ----------
    docs : list
        list of document ids.
    results : pandas DataFrame
        dataframe with topic modeling results generated using topic model plus
    results_text_field : string
        column in result dataframe where topic numbers are stored for the specified text column
    hazards : list
        list of hazards

    Returns
    -------
    topics_per_doc : dict
        dictionary with keys as document ids and values as a list of topic numbers
    hazard_topics_per_doc : dict
        nested dictionary with keys as document ids. inner dictionary has hazard names as keys
        and values as a list of topics

    """
    ##TODO: Speed up
    topics_per_doc = {doc:[] for doc in docs}
    for i in range(len(results[results_text_field])):
        row_docs = [doc for doc in docs if doc in results[results_text_field].at[i,'documents']]
        for doc in row_docs:
            topics_per_doc[doc].append(int(results[results_text_field].at[i, 'topic number']))
    hazard_topics_per_doc = {doc:{hazard:[] for hazard in hazards} for doc in topics_per_doc}
    return topics_per_doc, hazard_topics_per_doc

def get_hazard_info(hazard_file):
    """
    pulls hazard information from hazard spreadsheet

    Parameters
    ----------
    hazard_file :string
        filepath to hazard interpretation spreadsheet

    Returns
    -------
    hazard_info : pandas DataFrame
        pandas dataframe with columns for hazard names, hazard words, and topics per hazard
    hazards : list
        list of hazard names

    """
    hazard_info = pd.read_excel(hazard_file, sheet_name=['topic-focused'])
    hazards = hazard_info['topic-focused']['Hazard name'].tolist()#list(set(hazard_info['topic-focused']['Hazard name'].tolist()))
    hazards = [hazard for hazard in hazards if isinstance(hazard,str)]
    return hazard_info, hazards

def get_results_info(results_file, results_text_field, text_field, doc_topic_dist_field):
    """
    pulls topic modeling results from results spreadsheet generated with topic model plus

    Parameters
    ----------
    results_file : string
        filepath to results spreadsheet
    results_text_field : string
        column in result dataframe where topic numbers are stored for the specified text column
    text_field : string
        the text field of interest in the preprocessed_df. sometimes it is different from results_text_field but it is usually the same.
    doc_topic_dist_field : string or None
        the column storing the topic distribution per document information.
        Can be ommitted, only used when a user wants to filter results so a document
        only belings to a topic if the probability is above a specified threshold.

    Returns
    -------
    results : pandas DataFrame
        dataframe with topic modeling results generated using topic model plus
    results_text_field : string
        column in result dataframe where topic numbers are stored for the specified text column
    doc_topic_distribution : pandas DataFrame
        dataframe containing topic distributions per document
    begin_nums : int
        The topic index to begin at. is -1 if using bertopic since the top level 'topic' is really
        the cluster of documents not belonging to a topic. 0 otherwise.

    """
    if results_text_field == None:
        results_text_field = text_field
    if '.csv' in results_file: #note CSV is recommended because Excel has limited spaec
        results = pd.read_csv(results_file, index_col=0)
        results = {results_text_field: results}
        doc_topic_distribution = None
    elif '.xlsx' in results_file:
        results = pd.read_excel(results_file, sheet_name=[results_text_field])
        if doc_topic_dist_field:
            doc_topic_distribution = pd.read_excel(results_file, sheet_name=[doc_topic_dist_field])[doc_topic_dist_field]
        else:
            doc_topic_distribution = None
    if results[results_text_field].at[0,'topic number'] == -1:
        begin_nums = 1
    else:
        begin_nums = 0
    return  results, results_text_field, doc_topic_distribution, begin_nums

def set_up_docs_per_hazard_vars(preprocessed_df, id_field, hazards, time_field):
    """
    instaintiates variables used to find the documents per hazard

    Parameters
    ----------
    preprocessed_df : pandas DataFrame
        pandas datframe containing documents
    id_field : string
        the column in preprocessed df that contains document ids
    hazards : list
        list of hazards
    time_field : string
        the column in preprocessed df that contains document time values, such as report year

    Returns
    -------
    docs : list
        list of document ids.
    frequency : Dict
        dictionary used to store hazard frequencies. Keys are hazards and values are ints.
    docs_per_hazard : Dict
        nested dictionary used to store documents per hazard. Keys are hazards 
        and value is an inner dict. Inner dict has keys as time variables (e.g., years) and 
        values are lists.
    hazard_words_per_doc : Dict
        used to store the hazard words per document. keys are hazards and values are lists
        with an element for each document.

    """
    docs = preprocessed_df[id_field].tolist()
    hazard_words_per_doc = {hazard:['none' for doc in docs] for hazard in hazards}
    time_period = preprocessed_df[time_field].unique()
    frequency = {name:{str(time_p):0 for time_p in time_period} for name in hazards}
    docs_per_hazard = {hazard:{str(time_p):[] for time_p in time_period} for hazard in hazards}
    return docs, frequency, docs_per_hazard, hazard_words_per_doc

def get_hazard_df(hazard_info, hazards, i):
    """
    gets hazard information for a specified hazard

    Parameters
    ----------
    hazard_info : pandas DataFrame
        pandas dataframe with columns for hazard names, hazard words, and topics per hazard
    hazards : list
        list of hazard names
    i : int
        the index of the specified hazard in hazard list.

    Returns
    -------
    hazard_df : pandas DataFrame
        dataframe containing the information (topics, words, etc) for the specified hazard
    hazard_name : string
        name of the specified hazard

    """
    hazard_name = hazards[i]
    hazard_df = hazard_info['topic-focused'].loc[hazard_info['topic-focused']['Hazard name'] == hazard_name].reset_index(drop=True)
    return hazard_df, hazard_name

def get_hazard_topics(hazard_df, begin_nums):
    """
    gets topic numbers for a hazard from the hazard df

    Parameters
    ----------
    hazard_df : pandas DataFrame
        dataframe containing the information (topics, words, etc) for the specified hazard
    begin_nums : int
        The topic index to begin at. is -1 if using bertopic since the top level 'topic' is really
        the cluster of documents not belonging to a topic. 0 otherwise.

    Returns
    -------
    nums : list
        list of topic numbers associated with the hazard

    """
    nums = [int(i)+begin_nums for nums in hazard_df['Topic Number'] for i in str(nums).split(", ")]#identifies all topics related to this hazard
    return nums

def get_hazard_doc_ids(nums, results, results_text_field, docs, doc_topic_distribution, text_field, topic_thresh, preprocessed_df, id_field):
    """
    Gets the document ids associated with a specified hazard

    Parameters
    ----------
    nums : list
        list of topic numbers associated with the hazard
    results : pandas DataFrame
        dataframe with topic modeling results generated using topic model plus
    results_text_field : string
        column in result dataframe where topic numbers are stored for the specified text column
    docs : list
        list of document ids.
    doc_topic_distribution : pandas DataFrame
        dataframe containing topic distributions per document
    text_field : string
        the column in preprocessed df that stores the text. can be different from results_text_field,
        but is usually the same
    topic_thresh : float
        the probability threshold a document must have to be considered in a topic
    preprocessed_df : pandas DataFrame
        pandas datframe containing documents
    id_field : string
        the column in preprocessed df that contains document ids

    Returns
    -------
    temp_df : pandas DataFrame
        subset of preprocessed_df only containing documents associated with the specified hazard
    ids_ : list
        list of document ids associated with the specified hazard

    """
    ids_df = results[results_text_field].loc[nums]
    ids_ = "; ".join(ids_df['documents'].to_list())
    ids_ = [id_ for id_ in docs if id_ in ids_]   
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
    return temp_df, ids_

def get_hazard_topics_per_doc(ids, topics_per_doc, hazard_topics_per_doc, hazard_name, nums, begin_nums):
    """
    gets the topics per document that are associated with a specied hazard

    Parameters
    ----------
    ids :  list
        list of document ids associated with the specified hazard
    topics_per_doc : dict
        dictionary with keys as document ids and values as a list of topic numbers
    hazard_topics_per_doc : dict
        nested dictionary with keys as document ids. inner dictionary has hazard names as keys
        and values as a list of topic. inner dictionary values are all empty lists for input variable.
    hazard_name : string
        name of specified hazard
    nums : list
        list of topic numbers associated with the hazard
    begin_nums : int
        The topic index to begin at. is -1 if using bertopic since the top level 'topic' is really
        the cluster of documents not belonging to a topic. 0 otherwise.

    Returns
    -------
    hazard_topics_per_doc : dict
        nested dictionary with keys as document ids. inner dictionary has hazard names as keys
        and values as a list of topics

    """
    #get hazard_topics
    for doc in ids:
        hazard_topics_per_doc[doc][hazard_name] = [num-begin_nums for num in nums if num-begin_nums in topics_per_doc[doc]]
    return hazard_topics_per_doc

def get_hazard_words(hazard_df):
    """
    gets the hazard words for a specified hazard df

    Parameters
    ----------
    hazard_df : pandas DataFrame
        dataframe containing the information (topics, words, etc) for the specified hazard

    Returns
    -------
    hazard_words : list
        list of hazard words

    """
    hazard_words = hazard_df['Relevant hazard words'].to_list()
    hazard_words = [word for words in hazard_words for word in words.split(", ")]
    return hazard_words

def get_negation_words(hazard_df):
    """
    gets the negation words for a specified hazard df

    Parameters
    ----------
    hazard_df : pandas DataFrame
        dataframe containing the information (topics, words, etc) for the specified hazard

    Returns
    -------
    negation_words : list
        list of negation words

    """
    negation_words = hazard_df['Negation words'].to_list()
    negation_words = [word for word in negation_words if isinstance(word, str)]
    negation_words = [word for neg_words in negation_words for word in neg_words.split(", ")]
    return negation_words

def record_hazard_doc_info(hazard_name, year, docs_per_hazard, id_, frequency, hazard_words_per_doc, docs, h_word):
    """
    saves the information for a specified document that contains a specified hazard

    Parameters
    ----------
    hazard_name : string
        name of specified hazard
    year : int or str
        year that the report occurs in
    docs_per_hazard : Dict
        nested dictionary used to store documents per hazard. Keys are hazards 
        and value is an inner dict. Inner dict has keys as time variables (e.g., years) and 
        values are lists.
    id_ : string
        id of the specified document
    frequency : Dict
        dictionary used to store hazard frequencies. Keys are hazards and values are ints.
    hazard_words_per_doc : Dict
        used to store the hazard words per document. keys are hazards and values are lists
        with an element for each document.
    docs : list
        list of document ids.
    h_word : string
        hazard word

    Returns
    -------
    docs_per_hazard : Dict
        nested dictionary used to store documents per hazard. Keys are hazards 
        and value is an inner dict. Inner dict has keys as time variables (e.g., years) and 
        values are lists.
    frequency : Dict
        dictionary used to store hazard frequencies. Keys are hazards and values are ints.
    hazard_words_per_doc : Dict
        used to store the hazard words per document. keys are hazards and values are lists
        with an element for each document.

    """
    docs_per_hazard[hazard_name][str(year)].append(id_)
    frequency[hazard_name][str(year)] += 1
    hazard_words_per_doc[hazard_name][docs.index(id_)] = h_word
    return docs_per_hazard, frequency, hazard_words_per_doc

def get_doc_time(id_, temp_df, id_field, time_field):
    """
    gets the time value for a document. usually the year of the report, but could
    also be month or any other time value.

    Parameters
    ----------
    id_ : string
        id of the specified document
    temp_df : pandas DataFrame
        subset of preprocessed_df only containing documents associated with the specified hazard
    id_field : string
        the column in preprocessed df that contains document ids
    time_field : string
        the column in preprocessed df that contains document time values, such as report year

    Returns
    -------
    year : str
        the time value for the specified document, usually a year

    """
    year = temp_df.loc[temp_df[id_field]==id_][time_field].values[0]
    return year

def get_doc_text(id_, temp_df, id_field, text_field):
    """
    gets the text for a document

    Parameters
    ----------
    id_ : string
        id of the specified document
    temp_df : pandas DataFrame
        subset of preprocessed_df only containing documents associated with the specified hazard
    id_field : string
        the column in preprocessed df that contains document ids
    text_field : string
        the column in preprocessed df that stores the text. can be different from results_text_field,
        but is usually the same

    Returns
    -------
    text : string
        the document text

    """
    text = temp_df.loc[temp_df[id_field]==id_][text_field].values[0]
    text = " ".join(text)
    return text 

def check_if_word_contained_in_other_word():
    #UNDER DEVELOPMENT
    #intended function is to identify words contained in other words, e.g., rain is contained in terrain
    #so document text with "terrain" is not actually "rain"
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

def identify_docs_per_hazard(hazard_file, preprocessed_df, results_file, text_field, time_field, id_field, results_text_field=None, doc_topic_dist_field=None, topic_thresh=0.0): 
    """
    function that uses all above functions to output the documents per hazard.

    Parameters
    ----------
    hazard_file :string
        filepath to hazard interpretation spreadsheet
    preprocessed_df : pandas DataFrame
        pandas datframe containing documents.
    results_file : string
        filepath to results spreadsheet
    text_field : string
        the text field of interest in the preprocessed_df. sometimes it is different from results_text_field but it is usually the same.
    time_field : string
        the column in preprocessed df that contains document time values, such as report year
    id_field : string
        the column in preprocessed df that contains document ids
    results_text_field : string, optional
        column in result dataframe where topic numbers are stored for the specified text column
        The default is None.
    doc_topic_dist_field : string or None, optional
        the column storing the topic distribution per document information.
        Can be ommitted, only used when a user wants to filter results so a document
        only belings to a topic if the probability is above a specified threshold. The default is None.
    topic_thresh : float, optional
        the probability threshold a document must have to be considered in a topic. The default is 0.0.

    Returns
    -------
    frequency : Dict
        dictionary used to store hazard frequencies. Keys are hazards and values are ints.
    docs_per_hazard : Dict
        nested dictionary used to store documents per hazard. Keys are hazards 
        and value is an inner dict. Inner dict has keys as time variables (e.g., years) and 
        values are lists.
    hazard_words_per_doc : Dict
        used to store the hazard words per document. keys are hazards and values are lists
        with an element for each document.
    topics_per_doc : dict
        dictionary with keys as document ids and values as a list of topic numbers
    hazard_topics_per_doc : dict
        nested dictionary with keys as document ids. inner dictionary has hazard names as keys
        and values as a list of topics

    """
    #read in hazard_info
    hazard_info, hazards = get_hazard_info(hazard_file)
    #read in results_info
    results, results_text_field, doc_topic_distribution, begin_nums = get_results_info(results_file, results_text_field, text_field, doc_topic_dist_field)
    #set up docs variables
    docs, frequency, docs_per_hazard, hazard_words_per_doc = set_up_docs_per_hazard_vars(preprocessed_df, id_field, hazards, time_field)
    #set up topics per doc
    topics_per_doc, hazard_topics_per_doc = get_topics_per_doc(docs, results, results_text_field, hazards)
    for i in tqdm(range(len(hazards))):
        hazard_df, hazard_name = get_hazard_df(hazard_info, hazards, i)
        nums = get_hazard_topics(hazard_df, begin_nums)
        temp_df, ids = get_hazard_doc_ids(nums, results, results_text_field, docs, doc_topic_distribution, text_field, topic_thresh, preprocessed_df, id_field)
        hazard_topics_per_doc = get_hazard_topics_per_doc(ids, topics_per_doc, hazard_topics_per_doc, hazard_name, nums, begin_nums)
        hazard_words = get_hazard_words(hazard_df)
        negation_words = get_negation_words(hazard_df)
        for id_ in ids:
            text = get_doc_text(id_, temp_df, id_field, text_field)
            hazard_found = False
            for h_word in hazard_words:
                hazard_found = check_for_hazard_words(h_word, text)
                if negation_words!=[] and hazard_found == True:
                    hazard_found = check_for_negation_words(negation_words, text, h_word)
                if hazard_found == True:
                    break

            if hazard_found == True:
                year = get_doc_time(id_, temp_df, id_field, time_field)
                docs_per_hazard, frequency, hazard_words_per_doc = record_hazard_doc_info(hazard_name, year, docs_per_hazard, id_, frequency, hazard_words_per_doc, docs, h_word)

    return frequency, docs_per_hazard, hazard_words_per_doc, topics_per_doc, hazard_topics_per_doc

def plot_metric_time_series(metric_data, metric_name, line_styles=[], markers=[], title="", 
                            time_name="Year", scaled=False, xtick_freq=5, show_std=True, 
                            save=False, dataset_name="", yscale=None, legend=True, figsize=(6,4),
                            fontsize=16):
    """
    plots a time series for specified metrics for all hazards (i.e., line chart)

    Parameters
    ----------
    metric_data : dict
        nested dict where keys are hazards. inner dict has time value (usually years) as keys
        and list of metrics for values.
    metric_name : string
        name of metric, e.g., severity, used for axis and saving the figure
    line_styles : list, optional
        list of line styles to use. should have one value for each hazard. The default is [].
    markers : list, optional
        list of line markers to use. should have one value for each hazard. The default is [].
    title : string, optional
        title to add to plot. The default is "".
    time_name : string, optional
        name of the time interval used to label the x axis. The default is "Year".
    scaled : boolean, optional
        true to minmax scale data, false to use raw data. The default is False.
    xtick_freq : int, optional
        the number of values per x tick, e.g., the default would go 2015, 2020, 2025. The default is 5.
    show_std : boolean, optional
        true to show std deviation on time series as error bars. The default is True.
    save : boolean, optional
        true to save the plot as a pdf. The default is False.
    dataset_name : string, optional
        name of the dataset, used for saving the plot. The default is "".
    yscale : string, optional
        yscale parameter, can be used to change scaling to log. The default is None.
    legend : boolean, optional
        true to show legend, false to hide legend. The default is True.
    figsize : tuple, optional
        size of the plot in inches. The default is (6,4).
    fontsize : int, optional
        fontsize for the plot. The default is 16.

    Returns
    -------
    None.

    """
    time_vals = list(set([year for hazard in metric_data for year in metric_data[hazard]]))
    time_vals.sort()
    #scaled -> scaled the averages, how to scale stddev?
    if scaled: metric_data = {hazard: minmax_scale(metric_data[hazard]) for hazard in metric_data}
    averages = {hazard: [np.average(metric_data[hazard][year]) for year in time_vals] for hazard in metric_data}
    stddevs = {hazard: [np.std(metric_data[hazard][year]) for year in time_vals] for hazard in metric_data}
    colors = cm.tab20(np.linspace(0, 1, len(averages)))
    plt.figure(figsize=figsize)
    plt.title(title, fontsize=fontsize)
    plt.xlabel(time_name, fontsize=fontsize)
    plt.ylabel(metric_name, fontsize=fontsize)
    if yscale == 'log':
        plt.yscale('symlog')
    i=0
    for hazard in averages:
        temp_time_vals = time_vals.copy()
        nans = np.where(np.isnan(averages[hazard]))[0]
        hazard_avs = averages[hazard]
        hazard_stddev = stddevs[hazard]
        num_removed = 0
        for ind in nans:
            temp_time_vals.pop(ind-num_removed)
            hazard_avs.pop(ind-num_removed)
            hazard_stddev.pop(ind-num_removed)
            num_removed+=1
        temp_time_vals = [int(t) for t in temp_time_vals]
        if show_std == True:
            plt.errorbar(temp_time_vals, hazard_avs, yerr=hazard_stddev, color=colors[i], marker=markers[i], linestyle=line_styles[i], label=hazard, capsize=5, markeredgewidth=1)
        else:
            plt.plot(temp_time_vals, hazard_avs, color=colors[i], marker=markers[i], linestyle=line_styles[i], label=hazard)
        i += 1
    if legend: 
        plt.legend(bbox_to_anchor=(1, 1.1), loc='upper left', fontsize=fontsize-2)
    plt.xticks(np.arange(0, int(len(time_vals))+1, xtick_freq),rotation=45)
    plt.margins(x=0.05)
    plt.tick_params(labelsize=fontsize)
    if save: 
        plt.savefig(dataset_name+'_hazard_'+metric_name+'.pdf', bbox_inches="tight") 
    plt.show()
    
def plot_metric_averages(metric_data, metric_name, show_std=True, title="", save=False, legend=True, dataset_name="",
                         figsize=(6,4), fontsize=16):
    """
    plots metric averages as a barchart

    Parameters
    ----------
    metric_data : dict
        nested dict where keys are hazards. inner dict has time value (usually years) as keys
        and list of metrics for values.
    metric_name : string
        name of metric, e.g., severity, used for axis and saving the figure
    show_std : boolean, optional
        true to show std deviation on time series as error bars. The default is True.
    title : string, optional
        title to add to plot. The default is "".
    save : boolean, optional
        true to save the plot as a pdf. The default is False.
    legend : boolean, optional
        true to show legend, false to hide legend. The default is True.
    dataset_name : string, optional
        name of the dataset, used for saving the plot. The default is "".
    figsize : tuple, optional
        size of the plot in inches. The default is (6,4).
    fontsize : int, optional
        fontsize for the plot. The default is 16.

    Returns
    -------
    None.

    """
    import textwrap
    avg = {hazard: np.average([m for year in metric_data[hazard] for m in metric_data[hazard][year]]) for hazard in metric_data}
    stddev = {hazard: np.std([m for year in metric_data[hazard] for m in metric_data[hazard][year]]) for hazard in metric_data}
    x_pos = np.arange(len(metric_data))
    fig, ax = plt.subplots()
    colors = cm.tab20(np.linspace(0, 1, len(metric_data)))
    labels = [key for key in metric_data.keys()]
    ax.bar(x_pos, avg.values(), yerr=stddev.values(), align='center', ecolor='black', capsize=10, color=colors)
    plt.xlabel("Hazard", fontsize=fontsize)
    plt.ylabel(metric_name, fontsize=fontsize)
    plt.title(title, fontsize=fontsize)
    ax.yaxis.grid(True)
    plt.tick_params(labelsize=fontsize)
    if legend == True:
        ax.set_xticklabels([])
        handles = [plt.Rectangle((0,0),1,1, color=color) for color in colors]
        plt.legend(handles, labels, bbox_to_anchor=(1, 1.1), loc='upper left', fontsize=fontsize)
    elif legend == False:
        labels = list(metric_data.keys())
        mean_length = np.mean([len(i) for i in labels])
        labels = ["\n".join(textwrap.wrap(i,mean_length)) for i in labels]
        ax.set_xticks(np.asarray([i for i in range(len(metric_data))]))
        ax.set_xticklabels(labels,rotation=45,ha="right",rotation_mode='anchor')
    if save: plt.savefig(dataset_name+'_hazard_bar_'+metric_name+'.pdf', bbox_inches="tight") 
    plt.show()
    
def plot_frequency_time_series(metric_data, metric_name='Frequency', line_styles=[], 
                               markers=[], title="", time_name="Year", xtick_freq=5, 
                               scale=True, save=False, dataset_name="", legend=True,
                               figsize=(6,4), fontsize=16):
    """
    plots hazard frequency over time. 
    different from plot metric time series because of input data.

    Parameters
    ----------
    metric_data : dict
        nested dict where keys are hazards. inner dict has time value (usually years) as keys
        and frequency count as an integer for values.
    metric_name : string, optional
        name of metric. The default is 'Frequency'.
    line_styles : list, optional
        list of line styles to use. should have one value for each hazard. The default is [].
    markers : list, optional
        list of line markers to use. should have one value for each hazard. The default is [].
    title : string, optional
        title to add to plot. The default is "".
    time_name : string, optional
        name of the time interval used to label the x axis. The default is "Year".
    xtick_freq : int, optional
        the number of values per x tick, e.g., the default would go 2015, 2020, 2025. The default is 5.
    scale : boolean, optional
        true to minmax scale data, false to use raw data. The default is False.
    save : boolean, optional
        true to save the plot as a pdf. The default is False.
    dataset_name : string, optional
        name of the dataset, used for saving the plot. The default is "".
    legend : boolean, optional
        true to show legend, false to hide legend. The default is True.
    figsize : tuple, optional
        size of the plot in inches. The default is (6,4).
    fontsize : int, optional
        fontsize for the plot. The default is 16.

    Returns
    -------
    None.

    """
    time_vals = list(set([year for hazard in metric_data for year in metric_data[hazard]]))
    time_vals.sort()
    frequencies = {hazard: [metric_data[hazard][year] for year in time_vals] for hazard in metric_data}
    if scale==True:
        hazard_freqs_scaled = {hazard: minmax_scale(frequencies[hazard]) for hazard in frequencies}
        y_label = "Total Scaled "+metric_name
    else:
        hazard_freqs_scaled = frequencies
        y_label = metric_name
    colors = cm.tab20(np.linspace(0, 1, len(frequencies)))
    plt.figure(figsize=figsize)
    plt.ylabel(y_label, fontsize=fontsize)
    plt.xlabel(time_name, fontsize=fontsize)
    plt.title(title, fontsize=fontsize)
    i = 0
    for hazard in hazard_freqs_scaled:
        plt.plot(time_vals, hazard_freqs_scaled[hazard], color=colors[i], label=hazard, marker=markers[i], linestyle=line_styles[i])
        i += 1
    if legend: 
        plt.legend(bbox_to_anchor=(1, 1.1), loc='upper left', fontsize=fontsize-2)
    plt.xticks(np.arange(0, int(len(time_vals))+1, xtick_freq),rotation=45)
    plt.margins(x=0.05)
    plt.tick_params(labelsize=fontsize)
    if save: plt.savefig(dataset_name+'_hazard_'+metric_name+'.pdf', bbox_inches="tight") 
    plt.show()

def make_pie_chart(docs, data, predictor, hazards, id_field, predictor_label=None, save=True):
    """makes a set of pie charts, with one pie chart per hazard showing the distribution of the categorical predictor variable specified.

    Parameters
    ----------
    docs : Dict
        nested dictionary used to store documents per hazard. Keys are hazards 
        and value is an inner dict. Inner dict has keys as time variables (e.g., years) and 
        values are lists.
    data : pandas DataFrame
        pandas datframe containing documents.
    predictor : string
        column in the data that has the categorical predictor of interest
    hazards : list
        list of hazards
    id_field : string
        the column in preprocessed df that contains document ids
    predictor_label : string, optional
        predictor label to be shown in the figure title, by default None
    save : bool, optional
        True to save the figure, by default True
    """
    if not predictor_label: predictor_label=predictor
    num_rows = int(np.ceil(len(hazards)/3))
    extra_axes = len(hazards)%3
    fig, axes = plt.subplots(num_rows, 3, figsize=(17,9))
    if extra_axes>0:
        for x in range(1,extra_axes):
            fig.delaxes(axes[num_rows-1][3-x])
    #set up lables, colors dict
    total_docs_with_hazards = [doc for hazard in hazards for year in docs[hazard] for doc in docs[hazard][year] ]
    labels = data.loc[data[id_field].isin(total_docs_with_hazards)][predictor].value_counts().index.sort_values()
    colors = cm.coolwarm(np.linspace(0, 1, len(labels)))
    for ax, hazard in zip(axes.flatten(), hazards):
        total_docs = [doc for year in docs[hazard] for doc in docs[hazard][year]]
        hazard_data = data.loc[data[id_field].isin(total_docs)].reset_index(drop=True)
        val_counts = hazard_data[predictor].value_counts()
        values = [val_counts[val] if val in val_counts else 0 for val in labels]
        _, _, autopct = ax.pie(values, labels=labels, colors=colors, autopct='%1.1f%%', textprops={'fontsize': 12},labeldistance=None, pctdistance=1.2)
        for txt in autopct:
            if float(txt.get_text().strip("%"))<3.0:
                txt.set_visible(False)
                
        ax.set_title(hazard+" per "+predictor_label, fontdict={'fontsize': 14})
    axes[0,0].legend(bbox_to_anchor=(-0.2, 1),fontsize=14)
    plt.savefig('hazard_'+predictor+'.pdf', bbox_inches="tight") 
    plt.show()

def chi_squared_tests(preprocessed_df, hazards, predictors, pred_dict={}): 
    """ Performs chi-squared test for each predictor to determine if there is a statistically
    significant difference in the counts of the predictor between reports with  and without each hazard

    Parameters
    ----------
    preprocessed_df : pandas DataFrame
        pandas datframe containing documents.
    hazards : pandas DataFrame
        pandas datframe containing documents.
    predictors : list
        list of columns in the dataframe with categorical predictors of interest.
    pred_dict : dict, optional
        dictionary with predictors from predicotr list as keys and names to disply in the table as values, default is {}.

    Returns
    -------
    stats_df : pandas DataFrame
        pandas dataframe containing the chi-squared statistic and p-val for each hazard-predictor pair
    """
    count_dfs = {}
    pred_dict = {'region_corrected':'Region', 'Type': "Aircraft Type", 'Agency':'Agency'}
    if pred_dict == None:
        pred_dict = {predictor: predictor for predictor in predictors}
    stat_vals = {(pred_dict[pred],val): [] for pred in predictors for val in ["chi-squared", "p-val"]}
    for predictor in predictors:
        pred_vals = [val for val in preprocessed_df[predictor].value_counts().index]
        diff_observed_expected = {pred_val:[] for pred_val in pred_vals}
        for hazard in hazards:
            expected, observed, stats = pg.chi2_independence(preprocessed_df, x=predictor,y=hazard)
            stat_vals[(pred_dict[predictor], "p-val")].append((stats.iloc[0]['pval'].round(3)))
            stat_vals[(pred_dict[predictor], "chi-squared")].append((stats.iloc[0]['chi2'].round(3)))
            for i in range(len(expected)):
                pred_val = expected.index[i]
                diff_observed_expected[pred_val].append(observed.iloc[i][0] - expected.iloc[i][0])
                diff_observed_expected[pred_val].append(observed.iloc[i][1] - expected.iloc[i][1])
        iterables = [hazards, [0,1]]
        index = pd.MultiIndex.from_product(iterables, names=["Hazard", "Present"])
        pred_df = pd.DataFrame(diff_observed_expected, index=index)
        count_dfs[predictor] = pred_df
    iterables = [[pred_dict[pred] for pred in predictors], ["p-val", "chi-squared"]]
    index = pd.MultiIndex.from_product(iterables, names=["Predictor", "Measure"])
    stats_df = pd.DataFrame(stat_vals, index=hazards, columns=index)
    return stats_df

def create_correlation_matrix(predictors_scaled, frequencies_scaled, graph=True, mask_vals=False, figsize=(6,4), fontsize=12, save=False, results_path="", title=False):
    """
    creates the correlation matrix between all predictors and all hazard frequencies
    all arguments are outputs from create_metrics_time_series

    Parameters
    ----------
    predictors_scaled : dict
        dictionary with keys as predictor names, values as a time series list of values scaled using minmax
    frequencies_scaled : dict
        dictionary with keys as hazard names, values as times series list of frequencies scaled using minmax
    graph : boolean, optional
        True to graph the data, false to not graph. The default is True.
    mask_vals : boolen, optional
        True to mask values that are not significant. The default is False.
    figsize : tuple, optional
        size of the plot in inches. The default is (6,4).
    fontsize : int, optional
        fontsize for the plot. The default is 16.
    save : boolean, optional
        True to save the graph. The default is False.
    results_path : string, optional
        path to save the plot to. The default is "".
    title : Boolean, optional
        True to show a title on graph. The default is False.

    Returns
    -------
    corrMatrix : pandas DataFrame
        correlation matrix.
    correlation_mat_total : pandas DataFrame
        stores the hazard and predictor values used for the correlation matrix
    p_values : pandas DataFrame
        p-vals for each correlation.

    """
    
    correlation_mat_data = predictors_scaled.copy()
    for hazard in frequencies_scaled:
        correlation_mat_data[hazard] = frequencies_scaled[hazard]
    correlation_mat_total = pd.DataFrame(correlation_mat_data)
    corrMatrix =correlation_mat_total.corr()
    p_values = corr_sig(correlation_mat_total)                     # get p-Value
    mask = np.invert(np.tril(p_values<0.05)) 
    if graph == True:
        if mask_vals:
            fig, (cax, ax) = plt.subplots(nrows=2, figsize=figsize,  gridspec_kw={"height_ratios":[0.025, 1]})
            # Draw heatmap
            sn.heatmap(corrMatrix, annot=True, mask=mask, annot_kws={'fontsize':fontsize},# cbar_kws={"orientation": "horizontal"}, 
                       vmin=-1, vmax=1, center= 0, cbar=False)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")
            plt.tick_params(labelsize=fontsize)
            # colorbar
            cbar = fig.colorbar(ax.get_children()[0], cax=cax, orientation="horizontal")
            cbar.ax.tick_params(labelsize=fontsize) 
            if title: plt.title("Correlational Matrix for Trends in \n Fires, Operations, Intensity, and Hazard Frequency per year")
            if save: plt.savefig(results_path+'.pdf', bbox_inches="tight")
            plt.show()
        else:
            fig, (cax, ax) = plt.subplots(nrows=2, figsize=figsize,  gridspec_kw={"height_ratios":[0.025, 1]})
            # Draw heatmap
            sn.heatmap(corrMatrix, annot=True, annot_kws={'fontsize':fontsize},# cbar_kws={"orientation": "horizontal"}, 
                       vmin=-1, vmax=1, center= 0, cbar=False)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")
            plt.tick_params(labelsize=fontsize)
            # colorbar
            cbar = fig.colorbar(ax.get_children()[0], cax=cax, orientation="horizontal")
            cbar.ax.tick_params(labelsize=fontsize) 
            if title: plt.title("Correlational Matrix for Trends in \n Fires, Operations, Intensity, and Hazard Frequency per year")
            if save: plt.savefig(results_path+'.pdf', bbox_inches="tight")
            plt.show()


    return corrMatrix, correlation_mat_total, p_values

def reshape_correlation_matrix(corrMatrix, p_values, predictors, hazards, figsize=(8,8.025), fontsize=16):
    """
    reshapes the correlation matrix between all predictors and all hazard frequencies
    columns are predictors and rows are hazards
    arguments are outputs from create_correlation_matrix

    Parameters
    ----------
    corrMatrix : pandas DataFrame
        correlation matrix.
    p_values : pandas DataFrame
        p-vals for each correlation.
    predictors : list
        list of predictors
    hazards : list
        list of hazards
    figsize : tuple, optional
        size of the plot in inches. The default is =(8,8.025).
    fontsize : int, optional
        fontsize for the plot. The default is 16.

    Returns
    -------
    None.

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
   
    fig, (cax, ax) = plt.subplots(nrows=2, figsize=figsize,  gridspec_kw={"height_ratios":[0.025, 1]})

    # Draw heatmap
    sn.heatmap(new_corr_df, annot=annotation_df,  fmt="s", annot_kws={'fontsize':fontsize},# cbar_kws={"orientation": "horizontal"}, 
               vmin=-1, vmax=1, center= 0, cbar=False)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
    plt.tick_params(labelsize=fontsize)
    plt.xlabel("Predictor", fontsize=fontsize)
    plt.ylabel("Hazard", fontsize=fontsize)

    # colorbar
    cbar = fig.colorbar(ax.get_children()[0], cax=cax, orientation="horizontal")
    cbar.ax.tick_params(labelsize=fontsize) 
    plt.show()

def hazard_accuracy(docs_per_hazard, num, results_path, hazard_words_per_doc, preprocessed_df, text_col, id_col, seed=0):
    """
    creates a data sheet to calculate hazard extraction acccuracy by randomly sampling documents for each hazard.
    this method actually is calculating precision at k, k=num.
    note that this method is not the prefered method to evaluate hazard accuracy,
    instead a user should use the classification metrics and label a validation set.

    Parameters
    ----------
    docs_per_hazard : Dict
        nested dictionary used to store documents per hazard. Keys are hazards 
        and value is an inner dict. Inner dict has keys as time variables (e.g., years) and 
        values are lists.
    num : int
        number of documents to sample for each hazard.
    results_path : string
        filepath to topic modelresults spreadsheet
    hazard_words_per_doc : Dict
        used to store the hazard words per document. keys are hazards and values are lists
        with an element for each document.
    preprocessed_df : pandas DataFrame
        pandas datframe containing documents
    text_col : string
        column in preprocessed_df containing text
    id_col : string
        column in preprocessed_df containing document ids
    seed : int, optional
        seed for random sampling. The default is 0.

    Returns
    -------
    sampled_hazard_ids : dict
        dictionary with hazards as keys and a list of document ids for values
    total_ids : list
        list of all the ids of documents belonging to any hazard

    """
    hazards = [hazard for hazard in docs_per_hazard]
    sampled_hazard_ids = {hazard:[] for hazard in docs_per_hazard}
    num_total_ids = {hazard:0 for hazard in docs_per_hazard}
    docs = preprocessed_df[id_col].tolist()
    document_text = preprocessed_df[text_col].tolist()
    data = {}
    sentences = []
    for hazard in docs_per_hazard:
        total_ids = [id_ for year in docs_per_hazard[hazard] for id_ in docs_per_hazard[hazard][year]]
        sampled_ids = random.sample(total_ids, min(num, len(total_ids)))
        sampled_hazard_ids[hazard] = sampled_ids
        num_total_ids[hazard] = len(total_ids)
        
        hazard_words = [hazard_words_per_doc[hazard][docs.index(id_)] for id_ in sampled_ids]
        doc_text = [document_text[docs.index(id_)] for id_ in sampled_ids]
        sentences = [". ".join([sent for sent in ". ".join(doc_text[i]).split(".") if hazard_words[i] in sent]) for i in range(len(doc_text))]
        data[hazard] = pd.DataFrame({"ID": sampled_ids,
                                     "Contains Hazard": [0 for id_ in sampled_ids],
                                     "Hazard Word": hazard_words,
                                     "Document Text": doc_text,
                                     "Sentence": sentences})
    data["Summary"] = pd.DataFrame({"Hazards": [hazard for hazard in docs_per_hazard],
                       "# Total IDs": [num_total_ids[hazard] for hazard in num_total_ids],
                       "# Sampled IDs": [len(sampled_hazard_ids[hazard]) for hazard in sampled_hazard_ids],
                       "% Sampled": [len(sampled_hazard_ids[hazard])/num_total_ids[hazard] for hazard in num_total_ids],
                       "# Correct Sampled IDs": [0 for hazard in docs_per_hazard],
                       "Accuracy": [0 for hazard in docs_per_hazard]
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
                        worksheet.write_formula('F'+str(i+2),'{=E'+str(i+2)+'/C'+str(i+2)+'}')
                else:
                    worksheet = writer.sheets[sheet_name]
                    worksheet.write('B'+str(num+2),'{=SUM(B2:B'+str(num+1)+')}')
    return sampled_hazard_ids, total_ids

def get_likelihood_FAA(rates):
    """
    converts hazard rate of occurrence to an FAA likelihood category

    Parameters
    ----------
    rates : dict
        dictionary with hazard name as keys and a rate as a value

    Returns
    -------
    curr_likelihoods : dict
        dictionary with hazard name as keys and a likelihood category as a value

    """
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

def get_likelihood_USFS(rates):
    """
    converts hazard rate of occurrence to an USFS likelihood category

    Parameters
    ----------
    rates : dict
        dictionary with hazard name as keys and a rate as a value

    Returns
    -------
    curr_likelihoods : dict
        dictionary with hazard name as keys and a likelihood category as a value

    """
    curr_likelihoods = {hazard:0 for hazard in rates}
    for hazard in rates:
        r = rates[hazard]
        if r>=100:
            likelihood = 'Frequent'
        elif r>=10 and r<100:
            likelihood = 'Probable'
        elif r>=1 and r<10:
            likelihood = 'Occasional'
        elif r>=0.1 and r<1:
            likelihood = 'Remote'
        elif r<0.1:
            likelihood = 'Improbable'
        curr_likelihoods[hazard] = likelihood
    return curr_likelihoods

def plot_USFS_risk_matrix(likelihoods, severities, figsize=(9,5), save=False, results_path="", fontsize=12, max_chars=20, title=False):
    """
    plots a USFS risk matrix from likelihood and severity categories

    Parameters
    ----------
    likelihoods : dict
        dictionary with hazard name as keys and a likelihood category as a value
    severities : dict
        dictionary with hazard name as keys and a severity category as a value
    figsize : tuple, optional
        figure size in inches. The default is (9,5).
    save : boolean, optional
        true to save the figure. The default is False.
    results_path : string, optional
        path to save figure to. The default is "".
    fontsize : int, optional
        figure fontsize. The default is 12.
    max_chars : int, optional
        maximum characters per line in a cell of the risk matrix.
        used to improve readability and ensure hazard names are contained in a cell.
        The default is 20.
    title : boolean, optional
        Dtrue to show title. The default is False.

    Returns
    -------
    None.

    """
    hazards = [h for h in likelihoods]
    curr_likelihoods = likelihoods
    curr_severities = severities
    annotation_df = pd.DataFrame([["" for i in range(4)] for j in range(5)],
                         columns=['Negligible', 'Marginal', 'Critical', 'Catastrophic'],
                          index=['Frequent', 'Probable', 'Occasional', 'Remote','Improbable'])
    annot_font = fontsize
    hazard_likelihoods = {hazard:"" for hazard in hazards}; hazard_severities={hazard:"" for hazard in hazards}
    for hazard in hazards:
        hazard_likelihoods[hazard] = curr_likelihoods[hazard]
        hazard_severities[hazard] = curr_severities[hazard]
        new_annot = annotation_df.at[curr_likelihoods[hazard], curr_severities[hazard]]
        if new_annot != "": new_annot += ", "
        hazard_annot = hazard.split(" ")
        #if line>20 then new line
        if len(hazard_annot)>1 and len(new_annot.split("\n")[-1]) + len(hazard_annot[0]) +len(hazard_annot[1]) < max_chars:
            new_annot += hazard_annot[0] + " "+ hazard_annot[1] 
            annot_ind = 2
        elif len(new_annot.split("\n")[-1]) + len(hazard_annot[0]) < max_chars:
            new_annot += hazard_annot[0]
            annot_ind = 1
        elif len(hazard_annot)>1 and len(hazard_annot[1]) + len(hazard_annot[0]) < max_chars:
            new_annot += "\n" + hazard_annot[0] + " " + hazard_annot[1]
            annot_ind = 2
        else:
            new_annot += "\n" + hazard_annot[0]
            annot_ind = 1
        if len(hazard_annot)>1 and annot_ind<len(hazard_annot):
            new_annot += "\n"+" ".join(hazard_annot[annot_ind:])
        annotation_df.at[curr_likelihoods[hazard], curr_severities[hazard]] = new_annot #+= (str(hazard_annot))
    
    df = pd.DataFrame([[2, 3, 4, 4], [2, 3, 4, 4], [1, 2, 3, 4], [1, 2, 2, 3], [1, 2, 2, 2]],
                      columns=['Negligible', 'Marginal', 'Critical', 'Catastrophic'],
                      index=['Frequent', 'Probable', 'Occasional', 'Remote','Improbable'])
    fig,ax = plt.subplots(figsize=figsize)
    myColors = (mcolors.to_rgb(mcolors.cnames['green']),
                mcolors.to_rgb(mcolors.cnames['dodgerblue']),
                mcolors.to_rgb(mcolors.cnames['yellow']),
                mcolors.to_rgb(mcolors.TABLEAU_COLORS['tab:red']))
    cmap = LinearSegmentedColormap.from_list('Custom', myColors, len(myColors))
    #annot df has hazards in the cell they belong to #annot=annotation_df
    sn.heatmap(df, annot=annotation_df, fmt='s',annot_kws={'fontsize':annot_font},cbar=False,cmap=cmap)
    if title: plt.title("Risk Matrix", fontsize=fontsize)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)#, ha="right")
    plt.tick_params(labelsize=fontsize)
    plt.ylabel("Likelihood", fontsize=fontsize)
    plt.xlabel("Severity", fontsize=fontsize)
    minor_ticks = np.arange(1, 5, 1)
    ax.set_xticks(minor_ticks, minor=True)
    ax.set_yticks(minor_ticks, minor=True)
    ax.tick_params(which='minor',length=0, grid_color='black', grid_alpha=1)
    ax.grid(which='minor', alpha=1)
    if save: 
        plt.savefig(results_path+".pdf", bbox_inches="tight")
    plt.show()

def plot_risk_matrix(likelihoods, severities, figsize=(9,5), save=False, results_path="", fontsize=12, max_chars=20):
    """
    plots a FAA risk matrix from likelihood and severity categories

    Parameters
    ----------
    likelihoods : dict
        dictionary with hazard name as keys and a likelihood category as a value
    severities : dict
        dictionary with hazard name as keys and a severity category as a value
    figsize : tuple, optional
        figure size in inches. The default is (9,5).
    save : boolean, optional
        true to save the figure. The default is False.
    results_path : string, optional
        path to save figure to. The default is "".
    fontsize : int, optional
        figure fontsize. The default is 12.
    max_chars : int, optional
        maximum characters per line in a cell of the risk matrix.
        used to improve readability and ensure hazard names are contained in a cell.
        The default is 20.
    title : boolean, optional
        Dtrue to show title. The default is False.

    Returns
    -------
    None.

    """
    hazards = [h for h in likelihoods]
    curr_likelihoods = likelihoods
    curr_severities = severities
    annotation_df = pd.DataFrame([["" for i in range(5)] for j in range(5)],
                         columns=['Minimal Impact', 'Minor Impact', 'Major Impact', 'Hazardous Impact', 'Catastrophic Impact'],
                          index=['Frequent', 'Probable', 'Remote','Extremely Remote', 'Extremely Improbable'])
    annot_font = fontsize
    hazard_likelihoods = {hazard:"" for hazard in hazards}; hazard_severities={hazard:"" for hazard in hazards}
    for hazard in hazards:
        hazard_likelihoods[hazard] = curr_likelihoods[hazard]
        hazard_severities[hazard] = curr_severities[hazard]
        new_annot = annotation_df.at[curr_likelihoods[hazard], curr_severities[hazard]]
        if new_annot != "": new_annot += ", "
        hazard_annot = hazard.split(" ")
        #if line>20 then new line
        if len(hazard_annot)>1 and len(new_annot.split("\n")[-1]) + len(hazard_annot[0]) +len(hazard_annot[1]) < max_chars:
            new_annot += hazard_annot[0] + " "+ hazard_annot[1] 
            annot_ind = 2
        elif len(new_annot.split("\n")[-1]) + len(hazard_annot[0]) < max_chars:
            new_annot += hazard_annot[0]
            annot_ind = 1
        elif len(hazard_annot)>1 and len(hazard_annot[1]) + len(hazard_annot[0]) < max_chars:
            new_annot += "\n" + hazard_annot[0] + " " + hazard_annot[1]
            annot_ind = 2
        else:
            new_annot += "\n" + hazard_annot[0]
            annot_ind = 1
        if len(hazard_annot)>1 and annot_ind<len(hazard_annot):
            new_annot += "\n"+" ".join(hazard_annot[annot_ind:])
        annotation_df.at[curr_likelihoods[hazard], curr_severities[hazard]] = new_annot #+= (str(hazard_annot))
    
    df = pd.DataFrame([[0, 5, 10, 10, 10], [0, 5, 5, 10, 10], [0, 0, 5, 5, 10],
            [0, 0, 0, 5, 5], [0, 0, 0, 0, 5]],
          columns=['Minimal Impact', 'Minor Impact', 'Major Impact', 'Hazardous Impact', 'Catastrophic Impact'],
           index=['Frequent', 'Probable', 'Remote','Extremely Remote', 'Extremely Improbable'])
    fig,ax = plt.subplots(figsize=figsize)
    #annot df has hazards in the cell they belong to #annot=annotation_df
    sn.heatmap(df, annot=annotation_df, fmt='s',annot_kws={'fontsize':annot_font},cbar=False,cmap='RdYlGn_r')
    plt.title("Risk Matrix", fontsize=fontsize)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)#, ha="right")
    plt.tick_params(labelsize=fontsize)
    plt.ylabel("Likelihood", fontsize=fontsize)
    plt.xlabel("Severity", fontsize=fontsize)
    minor_ticks = np.arange(1, 6, 1)
    ax.set_xticks(minor_ticks, minor=True)
    ax.set_yticks(minor_ticks, minor=True)
    ax.tick_params(which='minor',length=0, grid_color='black', grid_alpha=1)
    ax.grid(which='minor', alpha=1)
    if save: 
        plt.savefig(results_path+".pdf", bbox_inches="tight")
    plt.show()

def sample_for_accuracy(preprocessed_df, id_col, text_col, hazards, save_path, num_sample=100): ##TODO: rename
    """
    generates a spreadsheet of randomly sampled documents to analyze the quality of hazard extraction

    Parameters
    ----------
    preprocessed_df : pandas DataFrame
        pandas datframe containing documents
    id_col : string
        column in preprocessed_df containing document ids
    text_col : string
        column in preprocessed_df containing text
    hazards : list
        list of hazards
    save_path : string
        location to save the file to
    num_sample : int, optional
        number of documents to sample. The default is 100.

    Returns
    -------
    sampled_df : pandas DataFrame
        dataframe of sampled documents

    """
    text = []
    hazards_dict = {hazard:[0 for t in range(num_sample)] for hazard in hazards}
    total_ids = preprocessed_df[id_col].to_list()
    sampled_ids = random.sample(total_ids, min(num_sample, len(total_ids)))
    for id_ in sampled_ids:
        ind = total_ids.index(id_)
        text.append(preprocessed_df.at[ind, text_col])
    data_dict = {id_col: sampled_ids,
                 text_col: text}
    data_dict.update(hazards_dict)
    sampled_df = pd.DataFrame(data_dict)
    sampled_df.to_csv(save_path)
    return sampled_df

def calc_classification_metrics(labeled_file, docs_per_hazard, id_col):
    """
    calculates classification metrics where the true lables come from a file and
    the predicted labels are from the hazard extraction process

    Parameters
    ----------
    labeled_file : string
        location the labeled file is saved at
    docs_per_hazard : Dict
        nested dictionary used to store documents per hazard. Keys are hazards 
        and value is an inner dict. Inner dict has keys as time variables (e.g., years) and 
        values are lists.
    id_col : string
        column in preprocessed_df containing document ids

    Returns
    -------
    metrics_df : pandas DataFrame
        dataframe with recall, precision, f1, accuracy, and support for each hazard
    labeled_docs : pandas DataFrame
        dataframe of the manually labled docs
    HEAT_labeled_docs : pandas DataFrame
        dataframe of the HEAT labled docs

    """
    labeled_docs = pd.read_csv(labeled_file)
    labeled_doc_ids = labeled_docs[id_col].tolist()
    hazards = docs_per_hazard.keys()
    docs_per_hazard_flat = {hazard:[i for year in docs_per_hazard[hazard] for i in docs_per_hazard[hazard][year]] for hazard in hazards}
    HEAT_labeled_docs = pd.DataFrame({hazard:[0 for i in range(len(labeled_doc_ids))] for hazard in docs_per_hazard})
    HEAT_labeled_docs[id_col] = labeled_doc_ids
    for i in range(len(labeled_doc_ids)):
        id_ = labeled_doc_ids[i]
        hazards_per_doc = [hazard for hazard in docs_per_hazard_flat if id_ in docs_per_hazard_flat[hazard]]
        for hazard in hazards_per_doc:
            HEAT_labeled_docs.at[i, hazard] = 1
        
    recall = {hazard: recall_score(labeled_docs[hazard].tolist(), HEAT_labeled_docs[hazard].tolist()) for hazard in docs_per_hazard}
    precision = {hazard: precision_score(labeled_docs[hazard].tolist(), HEAT_labeled_docs[hazard].tolist()) for hazard in docs_per_hazard}
    f1 = {hazard: f1_score(labeled_docs[hazard].tolist(), HEAT_labeled_docs[hazard].tolist()) for hazard in docs_per_hazard}
    accuracy = {hazard: accuracy_score(labeled_docs[hazard].tolist(), HEAT_labeled_docs[hazard].tolist()) for hazard in docs_per_hazard}
    support = {hazard:sum(labeled_docs[hazard].tolist()) for hazard in docs_per_hazard}
    metrics_df = pd.DataFrame({"Recall": recall.values(),
                              "Precision": precision.values(),
                              "F1": f1.values(),
                              "Accuracy": accuracy.values(),
                              "Support": support.values()},
                             index=hazards)
    metrics_df = metrics_df.round(3)
    return metrics_df, labeled_docs, HEAT_labeled_docs

def examine_hazard_extraction_mismatches(preprocessed_df, true, pred, hazards, hazard_words_per_doc, topics_per_doc, hazard_topics_per_doc, id_col, text_col, results_path):
    """
    used to examine which documents are mislabled by HEAT. used iteratively to refine hazard extraction.

    Parameters
    ----------
    preprocessed_df : TYPE
        DESCRIPTION.
    true : pandas DataFrame
        dataframe of the manually labled docs
    pred : pandas DataFrame
        dataframe of the HEAT labled docs
    hazards : list
        list of hazards
    hazard_words_per_doc : Dict
        used to store the hazard words per document. keys are hazards and values are lists
        with an element for each document.
    topics_per_doc : dict
        dictionary with keys as document ids and values as a list of topic numbers
    hazard_topics_per_doc : dict
        nested dictionary with keys as document ids. inner dictionary has hazard names as keys
        and values as a list of topics
    id_col : string
        column in preprocessed_df containing document ids
    text_col : string
        column in preprocessed_df containing text
    results_path : string
        location to save the resulting datasheets to

    Returns
    -------
    dfs : dict
        dictionary with keys as hazards and values as pandas dataframes storing document mismatches

    """
    dfs = {}
    docs = preprocessed_df[id_col].tolist()
    for hazard in hazards:
        true_vals = true[hazard].tolist()
        pred_vals = pred[hazard].tolist()
        inds = [i for i in range(len(true_vals)) if true_vals[i]!=pred_vals[i]]
        ids = [true.at[i, id_col] for i in inds]
        true_vals_for_inds = [true_vals[i] for i in inds]
        pred_vals_for_inds = [pred_vals[i] for i in inds]
        mismatches = preprocessed_df.loc[preprocessed_df[id_col].isin(ids)].reset_index(drop=True)
        mismatches = mismatches.set_index(id_col)
        mismatches = mismatches.reindex(ids)
        #print(len(inds), len(true_vals_for_inds), len(pred_vals_for_inds), len(mismatches), len(ids))
        mismatches['True'] = true_vals_for_inds
        mismatches['Predictions'] = pred_vals_for_inds
        hazard_words = [hazard_words_per_doc[hazard][docs.index(id_)] for id_ in ids]
        mismatches['Hazard Words'] = hazard_words
        mismatches['Topics'] = [topics_per_doc[doc] for doc in ids]
        mismatches['Hazard Topics'] = [hazard_topics_per_doc[doc][hazard] for doc in ids]
        mismatches = mismatches[['True', 'Predictions', text_col, 'Hazard Words', 'Topics', 'Hazard Topics']]
        dfs[hazard] = mismatches
    with pd.ExcelWriter(results_path+"/hazard_extraction_mismatches.xlsx", engine='xlsxwriter') as writer:
        for results in dfs:
            if len(results)>31:
                sheet_name = results[:30]
            else: 
                sheet_name = results
            dfs[results].to_excel(writer, sheet_name = sheet_name, index = True)
    return dfs

def get_word_frequencies(hazard_words_per_doc, hazards_sorted=None):
    """
    

    Parameters
    ----------
    hazard_words_per_doc : Dict
        used to store the hazard words per document. keys are hazards and values are lists
        with an element for each document.
    hazards_sorted : list, optional
        ordered list of hazards for generating frequencies. The default is None.

    Returns
    -------
    word_frequencies : dictionary
        nested dictionary where keys are hazards. inner dictionary has words as keys and word frequencies as values.

    """
    hazard_words_per_doc_cleaned = {hazard: [w for w in hazard_words_per_doc[hazard] if w!='none'] for hazard in hazard_words_per_doc}
    word_frequencies = {hazard:{np.unique(hazard_words_per_doc_cleaned[hazard], return_counts=True)[0][i]:np.unique(hazard_words_per_doc_cleaned[hazard], return_counts=True)[1][i] for i in range(len(np.unique(hazard_words_per_doc_cleaned[hazard], return_counts=True)[0]))} for hazard in hazard_words_per_doc_cleaned}
    if hazards_sorted:
        word_frequencies = {hazard: word_frequencies[hazard] for hazard in hazards_sorted}
    return word_frequencies

def build_word_clouds(word_frequencies, nrows, ncols, figsize=(8, 4), cmap=None, save=False, save_path=None, fontsize=10, wordcloud_kwargs={}):
    """
    builds a word cloud for each hazard

    Parameters
    ----------
    word_frequencies : dictionary
        nested dictionary where keys are hazards. inner dictionary has words as keys and word frequencies as values.
    nrows : int
        number of rows in the grid of word clouds
    ncols : int
        number of columns in the grid of word clouds
    figsize : tuple, optional
        figure size in inches. The default is (8, 4).
    cmap : matplotlib colormap, optional
        colormap object used for coloring the word clouds. The default is None.
    save : boolean, optional
        true to save figure. The default is False.
    save_path : string, optional
        path to save figure to. The default is None.
    fontsize : int, optional
        fontsize for title and minimum fontsize in wordcloud. The default is 10.
    wordcloud_kwargs : dict, optional
        optional keyword args to pass into the wordcloud object. The default is {}.

    Returns
    -------
    None

    """
    fig, axs = plt.subplots(nrows = nrows,
                            ncols = ncols,
                            figsize = figsize)
    i=0;j=0; 
    c=0
    colors = cm.tab20(np.linspace(0, 1, len(word_frequencies)))
    for hazard in word_frequencies:
        ax = axs[i,j]
        color = colors[c]
        def color_func(word, font_size, position, orientation,random_state=None, **kwargs):
            r, g, b, _ = 255 * np.array(color)
            return "rgb({:.0f}, {:.0f}, {:.0f})".format(r, g, b)
        wordcloud = WordCloud(prefer_horizontal=1,background_color='white', color_func=color_func, min_font_size=fontsize, 
                              font_path="times", **wordcloud_kwargs)
        wordcloud.generate_from_frequencies(word_frequencies[hazard])
        ax.imshow(wordcloud)
        ax.axis('off')
        ax.set_title(hazard, fontfamily="Times New Roman", fontsize=fontsize)
        i+=1
        if i==nrows:
            i=0
            j+=1
        c+=1
    while j < ncols:
        ax = axs[i,j]
        ax.set_axis_off()
        j+=1
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()
    if save: plt.savefig(save_path+'word_clouds.pdf', bbox_inches="tight") 
    plt.show()

def plot_predictors(predictors, predictor_labels, time, time_label='Year', title="", totals=True, averages=True, scaled=True, figsize=(12, 5), axs=[], fig=None, show=False, legend=True):
    """
    function for plotting predictor timeseries

    Parameters
    ----------
    predictors : list
        list of predictors
    predictor_labels : list
        list of labels for the predictors which will be shown on the graph
    time : list
        list of time values tat define the axis/time series.
    time_label : string, optional
        label for the time values. The default is 'Year'.
    title : string, optional
        figure title. The default is "".
    totals : boolean, optional
        true to graph total or sum values. The default is True.
    averages : boolean, optional
        true to graph average values. The default is True.
    scaled : boolean, optional
        true to minmax scale the timeseries data. The default is True.
    figsize : tuple, optional
        figure size in inches. The default is (12, 5).
    axs : matplotlib axs object, optional
        used to plot multiple graphs on one figure. The default is [].
    fig :  matplotlib fig object, optional
        used to plot multiple graphs on one figure. The default is None.
    show : boolean, optional
        true to show figure, false to return graph objects. The default is False.
    legend : boolean, optional
        true to show legend. The default is True.

    Returns
    -------
    fig : matplotlib fig object
        the figure object
    axs : matplotlib axs object
        axs object

    """
    if axs == []:
        fig, axs = plt.subplots(1,1, sharex=True, sharey=True, figsize=figsize)
    if totals:
        ylabel = 'Total'
        if scaled: ylabel += " Scaled"
    if averages:
        ylabel = 'Average'
        if scaled: ylabel += " Scaled"
    axs.set_ylabel(ylabel)
    axs.set_xlabel(time_label)
    axs.set_title(title)
    for i in range(len(predictors)):
        axs.plot(time, predictors[i], label=predictor_labels[i])
    if legend: axs.legend()
    if show:
        plt.show()
    return fig, axs

def proposed_topics(lists=[]):
    """
    experimental function to identify topics that may be relevent to specified hazards
    based on manually labeled data

    Parameters
    ----------
    lists : list, optional
        list of lists. inner lists are topic numbers for each document manually labeled
        as associated with a hazard. The default is [].

    Returns
    -------
    proposed_topics : list
        list of proposed new topics.

    """
    total_nums = [l for li in lists for l in li]
    topics, counts = np.unique(total_nums, return_counts=True)
    proposed_topics = [topics[i] for i in range(len(counts)) if counts[i]>=(len(lists)/3)]
    return proposed_topics
