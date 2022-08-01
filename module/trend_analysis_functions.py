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
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score, recall_score, precision_score, f1_score
from wordcloud import WordCloud
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

def remove_outliers(data, threshold=1.5, rm_outliers=True):
    if data == [] or rm_outliers == False:
        return data
    Q1 = np.quantile(data,0.25)
    Q3 = np.quantile(data,0.75)
    IQR = Q3 - Q1
    new_data = [pt for pt in data if (pt>(Q1-1.5*IQR)) and (pt<(Q3+1.5*IQR))]
    return new_data

def check_for_hazard_words(h_word, text):
    hazard_found = False
    if h_word in text:
            hazard_found = True
    return hazard_found

def check_for_negation_words(negation_words, text, h_word):
    #negation_words = [word for neg_words in negation_words for word in neg_words.split(", ")]
    #for neg_words in negation_words: #this doesnt seem right
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

def get_topics_per_doc(docs, results, results_text_field, hazards): ##TODO: Speed up
    topics_per_doc = {doc:[] for doc in docs}
    for i in range(len(results[results_text_field])):
        row_docs = [doc for doc in docs if doc in results[results_text_field].at[i,'documents']]
        for doc in row_docs:
            topics_per_doc[doc].append(int(results[results_text_field].at[i, 'topic number']))
    hazard_topics_per_doc = {doc:{hazard:[] for hazard in hazards} for doc in topics_per_doc}
    return topics_per_doc, hazard_topics_per_doc

def get_hazard_info(hazard_file):
    hazard_info = pd.read_excel(hazard_file, sheet_name=['topic-focused'])
    hazards = hazard_info['topic-focused']['Hazard name'].tolist()#list(set(hazard_info['topic-focused']['Hazard name'].tolist()))
    hazards = [hazard for hazard in hazards if isinstance(hazard,str)]
    return hazard_info, hazards

def get_results_info(results_file, results_text_field, text_field, doc_topic_dist_field):
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
    docs = preprocessed_df[id_field].tolist()
    hazard_words_per_doc = {hazard:['none' for doc in docs] for hazard in hazards}
    time_period = preprocessed_df[time_field].unique()
    frequency = {name:{str(time_p):0 for time_p in time_period} for name in hazards}
    docs_per_hazard = {hazard:{str(time_p):[] for time_p in time_period} for hazard in hazards}
    return docs, frequency, docs_per_hazard, hazard_words_per_doc

def get_hazard_df(hazard_info, hazards, i):
    hazard_name = hazards[i]
    hazard_df = hazard_info['topic-focused'].loc[hazard_info['topic-focused']['Hazard name'] == hazard_name].reset_index(drop=True)
    return hazard_df, hazard_name

def get_hazard_topics(hazard_df, begin_nums):
    nums = [int(i)+begin_nums for nums in hazard_df['Topic Number'] for i in str(nums).split(", ")]#identifies all topics related to this hazard
    return nums

def get_hazard_doc_ids(nums, results, results_text_field, docs, doc_topic_distribution, text_field, topic_thresh, preprocessed_df, id_field):
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
    #get hazard_topics
    for doc in ids:
        hazard_topics_per_doc[doc][hazard_name] = [num-begin_nums for num in nums if num-begin_nums in topics_per_doc[doc]]
    return hazard_topics_per_doc

def get_hazard_words(hazard_df):
    hazard_words = hazard_df['Relevant hazard words'].to_list()
    hazard_words = [word for words in hazard_words for word in words.split(", ")]
    return hazard_words

def get_negation_words(hazard_df):
    negation_words = hazard_df['Negation words'].to_list()
    negation_words = [word for word in negation_words if isinstance(word, str)]
    negation_words = [word for neg_words in negation_words for word in neg_words.split(", ")]
    return negation_words

def record_hazard_doc_info(hazard_name, year, docs_per_hazard, id_, frequency, hazard_words_per_doc, docs, h_word):
    #year = temp_fire_df.iloc[j][time_field]
    docs_per_hazard[hazard_name][str(year)].append(id_)
    frequency[hazard_name][str(year)] += 1
    hazard_words_per_doc[hazard_name][docs.index(id_)] = h_word
    return docs_per_hazard, frequency, hazard_words_per_doc

def get_doc_time(id_, temp_df, id_field, time_field):
    year = temp_df.loc[temp_df[id_field]==id_][time_field].values[0]
    return year

def get_doc_text(id_, temp_df, id_field, text_field):
    text = temp_df.loc[temp_df[id_field]==id_][text_field].values[0]
    text = " ".join(text)
    return text 

def check_if_word_contained_in_other_word():
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
def old_identify_docs_per_hazard(hazard_file, preprocessed_df, results_file, text_field, time_field, id_field, results_text_field=None, doc_topic_dist_field=None, topic_thresh=0.0, ids_to_drop=[]):
    hazard_info = pd.read_excel(hazard_file, sheet_name=['topic-focused'])
    hazards = list(set(hazard_info['topic-focused']['Hazard name'].tolist()))
    hazards = [hazard for hazard in hazards if isinstance(hazard,str)]
    docs = preprocessed_df[id_field].tolist()
    hazard_words_per_doc = {hazard:['none' for doc in docs] for hazard in hazards}
    time_period = preprocessed_df[time_field].unique()
    #categories = hazard_info['topic-focused']['Hazard Category'].tolist()
    ##punctuation = ['.', ',', "'", '"', '?', '!']
    ##stemmer = PorterStemmer()
    ##english_words = [w.lower() for w in words.words()]
    print("read hazard file")
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
    print("read results file")
    frequency = {name:{str(time_p):0 for time_p in time_period} for name in hazards}
    docs_per_hazard = {hazard:{str(time_p):[] for time_p in time_period} for hazard in hazards}
    if results[results_text_field].at[0,'topic number'] == -1:
        begin_nums = 1
    else:
        begin_nums = 0
    
    topics_per_doc, hazard_topics_per_doc = get_topics_per_doc(docs, results, results_text_field, hazards)
    #print("initiated topics per document")
    for i in tqdm(range(len(hazards))):
        hazard_name = hazards[i]
        hazard_df = hazard_info['topic-focused'].loc[hazard_info['topic-focused']['Hazard name'] == hazards[i]].reset_index(drop=True)

        nums = [int(i)+begin_nums for nums in hazard_df['Topic Number'] for i in str(nums).split(", ")]#identifies all topics related to this hazard
        ids_df = results[results_text_field].loc[nums]
        ids_ = "; ".join(ids_df['documents'].to_list())
        ids_ = [id_ for id_ in docs if id_ in ids_]   
        #get hazard_topics
        for doc in ids_:
            hazard_topics_per_doc[doc][hazards[i]] = [num-begin_nums for num in nums if num-begin_nums in topics_per_doc[doc]]
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
        ids = temp_df[id_field].unique()
        #check for hazard -- looks at the hazard relevant words from the topic
        #hazard_name = hazards[i]
        hazard_words = hazard_df['Relevant hazard words'].to_list()
        hazard_words = list(set([word for words in hazard_words for word in words.split(", ")]))
        negation_words = hazard_df['Negation words'].to_list()
        negation_words = [word for word in negation_words if isinstance(word, str)]
        for id_ in ids:
            temp_fire_df = temp_df.loc[temp_df[id_field]==id_].reset_index(drop=True)#just need text?
            for j in range(len(temp_fire_df)):
                text = temp_fire_df.iloc[j][text_field]
                text = " ".join(text)
                text = text.replace(".", " ")#???
                
                #need to check if a word in text is in hazard words
                hazard_found = False
                for h_word in hazard_words:
                    #end_ind = 0
                    if h_word in text:
                        hazard_found = True
                        #break
                    
                    if negation_words!=[] and hazard_found == True:
                        #negation_words = [word for neg_words in negation_words for word in neg_words.split(", ")]
                        for neg_words in negation_words: #this doesnt seem right
                            for word in neg_words.split(", "):#removes texts that have negation words
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
                    if hazard_found == True:
                        break

                if hazard_found == True:
                    year = temp_fire_df.iloc[j][time_field]
                    docs_per_hazard[hazard_name][str(year)].append(id_)
                    frequency[hazard_name][str(year)] += 1
                    hazard_words_per_doc[hazard_name][docs.index(id_)] = h_word

    return frequency, docs_per_hazard, hazard_words_per_doc, topics_per_doc, hazard_topics_per_doc

def identify_docs_per_hazard(hazard_file, preprocessed_df, results_file, text_field, time_field, id_field, results_text_field=None, doc_topic_dist_field=None, topic_thresh=0.0): #NOTE: removed ids_to_drop
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

def plot_metric_time_series(metric_data, metric_name, line_styles=[], markers=[], title="", time_name="Year", scaled=False, xtick_freq=5, show_std=True, save=False, dataset_name="", yscale=None, legend=True):
    time_vals = list(set([year for hazard in metric_data for year in metric_data[hazard]]))
    time_vals.sort()
    #scaled -> scaled the averages, how to scale stddev?
    if scaled: metric_data = {hazard: minmax_scale(metric_data[hazard]) for hazard in metric_data}
    averages = {hazard: [np.average(metric_data[hazard][year]) for year in time_vals] for hazard in metric_data}
    stddevs = {hazard: [np.std(metric_data[hazard][year]) for year in time_vals] for hazard in metric_data}
    colors = cm.tab20(np.linspace(0, 1, len(averages)))
    plt.figure()
    plt.title(title, fontsize=16)
    plt.xlabel(time_name, fontsize=16)
    plt.ylabel(metric_name, fontsize=16)
    if yscale == 'log':
        plt.yscale('symlog')
    i=0
    for hazard in averages:
        temp_time_vals = time_vals.copy()
        nans = np.where(np.isnan(averages[hazard]))[0]
        hazard_avs = averages[hazard]
        hazard_stddev = stddevs[hazard]
        for ind in nans:
            temp_time_vals.pop(ind)
            hazard_avs.pop(ind)
            hazard_stddev.pop(ind)
        if show_std == True:
            plt.errorbar(temp_time_vals, hazard_avs, yerr=hazard_stddev, color=colors[i], marker=markers[i], linestyle=line_styles[i], label=hazard, capsize=5, markeredgewidth=1)
        else:
            plt.plot(temp_time_vals, hazard_avs, color=colors[i], marker=markers[i], linestyle=line_styles[i], label=hazard)
        i += 1
    if legend: 
        plt.legend(bbox_to_anchor=(1, 1.1), loc='upper left', fontsize=14)
    plt.xticks(np.arange(0, int(len(time_vals))+1, xtick_freq),rotation=45)
    plt.margins(x=0.05)
    plt.tick_params(labelsize=16)
    if save: 
        plt.savefig(dataset_name+'_hazard_'+metric_name+'.pdf', bbox_inches="tight") 
    plt.show()

def plot_metric_averages(metric_data, metric_name, show_std=True, title="", save=False, legend=True, dataset_name=""):
    import textwrap
    avg = {hazard: np.average([m for year in metric_data[hazard] for m in metric_data[hazard][year]]) for hazard in metric_data}
    stddev = {hazard: np.std([m for year in metric_data[hazard] for m in metric_data[hazard][year]]) for hazard in metric_data}
    x_pos = np.arange(len(metric_data))
    fig, ax = plt.subplots()
    colors = cm.tab20(np.linspace(0, 1, len(metric_data)))
    labels = [key for key in metric_data.keys()]
    ax.bar(x_pos, avg.values(), yerr=stddev.values(), align='center', ecolor='black', capsize=10, color=colors)
    plt.xlabel("Hazard", fontsize=16)
    plt.ylabel(metric_name, fontsize=16)
    plt.title(title, fontsize=16)
    ax.yaxis.grid(True)
    plt.tick_params(labelsize=14)
    if legend == True:
        ax.set_xticklabels([])
        handles = [plt.Rectangle((0,0),1,1, color=color) for color in colors]
        plt.legend(handles, labels, bbox_to_anchor=(1, 1.1), loc='upper left', fontsize=14)
    elif legend == False:
        labels = list(metric_data.keys())
        mean_length = np.mean([len(i) for i in labels])
        labels = ["\n".join(textwrap.wrap(i,mean_length)) for i in labels]
        ax.set_xticks(np.asarray([i for i in range(len(metric_data))]))
        ax.set_xticklabels(labels,rotation=45,ha="right",rotation_mode='anchor')
    if save: plt.savefig(dataset_name+'_hazard_bar_'+metric_name+'.pdf', bbox_inches="tight") 
    plt.show()
    
def plot_frequency_time_series(metric_data, metric_name='Frequency', line_styles=[], markers=[], title="", time_name="Year", xtick_freq=5, scale=True, save=False, dataset_name="", legend=True):
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
    plt.figure()
    plt.ylabel(y_label, fontsize=16)
    plt.xlabel(time_name, fontsize=16)
    plt.title(title, fontsize=16)
    i = 0
    for hazard in hazard_freqs_scaled:
        plt.plot(time_vals, hazard_freqs_scaled[hazard], color=colors[i], label=hazard, marker=markers[i], linestyle=line_styles[i])
        i += 1
    if legend: 
        plt.legend(bbox_to_anchor=(1, 1.1), loc='upper left', fontsize=14)
    plt.xticks(np.arange(0, int(len(time_vals))+1, xtick_freq),rotation=45)
    plt.margins(x=0.05)
    plt.tick_params(labelsize=16)
    if save: plt.savefig(dataset_name+'_hazard_'+metric_name+'.pdf', bbox_inches="tight") 
    plt.show()

def create_correlation_matrix(predictors_scaled, frequencies_scaled, graph=True, mask_vals=False, figsize=(4,4)):
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
        correlation_mat_data[hazard] = frequencies_scaled[hazard]
    correlation_mat_total = pd.DataFrame(correlation_mat_data)
    corrMatrix =correlation_mat_total.corr()
    p_values = corr_sig(correlation_mat_total)                     # get p-Value
    mask = np.invert(np.tril(p_values<0.05)) 
    if graph == True:
        fig,ax = plt.subplots(figsize=figsize)
        if mask_vals:
            sn.heatmap(corrMatrix, annot=True, mask=mask)
            plt.title("Correlational Matrix for Trends in \n Fires, Operations, Intensity, and Hazard Frequency per year")
            plt.show()
        else:
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
    sentences = []
    for hazard in ids:
        total_ids = [id_ for year in ids[hazard] for id_ in ids[hazard][year]]
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
                        worksheet.write_formula('F'+str(i+2),'{=E'+str(i+2)+'/C'+str(i+2)+'}')
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

def get_severities(severities): #SAFECOM
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

def sample_for_recall(preprocessed_df, id_col, text_col, hazards, save_path, num_sample=100):
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
    hazard_words_per_doc_cleaned = {hazard: [w for w in hazard_words_per_doc[hazard] if w!='none'] for hazard in hazard_words_per_doc}
    word_frequencies = {hazard:{np.unique(hazard_words_per_doc_cleaned[hazard], return_counts=True)[0][i]:np.unique(hazard_words_per_doc_cleaned[hazard], return_counts=True)[1][i] for i in range(len(np.unique(hazard_words_per_doc_cleaned[hazard], return_counts=True)[0]))} for hazard in hazard_words_per_doc_cleaned}
    if hazards_sorted:
        word_frequencies = {hazard: word_frequencies[hazard] for hazard in hazards_sorted}
    return word_frequencies

def build_word_clouds(word_frequencies, nrows, ncols, figsize=(8, 4), cmap=None, save=False, save_path=None, fontsize=10, wordcloud_kwargs={}):
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
    total_nums = [l for li in lists for l in li]
    topics, counts = np.unique(total_nums, return_counts=True)
    proposed_topics = [topics[i] for i in range(len(counts)) if counts[i]>=(len(lists)/3)]
    return proposed_topics
