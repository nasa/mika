# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 10:04:44 2021
Functions for ICS-209-PLUS
@author: srandrad
"""

import sys
import os
sys.path.append(os.path.join(".."))

from module.topic_model_plus_class import Topic_Model_plus
from module.stopwords.ICS_stop_words import stop_words
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic
import os
import pandas as pd 



if __name__ == '__main__': 
    ICS_stop_words = stop_words
    
    extra_cols = ["CY","DISCOVERY_DATE", "START_YEAR", "REPORT_DOY", "DISCOVERY_DOY",
                  "TOTAL_PERSONNEL", "TOTAL_AERIAL", "PCT_CONTAINED_COMPLETED",
                  "ACRES", "WF_FSR", "INJURIES", "FATALITIES","EST_IM_COST_TO_DATE", "STR_DAMAGED",
                  "STR_DESTROYED", "NEW_ACRES","POO_STATE", "POO_LATITUDE", "POO_LONGITUDE", "WEATHER_CONCERNS_NARR", "INC_MGMT_ORG_ABBREV",
                  "EVACUATION_IN_PROGRESS"]
    #"""#Preprocess data
    list_of_attributes = ["REMARKS", "SIGNIF_EVENTS_SUMMARY", "MAJOR_PROBLEMS"]
    document_id_col = "INCIDENT_ID"
    
    file_name = os.path.join('data','209-PLUS','ics209-plus-wildfire','ics209-plus-wildfire','ics209-plus-wf_sitreps_1999to2014.csv')
    name = os.path.join('ICS_bertopic')
    
    ICS = Topic_Model_plus(document_id_col=document_id_col, extra_cols=extra_cols, csv_file=file_name, list_of_attributes=list_of_attributes, database_name=name, combine_cols=True, create_ids=True)
    ICS.prepare_data(dtype=str)
    save_words = ['jurisdictions', 'team', 'command', 'organization', 'type', 'involved', 'transition', 'transfer', 'impact', 'concern', 'site', 'nation', 'political', 'social', 'adjacent', 'community', 'cultural', 'tribal', 'monument', 'archeaological', 'highway', 'traffic', 'road', 'travel', 'interstate', 'closure', 'remain', 'remains', 'close', 'block', 'continue', 'impact', 'access', 'limit', 'limited', 'terrain', 'rollout', 'snag', 'steep', 'debris', 'access', 'terrian', 'concern', 'hazardous', 'pose', 'heavy', 'rugged', 'difficult', 'steep', 'narrow', 'violation', 'notification', 'respond', 'law', 'patrol', 'cattle', 'buffalo', 'grow', 'allotment', 'ranch', 'sheep', 'livestock', 'grazing', 'pasture', 'threaten', 'concern', 'risk', 'threat', 'evacuation', 'evacuate', ' threaten', 'threat', 'resident', ' residence', 'level', 'notice', 'community', 'structure', 'subdivision', 'mandatory', 'order', 'effect', 'remain', 'continue', 'issued', 'issue', 'injury', 'hospital', 'injured', 'accident', 'treatment', 'laceration', 'firefighter', 'treated', 'minor', 'report', 'transport', 'heat', 'shoulder', 'ankle', 'medical', 'released', 'military', 'unexploded', 'national', 'training', 'present', 'ordinance', 'guard', 'infrastructure', 'utility', 'powerline', 'water', 'electric', 'pipeline', 'powerlines', 'watershed', 'pole', 'power', 'gas', 'concern', 'near', 'hazard', 'critical', 'threaten', 'threat', 'off', 'weather', 'behavior', 'wind', 'thunderstorm', 'storm', 'gusty', 'lightning', 'flag', 'unpredictable', 'extreme', 'erratic', 'strong', 'red', 'warning', 'species', 'specie', 'habitat', 'animal', 'plant', 'conservation', 'threaten', 'endanger', 'threat', 'sensitive', 'threatened', 'endangered', 'risk', 'loss', 'impacts', 'unstaffed', 'resources', 'support', 'crew', 'aircraft', 'helicopter', 'engines', 'staffing', 'staff', 'lack', 'need', 'shortage', 'minimal', 'share', 'necessary', 'limited', 'limit', 'fatigue', 'flood', 'flashflood', 'flash', 'risk', 'potential', 'mapping', 'map', 'reflect', 'accurate', 'adjustment', 'change', 'reflect', 'aircraft', 'heli', 'helicopter', 'aerial', 'tanker', 'copter', 'grounded', 'ground', 'suspended', 'suspend', 'smoke', 'impact', 'hazard', 'windy', 'humidity', 'moisture', 'hot', 'drought', 'low', 'dry', 'prolonged']
    #use filtered reports
    file = os.path.join('data','ICS_data','summary_reports_cleaned.csv')
    filtered_df = pd.read_csv(file)
    filtered_ids = filtered_df['INCIDENT_ID'].unique()
    ICS.data_df = ICS.data_df.loc[ICS.data_df['INCIDENT_ID'].isin(filtered_ids)].reset_index(drop=True)
    ICS.doc_ids = ICS.data_df['Unique IDs'].tolist()
    raw_text = ICS.data_df[ICS.list_of_attributes]
    # break up df into sentences 
    #ICS.preprocess_data(domain_stopwords=ICS_stop_words, ngrams=False, min_count=1, save_words=save_words)
    ICS.split_doc_to_sentences()
    ICS.save_preprocessed_data()
    
    #use filtered reports
    """
    file = os.path.join('data','ICS_data','ICS_filtered_preprocessed_combined_data.csv')
    filtered_df = pd.read_csv(file)
    filtered_ids = filtered_df['INCIDENT_ID'].unique()
    ICS.data_df = ICS.data_df.loc[ICS.data_df['INCIDENT_ID'].isin(filtered_ids)].reset_index(drop=True)
    save_words = ['jurisdictions', 'team', 'command', 'organization', 'type', 'involved', 'transition', 'transfer', 'impact', 'concern', 'site', 'nation', 'political', 'social', 'adjacent', 'community', 'cultural', 'tribal', 'monument', 'archeaological', 'highway', 'traffic', 'road', 'travel', 'interstate', 'closure', 'remain', 'remains', 'close', 'block', 'continue', 'impact', 'access', 'limit', 'limited', 'terrain', 'rollout', 'snag', 'steep', 'debris', 'access', 'terrian', 'concern', 'hazardous', 'pose', 'heavy', 'rugged', 'difficult', 'steep', 'narrow', 'violation', 'notification', 'respond', 'law', 'patrol', 'cattle', 'buffalo', 'grow', 'allotment', 'ranch', 'sheep', 'livestock', 'grazing', 'pasture', 'threaten', 'concern', 'risk', 'threat', 'evacuation', 'evacuate', ' threaten', 'threat', 'resident', ' residence', 'level', 'notice', 'community', 'structure', 'subdivision', 'mandatory', 'order', 'effect', 'remain', 'continue', 'issued', 'issue', 'injury', 'hospital', 'injured', 'accident', 'treatment', 'laceration', 'firefighter', 'treated', 'minor', 'report', 'transport', 'heat', 'shoulder', 'ankle', 'medical', 'released', 'military', 'unexploded', 'national', 'training', 'present', 'ordinance', 'guard', 'infrastructure', 'utility', 'powerline', 'water', 'electric', 'pipeline', 'powerlines', 'watershed', 'pole', 'power', 'gas', 'concern', 'near', 'hazard', 'critical', 'threaten', 'threat', 'off', 'weather', 'behavior', 'wind', 'thunderstorm', 'storm', 'gusty', 'lightning', 'flag', 'unpredictable', 'extreme', 'erratic', 'strong', 'red', 'warning', 'species', 'specie', 'habitat', 'animal', 'plant', 'conservation', 'threaten', 'endanger', 'threat', 'sensitive', 'threatened', 'endangered', 'risk', 'loss', 'impacts', 'unstaffed', 'resources', 'support', 'crew', 'aircraft', 'helicopter', 'engines', 'staffing', 'staff', 'lack', 'need', 'shortage', 'minimal', 'share', 'necessary', 'limited', 'limit', 'fatigue', 'flood', 'flashflood', 'flash', 'risk', 'potential', 'mapping', 'map', 'reflect', 'accurate', 'adjustment', 'change', 'reflect', 'aircraft', 'heli', 'helicopter', 'aerial', 'tanker', 'copter', 'grounded', 'ground', 'suspended', 'suspend', 'smoke', 'impact', 'hazard', 'windy', 'humidity', 'moisture', 'hot', 'drought', 'low', 'dry', 'prolonged']
    ICS.preprocess_data(domain_stopwords = ICS_stop_words, percent=0.5, ngrams=False, min_count=1, save_words=save_words)
    ICS.save_preprocessed_data()
    """
    
    #BERTopic
    #"""
    from nltk.corpus import stopwords
    total_stopwords = stopwords.words('english')+ICS_stop_words
    #ICS.data_df[ICS.list_of_attributes] = raw_text #uses raw text
    vectorizer_model = CountVectorizer(ngram_range=(1, 3), stop_words=total_stopwords) #removes stopwords
    seed_topic_list = [['highway', 'traffic', 'road', 'travel', 'interstate', 'closure', 'remain', 'remains', 'close', 'block', 'impact', 'access', 'limit', 'limited'], 
                       ['transition', 'transfer'], 
                       ['evacuation', 'evacuate',], 
                       ['mapping', 'map', 'reflect', 'accurate', 'adjustment', 'change', 'reflect', 'inaccurate'], 
                       ['aerial','inversion', 'suspend', 'suspendsion', 'prohibit', 'delay', 'hamper', 'unable', 'cancel', 'inability', 'loss', 'curtail', 'challenge', 'smoke'], 
                       ['unstaffed', 'resource', 'lack', 'need', 'shortage', 'minimal', 'share', 'necessary', 'limited', 'limit', 'fatigue'], 
                       ['injury', 'hospital', 'injured', 'accident', 'treatment', 'laceration', 'firefighter', 'treat', 'minor', 'report', 'transport', 'heat', 'shoulder', 'ankle', 'medical', 'release'], 
                       ['cultural', 'tribal', 'monument', 'archaeological', 'heritage', 'site', 'nation', 'political', 'social', 'adjacent', 'community'], 
                       ['cattle', 'buffalo', 'allotment', 'ranch', 'sheep', 'livestock', 'grazing', 'pasture', 'threaten', 'concern', 'risk', 'threat', 'private', 'area', 'evacuate', 'evacuation', 'order'], 
                       ['violation', 'arson', 'notification', 'respond', 'law'], 
                       ['military', 'unexploded', 'training', 'present', 'ordinance', 'proximity', 'activity', 'active', 'base', 'area'], 
                       ['infrastructure', 'utility', 'powerline', 'water', 'electric', 'pipeline', 'powerlines', 'watershed', 'pole', 'power', 'gas'], 
                       ['weather', 'behavior', 'wind', 'thunderstorm', 'storm', 'gusty', 'lightning', 'flag', 'unpredictable', 'extreme', 'erratic', 'strong', 'red', 'warning', 'warn'], 
                       ['species', 'habitat', 'animal', 'plant', 'conservation', 'threaten', 'endanger', 'threat', 'sensitive', 'risk', 'loss', 'impact'], 
                       ['terrain', 'rollout', 'snag', 'steep', 'debris', 'access', 'concern', 'hazardous', 'pose', 'heavy', 'rugged', 'difficult', 'steep', 'narrow'], 
                       ['humidity', 'moisture', 'hot', 'drought', 'low', 'dry', 'prolong']]
    
    """#load bertopic model
    bert_model = BERTopic.load(model_path, embedding_model="all-MiniLM-L6-v2")
    ICS.BERT_models = {}
    ICS.BERT_models["Combined Text"] = bert_model
    topics, probs = ICS.BERT_models["Combined Text"].transform(ICS.data_df['Combined Text'])
    ICS.BERT_model_topics_per_doc["Combined Text"] = topics
    ICS.BERT_model_probs["Combined Text"] = probs
    ICS.reduced = True
    ICS.save_bert_results()
    """
    
    BERTkwargs={"calculate_probabilities":True, 'seed_topic_list':seed_topic_list,
                "top_n_words": 20, 'min_topic_size':150}
    ICS.bert_topic(count_vectorizor=vectorizer_model, BERTkwargs=BERTkwargs, from_probs=True)
    ICS.save_bert_results(from_probs=True) #warning: saving in excel can result in missing data when char limit is reached
    ICS.save_bert_topics_from_probs()
    #get coherence
    ICS.save_bert_coherence(coh_method='c_v')
    ICS.save_bert_coherence(coh_method='c_npmi')
    ICS.save_bert_vis()
    ICS.save_bert_model()
    """
    ICS.reduce_bert_topics(num=100, from_probs=True)
    ICS.save_bert_results(from_probs=True)  #warning: saving in excel can result in missing data when char limit is reached
    ICS.save_bert_vis()
    ICS.save_bert_coherence(coh_method='c_v')
    ICS.save_bert_coherence(coh_method='c_npmi')
    ICS.save_bert_model()
    ICS.save_bert_topics_from_probs()
    """
    #"""
    
    """#Extract preprocessed data
    list_of_attributes = ["Combined Text"]
    name =  smart_nlp_path+r"\output data\ICS_50_topics"
    file = smart_nlp_path+r"\input data\ICS_filtered_preprocessed_combined_data.csv"
    filtered_df = pd.read_csv(file)
    filtered_ids = filtered_df['INCIDENT_ID'].unique()
    document_id_col = "Unique IDs"
    ICS = Topic_Model_plus(document_id_col=document_id_col, extra_cols=extra_cols, list_of_attributes=list_of_attributes, name=name, combine_cols=False)
    ICS.extract_preprocessed_data(file)
    """
    
    """#Run topic modeling
    list_of_attributes = ["Combined Text"]
    #ICS.lda_optimization(min_cf=5, max_topics = 200)
    num_topics = {attr:50 for attr in list_of_attributes}
    ICS.lda(min_cf=1, num_topics=num_topics, min_df=1, alpha=1, eta=0.0001)
    ICS.save_lda_results()
    ICS.save_lda_models()
    for attr in list_of_attributes:
        ICS.lda_visual(attr)
    """
    """#Run hlda
    ICS.hlda(levels=3, eta=0.50, min_cf=1, min_df=1)
    ICS.save_hlda_models()
    ICS.save_hlda_results()
    for attr in list_of_attributes:
        ICS.hlda_visual(attr)
    """
    
    """#Run Results from extracted models
    list_of_attributes = ["Combined Text"]
    document_id_col = "Unique IDs"
    ICS = Topic_Model_plus(document_id_col=document_id_col, extra_cols=extra_cols, list_of_attributes=list_of_attributes, combine_cols=False)
    ICS.combine_cols = True
    filepath = smart_nlp_path+r"\output data\ICS__combined_topics-May-26-2021"
    ICS.hlda_extract_models(filepath)
    ICS.save_hlda_results()
    """
