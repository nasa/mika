# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 10:04:44 2021
Functions for ICS-209-PLUS
@author: srandrad
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),"..","..","..", ".."))

from mika.kd import Topic_Model_plus
from mika.utils import Data
from mika.utils.stopwords.ICS_stop_words import stop_words
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
    #Preprocess data
    text_columns = ["REMARKS", "SIGNIF_EVENTS_SUMMARY", "MAJOR_PROBLEMS"]
    document_id_col = "INCIDENT_ID"
    
    file_name = os.path.join('data','ICS','ics209-plus-wf_sitreps_1999to2014.csv')
    name = os.path.join('ICS_bertopic')
    ICS_data = Data()
    ICS_data.load(file_name, id_col=document_id_col, text_columns=text_columns, load_kwargs={'dtype':str})
    ICS_data.prepare_data(create_ids=True, combine_columns=text_columns, remove_incomplete_rows=True)
    ICS_data.text_columns = ['Combined Text']
    save_words = ['jurisdictions', 'team', 'command', 'organization', 'type', 'involved', 'transition', 'transfer', 'impact', 'concern', 'site', 'nation', 'political', 'social', 'adjacent', 'community', 'cultural', 'tribal', 'monument', 'archeaological', 'highway', 'traffic', 'road', 'travel', 'interstate', 'closure', 'remain', 'remains', 'close', 'block', 'continue', 'impact', 'access', 'limit', 'limited', 'terrain', 'rollout', 'snag', 'steep', 'debris', 'access', 'terrian', 'concern', 'hazardous', 'pose', 'heavy', 'rugged', 'difficult', 'steep', 'narrow', 'violation', 'notification', 'respond', 'law', 'patrol', 'cattle', 'buffalo', 'grow', 'allotment', 'ranch', 'sheep', 'livestock', 'grazing', 'pasture', 'threaten', 'concern', 'risk', 'threat', 'evacuation', 'evacuate', ' threaten', 'threat', 'resident', ' residence', 'level', 'notice', 'community', 'structure', 'subdivision', 'mandatory', 'order', 'effect', 'remain', 'continue', 'issued', 'issue', 'injury', 'hospital', 'injured', 'accident', 'treatment', 'laceration', 'firefighter', 'treated', 'minor', 'report', 'transport', 'heat', 'shoulder', 'ankle', 'medical', 'released', 'military', 'unexploded', 'national', 'training', 'present', 'ordinance', 'guard', 'infrastructure', 'utility', 'powerline', 'water', 'electric', 'pipeline', 'powerlines', 'watershed', 'pole', 'power', 'gas', 'concern', 'near', 'hazard', 'critical', 'threaten', 'threat', 'off', 'weather', 'behavior', 'wind', 'thunderstorm', 'storm', 'gusty', 'lightning', 'flag', 'unpredictable', 'extreme', 'erratic', 'strong', 'red', 'warning', 'species', 'specie', 'habitat', 'animal', 'plant', 'conservation', 'threaten', 'endanger', 'threat', 'sensitive', 'threatened', 'endangered', 'risk', 'loss', 'impacts', 'unstaffed', 'resources', 'support', 'crew', 'aircraft', 'helicopter', 'engines', 'staffing', 'staff', 'lack', 'need', 'shortage', 'minimal', 'share', 'necessary', 'limited', 'limit', 'fatigue', 'flood', 'flashflood', 'flash', 'risk', 'potential', 'mapping', 'map', 'reflect', 'accurate', 'adjustment', 'change', 'reflect', 'aircraft', 'heli', 'helicopter', 'aerial', 'tanker', 'copter', 'grounded', 'ground', 'suspended', 'suspend', 'smoke', 'impact', 'hazard', 'windy', 'humidity', 'moisture', 'hot', 'drought', 'low', 'dry', 'prolonged']
    #use filtered reports
    file = os.path.join('data','ICS','summary_reports_cleaned.csv')
    filtered_df = pd.read_csv(file)
    filtered_ids = filtered_df['INCIDENT_ID'].unique()
    ICS_data.data_df = ICS_data.data_df.loc[ICS_data.data_df['INCIDENT_ID'].isin(filtered_ids)].reset_index(drop=True)
    ICS_data.doc_ids = ICS_data.data_df['Unique IDs'].tolist()
    raw_text = ICS_data.data_df[ICS_data.text_columns]
    ICS_data.sentence_tokenization()
    ICS_tm = Topic_Model_plus(text_columns=['Combined Text Sentences'], data=ICS_data)

    #BERTopic
    from nltk.corpus import stopwords
    total_stopwords = stopwords.words('english')+ICS_stop_words
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
    
    BERTkwargs={"calculate_probabilities":True, 'seed_topic_list':seed_topic_list,
                "top_n_words": 20, 'min_topic_size':150}
    ICS_tm.bert_topic(count_vectorizor=vectorizer_model, BERTkwargs=BERTkwargs, from_probs=True)
    ICS_tm.save_bert_results(from_probs=True) #warning: saving in excel can result in missing data when char limit is reached
    ICS_tm.save_bert_topics_from_probs()
    #get coherence
    ICS_tm.save_bert_coherence(coh_method='c_v')
    ICS_tm.save_bert_coherence(coh_method='c_npmi')
    ICS_tm.save_bert_vis()
    ICS_tm.save_bert_model()