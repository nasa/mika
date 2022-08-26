# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 12:01:37 2022

@author: srandrad
"""
import pandas as pd
import numpy as np
import os
from summarizer.sbert import SBertSummarizer
from transformers import pipeline


safecom = pd.read_csv(os.path.join('data', 'SAFECOM_data.csv'), index_col=0)
UAS_safecom = safecom.loc[~(safecom['UAS'].isna())|(safecom['Type']=='Unmanned Aircraft System (UAS)')].reset_index(drop=True)
UAS_safecom['Mission Type'].unique()
non_fire_mission_types = ['Other', 'Wildlife, Animal Counting', 'Training, Other',
                          'Research', np.nan, 'Reconnaissance (Non-Fire)', 'Aerial Photography',
                          'Training, Pilot','Survey/Forest Health Protection (Non-Fire)','Training, Aircrew',
                          'Proficiency, Pilot', 'Wildlife, Animal Survey','Inspection (Aircraft)',
                          'Survey/Observation (Non-Fire)','Passenger Transport (Non-Fire)','Search/Rescue',
                          'Law Enforcement','Air Quality Monitoring','Offshore', 'Maintenance Test Flight'
                         ]
UAS_safecom_fire = UAS_safecom.loc[~UAS_safecom['Mission Type'].isin(non_fire_mission_types)].reset_index(drop=True)


model = SBertSummarizer('paraphrase-MiniLM-L6-v2')
body = UAS_safecom_fire['Narrative'].to_list()
narratives = []
for text in body:
    if type(text) is str:
        narr = model(text, num_sentences=3)
        narratives.append(narr)
    else:
        narratives.append(text)
body = UAS_safecom_fire['Corrective Action'].to_list()
correctives = []
for text in body:
    if type(text) is str:
        corrective = model(text, num_sentences=1)
        correctives.append(corrective)
    else: 
        correctives.append(text)
UAS_safecom_fire['Extractive Summarized Narrative'] = narratives
UAS_safecom_fire['Extractive Summarized Corrective Action'] = correctives

# using pipeline API for summarization task
"""
summarization = pipeline("summarization")
narratives = []
body = UAS_safecom_fire['Narrative'].to_list()
for text in body:
    if type(text) is str:
        max_len = min(30, len(text))
        narr = summarization(text,min_length=0, max_length=max_len)[0]['summary_text']
        narratives.append(narr)
    else:
        narratives.append(text)
body = UAS_safecom_fire['Corrective Action'].to_list()
correctives = []
for text in body:
    if type(text) is str:
        max_len = min(30, len(text))
        print(text, max_len, body.index(text))
        corrective = summarization(text,min_length=0, max_length=max_len)[0]['summary_text']
        correctives.append(corrective)
    else: 
        correctives.append(text)"""
from transformers import T5ForConditionalGeneration, T5Tokenizer

# initialize the model architecture and weights
model = T5ForConditionalGeneration.from_pretrained("t5-base")
# initialize the model tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-base")

narratives = []
body = UAS_safecom_fire['Narrative'].to_list()
for text in body:
    if type(text) is str:
        # encode the text into tensor of integers using the appropriate tokenize
        inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
        # generate the summarization output
        outputs = model.generate(
            inputs, 
            max_length=150, 
            min_length=40, 
            length_penalty=2.0, 
            num_beams=4, 
            early_stopping=True)
        narr = tokenizer.decode(outputs[0])
        narratives.append(narr)
    else:
        narratives.append(text)
body = UAS_safecom_fire['Corrective Action'].to_list()
correctives = []
for text in body:
    if type(text) is str:
        # encode the text into tensor of integers using the appropriate tokenize
        inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
        # generate the summarization output
        outputs = model.generate(
            inputs, 
            max_length=150, 
            min_length=40, 
            length_penalty=2.0, 
            num_beams=4, 
            early_stopping=True)
        corrective  = tokenizer.decode(outputs[0])
        correctives.append(corrective)
    else: 
        correctives.append(text)
        
UAS_safecom_fire['Abstractive Summarized Narrative'] = narratives
UAS_safecom_fire['Abstractive Summarized Corrective Action'] = correctives
UAS_safecom_fire.to_csv(os.path.join('data', 'SAFECOM_UAS_fire_data.csv'))
print(UAS_safecom_fire.iloc[0][:])