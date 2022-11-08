# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 09:50:41 2022

@author: srandrad
"""

import os
import pandas as pd
import numpy as np
from transformers import Trainer, AutoTokenizer, DataCollatorForTokenClassification, BertForTokenClassification
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),"..","..", ".."))
from mika.kd.NER import *
from mika.kd import FMEA #import FMEA
from datasets import load_from_disk, Dataset
from torch import cuda
import random

def calc_severity(df):
    severities = []
    for i in range(len(df)):
        severities.append(safecom_severity(df.iloc[i]['Hazardous Materials'], df.iloc[i]['Injuries'], df.iloc[i]['Damages']))
    df['severity'] = severities
    return df

def safecom_severity(hazardous_mat, injury, damage):
    key_dict = {"No":0, "Yes":1}
    severity = key_dict[hazardous_mat] + key_dict[injury] + key_dict[damage]
    if np.isnan(severity):
        severity=0
    return severity

if __name__ == '__main__':
    
    model_checkpoint = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir, os.pardir)),"models", "FMEA-ner-model", "checkpoint-1424")
    print(model_checkpoint)
    #device = 'cuda' if cuda.is_available() else 'cpu'
    #cuda.empty_cache()
    device = 'cpu'
    print(device)
    
    fmea = FMEA()
    fmea.load_model(model_checkpoint)
    print("loaded model")
    
    file = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir, os.pardir)),"data/SAFECOM/SAFECOM_UAS_fire_data.csv")
    #TODO: join annotations to raw df
    input_data = fmea.load_data(filepath=file, formatted=False, text_col='Text', id_col='Tracking #')
    
    print("loaded data")
    preds = fmea.predict()
    df = fmea.get_entities_per_doc()
    fmea.display_doc(doc_id="21-0098", save=True, output_path="test", colors_path=os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir, os.pardir)),'data','doccano','NER_label_config.json'))
    #"""
    manual_cluster_file = os.path.join(os.getcwd(),"SAFECOM_UAS_clusters_V1.xlsx")
    fmea.group_docs_manual(manual_cluster_file, grouping_col='Mode', additional_cols=['Mission Type'])
    fmea.calc_severity(calc_severity)
    fmea.get_year_per_doc('Date')
    fmea.calc_frequency('Year')
    fmea.calc_risk()
    fmea.post_process_fmea(id_name='SAFECOM', max_words=10)
    #"""
    fmea.fmea_df.to_csv(os.path.join(os.getcwd(),"safecom_fmea_test.csv"))
    
