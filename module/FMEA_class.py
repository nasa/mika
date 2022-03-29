# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 16:46:38 2022

@author: srandrad
"""
import os
import pandas as pd
import numpy as np
from transformers import Trainer, AutoTokenizer, DataCollatorForTokenClassification, BertForTokenClassification
from module.NER_utils import build_confusion_matrix, compute_classification_report
from datasets import load_from_disk, Dataset
from torch import cuda

class FMEA():
    def __init__(self):
        return
    def load_model(self, model_checkpoint):
        self.checkpoint = model_checkpoint
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        self.data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer)
        self.model = BertForTokenClassification.from_pretrained(model_checkpoint)
        self.trainer = Trainer(self.model, data_collator=self.data_collator)
        self.id2label = self.model.config.id2label

    def load_data(self, filepath, formatted=False, text_col="Narrative", id_col="Tracking #", label_col="labels"):
        if formatted == True:
            self.input_data = load_from_disk(filepath)
            self.true_labels = self.input_data[label_col]
        elif formatted == False:
            test_data = pd.read_csv(filepath, index_col=0)
            test_data = test_data.dropna(subset=text_col)
            self.data_ids = test_data["Tracking #"].tolist()
            self.data_df = test_data
            if label_col in self.data_df.columns:
                self.true_labels = self.data_df[label_col].to_list()
            input_data = self.tokenizer(test_data[text_col].tolist(),padding=True, truncation=True)
            self.input_data = Dataset(input_data)
            
    
    def predict(self):
        self.raw_pred, self.label_ids, self.pred_metric = self.trainer.predict(self.input_data)
        return self.raw_pred, self.label_ids, self.pred_metric
    
    def evaluate_preds(self, cm=True, class_report=True):
        return_vals = {}
        if cm == True:
            cm, true_predictions, labels = build_confusion_matrix(self.true_labels, self.raw_pred, self.label_ids, self.id2label)
            return_vals['Confusion Matrix'] = cm
        if class_report == True:
            classification_report = compute_classification_report(self.true_labels, self.raw_pred, self.label_ids, self.id2label)
            return_vals['Classification Report'] = classification_report
        return return_vals
    
    def get_entities_per_doc(self):
        return
    
    def group_docs(self):
        return
    
    def calc_frequency(self):
        return
    
    def calc_severity(self):
        return
    
    def calc_risk(self):
        return
    
    def build_fmea(self):
        self.get_entities_per_doc()
        self.group_docs()
        self.calc_frequency()
        self.calc_severity()
        self.calc_risk()
        return
    