# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 16:02:16 2022
models init

@author: srandrad
"""

from transformers import pipeline
import os 

#NER_model_checkpoint = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)), "models", "FMEA-ner-model", "checkpoint-1424")
#FMEA_NER = pipeline("token-classification", model=NER_model_checkpoint, aggregation_strategy="simple")