# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 09:50:41 2022

@author: srandrad
"""

import os
import pandas as pd
import numpy as np
from transformers import Trainer, AutoTokenizer, DataCollatorForTokenClassification, BertForTokenClassification
from module.NER_utils import *
from module.FMEA_class import FMEA
from datasets import load_from_disk, Dataset
from torch import cuda

model_checkpoint = os.path.join(os.getcwd(),"models", "FMEA-ner", "checkpoint-4490")
device = 'cuda' if cuda.is_available() else 'cpu'
cuda.empty_cache()
print(device)

fmea = FMEA()
fmea.load_model(model_checkpoint)
#file = "data/SAFECOM_UAS_fire_data.csv"
file = "data/NER_test_dataset"
fmea.load_data(file, formatted=True)
fmea.predict()
metrics = fmea.evaluate_preds()
print(metrics["Confusion Matrix"])
