# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 16:27:10 2022

@author: srandrad
"""

import pandas as pd
import numpy as np
import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),".."))

from mika.utils import Data
from mika.kd.NER import plot_eval_results
from transformers import AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling
from transformers import TrainingArguments, Trainer
from torch import cuda
from datasets import Dataset, concatenate_datasets
import warnings
warnings.filterwarnings("ignore")
device = 'cuda' if cuda.is_available() else 'cpu'
cuda.empty_cache()
print(device)
import os
#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

#load training data: ASRS, NTSB
ASRS_file = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)),'data/ASRS/ASRS_1988_2022.csv')
ASRS_id_col = 'ACN'
ASRS_text_cols = ['Report 1', 'Report 1.1', 'Report 2',	'Report 2.1', 'Report 1.2']
ASRS = Data()
ASRS.load(ASRS_file, id_col=ASRS_id_col, text_columns=ASRS_text_cols)
ASRS_df = ASRS.data_df

NTSB_file = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)),'data/NTSB/ntsb_full_narratives.csv')
NTSB_id_col = 'ev_id'
NTSB_text_cols = ['narr_accp', 'narr_accf', 'narr_cause', 'narr_inc', 'REMARKS', 'CAUSE']
NTSB = Data()
NTSB.load(NTSB_file, id_col=NTSB_id_col, text_columns=NTSB_text_cols)
NTSB_df = NTSB.data_df

print(len(ASRS_df), len(NTSB_df))

model_checkpoint = "bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, truncation=True, padding='max_length', return_special_tokens_mask=True)

text = []

for col in ASRS_text_cols:
    text += ASRS_df[col].tolist()
for col in NTSB_text_cols:
    text += NTSB_df[col].to_list()

print("created new df of just text")
text_df = pd.DataFrame({'Text':text})
text_df = text_df.dropna().reset_index(drop=True)
text_df = text_df[:2000]
# set up train and eval dataset
train_size=0.8
train_dataset = text_df.sample(frac=train_size,random_state=200)
test_dataset = text_df.drop(train_dataset.index).reset_index(drop=True)
train_dataset = train_dataset.reset_index(drop=True)

print("defined training and test set")
def tokenize(text_df, tokenizer):
    tokenized_inputs = tokenizer(text_df["Text"], is_split_into_words=False, padding='max_length', 
                                 truncation=True, 
                                 return_special_tokens_mask=True)# , return_tensors="pt").to(device)
    #, padding=True, truncation=True)
    return tokenized_inputs

train_data = Dataset.from_pandas(train_dataset).map(tokenize,
    fn_kwargs={'tokenizer':tokenizer},
    remove_columns=['Text'])

test_data = Dataset.from_pandas(test_dataset).map(tokenize,
    fn_kwargs={'tokenizer':tokenizer},
    remove_columns=['Text'])

print("tokenized data")

test_labels = Dataset.from_pandas(pd.DataFrame({'labels':test_data['input_ids'].copy()}))
train_labels = Dataset.from_pandas(pd.DataFrame({'labels':train_data['input_ids'].copy()}))
test_data = concatenate_datasets([test_data, test_labels], axis=1)
train_data = concatenate_datasets([train_data, train_labels], axis=1)
test_data.set_format("torch")
train_data.set_format("torch")
#initiating model
model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)

print("model loaded")
model.to(device)
print(model.device)

#training set up
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer)
args = TrainingArguments(
    os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)),"models/SafeAeroBERT"),
    evaluation_strategy="steps",
    save_strategy="epoch",
    learning_rate=1e-3,
    num_train_epochs=1,
    weight_decay=0.01,
    push_to_hub=False,
    per_device_train_batch_size = 8,#256,
    per_device_eval_batch_size = 8,#256,
    logging_steps=1,
    eval_steps = 1,
    save_total_limit = 3, #saves only last 3 checkpoints
    gradient_accumulation_steps=32,#64,
    gradient_checkpointing=True,
    fp16=True,
    optim="adafactor"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_data,
    eval_dataset=test_data,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

train_result = trainer.train()
trainer.save_model()
final_train_metrics = train_result.metrics
num_steps = trainer.state.max_steps
filename = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)),"models", "SafeAeroBERT", "checkpoint-"+str(num_steps), "trainer_state.json")
plot_eval_results(filename, save=True, savepath="SafeAeroBERT_", final_train_metrics=final_train_metrics, loss=True, metrics=False)

r""" #get categories
df = pd.read_excel(r"C:\Users\srandrad\OneDrive - NASA\Desktop\ASRS_DBOnline.xlsx")
contributing_factors = [f for factors in df['Assessments'].tolist()[2:] if type(factors) is str for f in factors.split("; ") ]
print(np.unique(contributing_factors, return_counts=True))
factor, counts = np.unique(contributing_factors, return_counts=True)
[print(factor[i], counts[i]) for i in range(len(factor))]"""