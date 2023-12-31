# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 16:27:10 2022

@author: srandrad
"""

import pandas as pd
import numpy as np
import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),".."))

from time import sleep
from mika.utils import Data
from mika.kd.NER import plot_eval_results
from transformers import AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling
from transformers import TrainingArguments, Trainer
from torch import cuda
from datasets import Dataset, concatenate_datasets
device = 'cuda' if cuda.is_available() else 'cpu'
cuda.empty_cache()
print(device)
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# load training data: ASRS, NTSB
ASRS_file = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)),'data/ASRS/ASRS_1988_2022_cleaned.csv')
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
train_dataset = text_df

train_size=0.9
train_dataset = text_df.sample(frac=train_size,random_state=200)
test_dataset = text_df.drop(train_dataset.index).reset_index(drop=True)
train_dataset = train_dataset.reset_index(drop=True)

print("defined training set")
def tokenize(text_df, tokenizer):
    tokenized_inputs = tokenizer(text_df["Text"], is_split_into_words=False, padding='max_length', 
                                 truncation=True, 
                                 return_special_tokens_mask=True)
    return tokenized_inputs

train_data = Dataset.from_pandas(train_dataset).map(tokenize,
                                                    fn_kwargs={'tokenizer':tokenizer},
                                                    remove_columns=['Text'],
                                                    batched=True, num_proc=4)

test_data = Dataset.from_pandas(test_dataset).map(tokenize,
                                                  fn_kwargs={'tokenizer':tokenizer},
                                                  remove_columns=['Text'],
                                                  batched=True, num_proc=4)

#num_tokens = sum([len(tokens) for tokens in train_data['tokens']])

print("tokenized data")
print(min(min(train_data['input_ids'].copy())))

train_labels = Dataset.from_pandas(pd.DataFrame({'labels':train_data['input_ids'].copy()}))
train_data = concatenate_datasets([train_data, train_labels], axis=1)
train_data.set_format("torch")

test_labels = Dataset.from_pandas(pd.DataFrame({'labels':test_data['input_ids'].copy()}))
test_data = concatenate_datasets([test_data, test_labels], axis=1)
test_data.set_format("torch")

#initiating model
model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)

print("model loaded")
#model.to(1)
model.cuda()
print(model.device)

#training set up
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
args = TrainingArguments(
    os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)),"models/SafeAeroBERT_final"),
    save_strategy="steps",
    learning_rate=1e-5,
    evaluation_strategy="no",
    num_train_epochs=2,
    weight_decay=0.01,
    push_to_hub=False,
    per_device_train_batch_size = 8,#256,
    save_steps = 5,
    logging_steps = 100,
    save_total_limit = 5, #saves only last 5 checkpoints
    gradient_accumulation_steps=16,#64,
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


trainer.train()
trainer.save_model()
num_steps = trainer.state.max_steps
filename = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)),"models", "SafeAeroBERT_final", "checkpoint-"+str(num_steps), "trainer_state.json")
plot_eval_results(filename, save=True, savepath="SafeAeroBERT_", loss=True, metrics=False)
