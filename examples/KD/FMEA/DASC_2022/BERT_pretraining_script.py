# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 09:34:34 2022

@author: srandrad
"""

import pandas as pd
import os
import numpy as np
from transformers import AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling
from transformers import TrainingArguments, Trainer
from torch import cuda
from datasets import Dataset, concatenate_datasets

device = 'cuda' if cuda.is_available() else 'cpu'
cuda.empty_cache()
print(device)

#load training data: LLIS, all of SAFECOM
safecom = pd.read_csv(os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir, os.pardir, os.pardir)),'data/SAFECOM/SAFECOM_data.csv'))
llis = pd.read_excel(os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir, os.pardir, os.pardir)),'data/LLIS/lessons_learned_2021-12-10.xlsx'))
print(len(llis), len(safecom))
safecom_text = safecom[['Narrative', 'Corrective Action']]
llis_text = llis[['Recommendation(s)', 'Lesson(s) Learned', 'Driving Event']]

model_checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, truncation=True, padding='max_length')#, return_special_tokens_mask=True)

text = []
for i in range(len(safecom_text)):
    text.append(safecom_text.iloc[i]['Narrative'])
    text.append(safecom_text.iloc[i]['Corrective Action'])
for i in range(len(llis_text)):
    text.append(llis_text.iloc[i]['Recommendation(s)'])
    text.append(llis_text.iloc[i]['Lesson(s) Learned'])
    text.append(llis_text.iloc[i]['Driving Event'])
text_df = pd.DataFrame({'Text':text})
text_df = text_df.dropna().reset_index(drop=True)

# set up train and eval dataset
train_size=0.8
train_dataset = text_df.sample(frac=train_size,random_state=200)
test_dataset = text_df.drop(train_dataset.index).reset_index(drop=True)
train_dataset = train_dataset.reset_index(drop=True)

        
def tokenize(text_df, tokenizer):
    tokenized_inputs = tokenizer(text_df["Text"], is_split_into_words=False, padding='max_length', 
                                 truncation=True, 
                                 return_special_tokens_mask=True 
                                 )#return_tensors="pt").to(device)#, padding=True, truncation=True)
    return tokenized_inputs

train_data = Dataset.from_pandas(train_dataset).map(tokenize,
    fn_kwargs={'tokenizer':tokenizer},
    remove_columns=['Text'])
test_data = Dataset.from_pandas(test_dataset).map(tokenize,
    fn_kwargs={'tokenizer':tokenizer},
    remove_columns=['Text'])

test_labels = Dataset.from_pandas(pd.DataFrame({'labels':test_data['input_ids'].copy()}))
train_labels = Dataset.from_pandas(pd.DataFrame({'labels':train_data['input_ids'].copy()}))
test_data = concatenate_datasets([test_data, test_labels], axis=1)
train_data = concatenate_datasets([train_data, train_labels], axis=1)

#initiating model
#model = AutoModelForTokenClassification.from_pretrained(model_checkpoint)#
model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)

model = model.to(device)
#training set up
#data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)#
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer)
args = TrainingArguments(
    os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir, os.pardir, os.pardir)),"models/Pre-trained-BERT"),
    evaluation_strategy="steps",
    save_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=10,
    weight_decay=0.01,
    push_to_hub=False,
    per_device_train_batch_size = 1,
    per_device_eval_batch_size = 1,
    logging_steps=5000,
    eval_steps = 5000,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_data,
    eval_dataset=test_data,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

#train_result = trainer.train()
#trainer.save_model()

