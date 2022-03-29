# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 13:28:23 2022

@author: srandrad
"""

import pandas as pd
import os
from spacy.training import offsets_to_biluo_tags
import spacy
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification
from transformers import TrainingArguments, Trainer
from torch import cuda
from datasets import Dataset
#import seaborn as sn
#import matplotlib.pyplot as plt
#from matplotlib.colors import LogNorm
#import copy
#import matplotlib.cm as cm 

from module.NER_utils import read_doccano_annots, clean_doccano_annots, split_docs_to_sentances, check_doc_to_sentence_split, tokenize_and_align_labels, compute_metrics, compute_classification_report, build_confusion_matrix, plot_eval_results

#set up GPU
device = 'cuda' if cuda.is_available() else 'cpu'
cuda.empty_cache()
print(device)

#load spacy sentencizer
nlp = spacy.load("en_core_web_trf")
nlp.add_pipe("sentencizer")

file = os.path.join('data','UAS_fire_SAFECOM_annotated_SA.jsonl')
#read in doccano annotations
df = read_doccano_annots(file)
text_df = df[['data', 'label', 'Tracking #']] #"Lesson ID"]]

#clean doccano annotations
text_df = clean_doccano_annots(text_df)

#convert text to spacy doc 
text_data = text_df['data'].tolist()
docs = [nlp(doc) for doc in text_data]
print(text_df.columns)
text_df['docs'] = docs#[nlp(doc) for doc in text_data]

#convert offset labels to biluo tags
text_df['tags'] = [offsets_to_biluo_tags(text_df.at[i,'docs'], text_df.at[i,'label']) for i in range(len(text_df))]

#check for tags with issues
inds_with_issues = [i for i in range(len(text_df)) if '-' in text_df.iloc[i]['tags']]
text_df_issues = text_df.iloc[inds_with_issues][:].reset_index(drop=True)
if len(text_df_issues)>0: print("error: there are documents with invalid tags")

#set up labels
total_tags= pd.DataFrame({'tags':[t for tag in text_df['tags'] for t in tag]})
labels_to_ids = {k: v for v, k in enumerate(total_tags['tags'].unique())}
ids_to_labels = {v: k for v, k in enumerate(total_tags['tags'].unique())}

#look at frequencies
#print("Number of tags: {}".format(len(total_tags['tags'].unique())))
#frequencies = total_tags['tags'].value_counts()
#print(frequencies)

#prepare dataset and dataloader
#convert df from docs to sentences
sentence_df = split_docs_to_sentances(text_df, id_col='Tracking #')
check_doc_to_sentence_split(sentence_df)

#convert labels to numerical tags
ner_tags = [[labels_to_ids[label] for label in labels] for labels in sentence_df['tags']]
sentence_df['ner_tags'] = ner_tags

#load tokenizer
sentence_df['tokens'] = [[tok.orth_ for tok in sentence_df.iloc[i]['sentence']] for i in range(len(sentence_df))]

model_checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

train_size = 0.8

train_dataset = sentence_df.sample(frac=train_size,random_state=200)
test_dataset = sentence_df.drop(train_dataset.index).reset_index(drop=True)
train_dataset = train_dataset.reset_index(drop=True)

print("FULL Dataset: {}".format(sentence_df.shape))
print("TRAIN Dataset: {}".format(train_dataset.shape))
print("TEST Dataset: {}".format(test_dataset.shape))

label_list = total_tags['tags'].unique()

print(len(label_list))

cols_to_drop = [col for col in train_dataset.columns if col not in ["tokens", 'ner_tags', 'Tracking #']]
train_dataset = train_dataset.drop(cols_to_drop, axis=1)
test_dataset = test_dataset.drop(cols_to_drop, axis=1)
train_data = Dataset.from_pandas(train_dataset).map(
    tokenize_and_align_labels,
    batched=True,
    fn_kwargs={'tokenizer':tokenizer})
test_data = Dataset.from_pandas(test_dataset).map(
    tokenize_and_align_labels,
    batched=True,
    fn_kwargs={'tokenizer':tokenizer})

#test_data.save_to_disk("data/NER_test_dataset")
#train_data.save_to_disk("data/NER_train_dataset")

#initiating model
model =AutoModelForTokenClassification.from_pretrained( #BertForSequenceClassification.from_pretrained(
    model_checkpoint,
    id2label=ids_to_labels,
    label2id=labels_to_ids,
    ignore_mismatched_sizes=True  # set to True to use custom labels
)
#training setup
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
args = TrainingArguments(
    "models/FMEA-ner",
    evaluation_strategy="steps",
    save_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=2,
    weight_decay=0.01,
    push_to_hub=False,
    per_device_train_batch_size = 4,
    per_device_eval_batch_size = 4,
    logging_steps=50,
    eval_steps = 50,
    #label_names = label_list
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_data,
    eval_dataset=test_data,
    data_collator=data_collator,
    compute_metrics= lambda x: compute_metrics(x, id2label=ids_to_labels),
    tokenizer=tokenizer,
)


train_result = trainer.train()
final_train_metrics = train_result.metrics
metrics=trainer.evaluate()
final_eval_metrics = metrics
preds, label_ids, pred_metric = trainer.predict(test_data)
labels = test_data['labels']
y_pred = np.argmax(preds, axis=1)

print(y_pred)
print(compute_classification_report(labels, preds, label_ids, ids_to_labels))
build_confusion_matrix(labels, preds, label_ids, ids_to_labels)
num_train = len(train_dataset); num_epochs = args.num_train_epochs; num_batch = args.per_device_train_batch_size
num_steps = int((num_train / num_batch) * num_epochs)
filename = os.path.join(os.getcwd(),"models", "FMEA-ner", "checkpoint-"+str(num_steps), "trainer_state.json")
plot_eval_results(filename)
trainer.save_model()