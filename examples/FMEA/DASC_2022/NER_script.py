# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 13:28:23 2022

requires CUDA GPU set up

@author: srandrad
"""

import pandas as pd
import os
from spacy.training import offsets_to_biluo_tags
import spacy
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification, BertForTokenClassification
from transformers import TrainingArguments, Trainer
from torch import cuda, FloatTensor
from torch.nn import CrossEntropyLoss
from datasets import Dataset

from mika.kd.NER import read_doccano_annots, clean_doccano_annots, split_docs_to_sentances, check_doc_to_sentence_split, tokenize_and_align_labels, compute_metrics, compute_classification_report, build_confusion_matrix, plot_eval_results

#set up GPU
device = 'cuda' if cuda.is_available() else 'cpu'
cuda.empty_cache()
print(device)

#load spacy sentencizer
nlp = spacy.load("en_core_web_trf")
nlp.add_pipe("sentencizer")

file = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir, os.pardir)), 'data','doccano','annotations','srandrad_safecom_v2.jsonl')
LLIS_folder = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir, os.pardir)), 'data','doccano','annotations')
#read in doccano annotations
df = read_doccano_annots(file)
text_df = df[['data', 'label', 'Tracking #']] #"Lesson ID"]]
llis_files = ['srandrade_DE.jsonl', 'srandrade_REC.jsonl', 'srandrade_LL.jsonl',
              'LLIS_DE_150_160_HW.jsonl', 'LLIS_REC_150_160_HW.jsonl', 'LLIS_LL_150_160_HW.jsonl']
llis_dfs = []
for llis_file in llis_files:
    if "HW" in llis_file: encoding=True
    else: encoding = False
    llis_file = os.path.join(LLIS_folder, llis_file)
    llis_dfs.append(read_doccano_annots(llis_file, encoding=encoding))
llis_df = pd.concat(llis_dfs).reset_index(drop=True)
llis_df = llis_df[['data', 'label', 'Lesson ID']]
#clean doccano annotations
text_df = clean_doccano_annots(text_df)
llis_df = clean_doccano_annots(llis_df)

#convert text to spacy doc 
text_data = text_df['data'].tolist()
docs = [nlp(doc) for doc in text_data]
#print(text_df.columns)
text_df['docs'] = docs#[nlp(doc) for doc in text_data]

llis_text_data = llis_df['data'].tolist()
docs = [nlp(doc) for doc in llis_text_data]
#print(text_df.columns)
llis_df['docs'] = docs#[nlp(doc) for doc in text_data]


#convert offset labels to biluo tags
text_df['tags'] = [offsets_to_biluo_tags(text_df.at[i,'docs'], text_df.at[i,'label']) for i in range(len(text_df))]
for i in range(len(llis_df)):
    #print(llis_df.iloc[i]['Lesson ID'], llis_df.at[i,'docs'])
    tags = offsets_to_biluo_tags(llis_df.at[i,'docs'], llis_df.at[i,'label'])
    if '-' in tags:
        print(llis_df.iloc[i]['Lesson ID'], llis_df.at[i,'docs'])
        print(tags)
llis_df['tags'] = [offsets_to_biluo_tags(llis_df.at[i,'docs'], llis_df.at[i,'label']) for i in range(len(llis_df))]

#check for tags with issues
inds_with_issues = [i for i in range(len(text_df)) if '-' in text_df.iloc[i]['tags']]
text_df_issues = text_df.iloc[inds_with_issues][:].reset_index(drop=True)
if len(text_df_issues)>0: print("error: there are documents with invalid tags")

inds_with_issues = [i for i in range(len(llis_df)) if '-' in llis_df.iloc[i]['tags']]
llis_df_issues = llis_df.iloc[inds_with_issues][:].reset_index(drop=True)
if len(llis_df_issues)>0: print("error: there are documents with invalid tags")

#set up labels
total_tags= pd.DataFrame({'tags':[t for tag in llis_df['tags'] for t in tag]})
tags_simple = pd.DataFrame({'tags':[t.split("-")[1] if "-" in t else t for tag in llis_df['tags'] for t in tag]})
labels_to_ids = {k: v for v, k in enumerate(total_tags['tags'].unique())}
ids_to_labels = {v: k for v, k in enumerate(total_tags['tags'].unique())}

#look at frequencies
print("Number of tags: {}".format(len(total_tags['tags'].unique())))
frequencies = total_tags['tags'].value_counts()
print(frequencies)
print("Number of tags: {}".format(len(tags_simple['tags'].unique())))
frequencies = tags_simple['tags'].value_counts()
print(frequencies)

#prepare dataset and dataloader
#convert df from docs to sentences
sentence_df = split_docs_to_sentances(text_df, id_col='Tracking #', tags=True)
check_doc_to_sentence_split(sentence_df)

inds_to_drop = [i for i in range(len(sentence_df)) if set(sentence_df.iloc[i]['tags'])=={'O'}]
sentence_df = sentence_df.drop(inds_to_drop).reset_index(drop=True)
llis_sentence_df = split_docs_to_sentances(llis_df, id_col='Lesson ID',  tags=True)
check_doc_to_sentence_split(llis_sentence_df)
inds_to_drop = [i for i in range(len(llis_sentence_df)) if set(llis_sentence_df.iloc[i]['tags'])=={'O'}]
llis_sentence_df = llis_sentence_df.drop(inds_to_drop).reset_index(drop=True)
#convert labels to numerical tags
ner_tags = [[labels_to_ids[label] for label in labels] for labels in sentence_df['tags']]
sentence_df['ner_tags'] = ner_tags
ner_tags = [[labels_to_ids[label] for label in labels] for labels in llis_sentence_df['tags']]
llis_sentence_df['ner_tags'] = ner_tags

#load tokenizer
sentence_df['tokens'] = [[tok.orth_ for tok in sentence_df.iloc[i]['sentence']] for i in range(len(sentence_df))]
llis_sentence_df['tokens'] = [[tok.orth_ for tok in llis_sentence_df.iloc[i]['sentence']] for i in range(len(llis_sentence_df))]

#model_checkpoint = "bert-base-uncased"
model_checkpoint = os.path.join("models","Pre-trained-BERT","checkpoint-318766")
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

train_size = 0.8

#train, test both llis
train_dataset = llis_sentence_df.sample(frac=train_size,random_state=200)
test_dataset = llis_sentence_df.drop(train_dataset.index).reset_index(drop=True)
train_dataset = train_dataset.reset_index(drop=True)

#train_dataset = llis_sentence_df
#test_dataset = sentence_df
print("FULL Dataset: {}".format(llis_sentence_df.shape))
print("TRAIN Dataset: {}".format(train_dataset.shape))
print("TEST Dataset: {}".format(test_dataset.shape))

label_list = total_tags['tags'].unique()

#print(len(label_list))

cols_to_drop = [col for col in train_dataset.columns if col not in ["tokens", 'ner_tags', 'Tracking #', 'Lesson ID']]
train_dataset = train_dataset.drop(cols_to_drop, axis=1)
test_dataset = test_dataset.drop(cols_to_drop, axis=1)
sentence_df = sentence_df.drop(cols_to_drop, axis=1)
train_data = Dataset.from_pandas(train_dataset).map(
    tokenize_and_align_labels,
    batched=True,
    fn_kwargs={'tokenizer':tokenizer})
test_data = Dataset.from_pandas(test_dataset).map(
    tokenize_and_align_labels,
    batched=True,
    fn_kwargs={'tokenizer':tokenizer})
safecom_data = Dataset.from_pandas(sentence_df).map(
    tokenize_and_align_labels,
    batched=True,
    fn_kwargs={'tokenizer':tokenizer})

#test_data.save_to_disk("data/NER_test_dataset")
#train_data.save_to_disk("data/NER_train_dataset")

#initiating model
model =BertForTokenClassification.from_pretrained( #AutoModelForTokenClassification
    model_checkpoint,
    id2label=ids_to_labels,
    label2id=labels_to_ids,
    ignore_mismatched_sizes=True,  # set to True to use custom labels
    force_download=True
)
#training setup
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
args = TrainingArguments(
    os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir, os.pardir)), "models/FMEA-ner-model"),
    evaluation_strategy="steps",
    save_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=3,
    weight_decay=0.01,
    push_to_hub=False,
    per_device_train_batch_size = 4,
    per_device_eval_batch_size = 4,
    logging_steps=50,
    eval_steps = 50
)
"""
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_data,
    eval_dataset=test_data,
    data_collator=data_collator,
    compute_metrics= lambda x: compute_metrics(x, id2label=ids_to_labels),
    tokenizer=tokenizer,
)
"""
# custom loss
classDistribution_raw = list(np.unique([l for lis in train_dataset['ner_tags'].tolist() for l in lis], return_counts=True)[1])#[97, 3]
normedWeights = [1 - (x / sum(classDistribution_raw)) for x in classDistribution_raw]
normedWeights = FloatTensor(normedWeights).cuda()
num_labels = len(normedWeights)
class MyTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        
        if "labels" in inputs:
          labels = inputs.pop("labels")
        
        outputs = model(**inputs)
        logits = outputs.logits
        loss_function = CrossEntropyLoss(weight = normedWeights)

        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = loss_function(logits.view(-1, num_labels), labels.view(-1))
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

trainer = MyTrainer(
    model=model,
    args=args,
    train_dataset=train_data,
    eval_dataset=test_data,
    data_collator=data_collator,
    compute_metrics= lambda x: compute_metrics(x, id2label=ids_to_labels),
    tokenizer=tokenizer,
)

#"""
train_result = trainer.train()
final_train_metrics = train_result.metrics
metrics=trainer.evaluate()
final_eval_metrics = metrics
preds, label_ids, pred_metric = trainer.predict(test_data)
labels = test_data['labels']
y_pred = np.argmax(preds, axis=1)
print(compute_classification_report(labels, preds, label_ids, ids_to_labels))
build_confusion_matrix(labels, preds, label_ids, ids_to_labels)
#"""
#"""
#for safecom:
preds, label_ids, pred_metric = trainer.predict(safecom_data)
labels = safecom_data['labels']
y_pred = np.argmax(preds, axis=1)
print(compute_classification_report(labels, preds, label_ids, ids_to_labels))
build_confusion_matrix(labels, preds, label_ids, ids_to_labels, save=True, savepath="examples/FMEA/DASC_2022/safecom_")
#"""
#"""
num_steps = trainer.state.max_steps
filename = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir, os.pardir)),"models", "FMEA-ner-model", "checkpoint-"+str(num_steps), "trainer_state.json")
plot_eval_results(filename, save=True, savepath='examples/FMEA/DASC_2022/')
trainer.save_model()
#"""