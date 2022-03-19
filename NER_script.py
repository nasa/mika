# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 11:16:56 2022

@author: srandrad
"""
import pandas as pd
import os
import json
from spacy.training import offsets_to_biluo_tags
import spacy
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification
from transformers import TrainingArguments, Trainer
from seqeval.metrics import classification_report
from torch import cuda
from datasets import load_metric, Dataset, DatasetDict

device = 'cuda' if cuda.is_available() else 'cpu'
print(device)
nlp = spacy.load("en_core_web_trf")
nlp.add_pipe("sentencizer")

file = os.path.join('data','UAS_fire_SAFECOM_annotated_SA.jsonl')

def read_doccano_annots(file):
    f = open(file, "r")
    list_of_str_jsons = f.read().split("\n")[:-2]#removing last item which is empty
    list_of_dict_jsons = [json.loads(data) for data in list_of_str_jsons]
    df = pd.DataFrame(list_of_dict_jsons)
    return df

def clean_doccano_annots(df):
    for i in range(len(df)):
        text = df.iloc[i]['data']
        label = df.iloc[i]['label']
        new_labels, new_text = clean_text_tags(text, label)
        df.at[i, 'label'] = new_labels
        df.at[i, 'data'] = new_text
    return df

def clean_annots_from_str(df):
    cleaned_labels = []
    for label in df['label'].tolist():
        text_lists = []
        for t in label.split("],"):
            ts = t.strip("['']").split(",")
            temp = [te.strip(" []'") for te in ts]
            temp[0] = int(temp[0])
            temp[1] = int(temp[1])
            text_lists.append(temp)
        cleaned_labels.append(text_lists)
    df['label'] = cleaned_labels
    return df

def clean_text_tags(text, labels): #input single text, list of labels [beg, end, tag]
    new_labels = []
    spaces_added = 0
    add_spaces = False
    labels.sort()
    for label in labels:
        new_label = label
        new_text = text
        if add_spaces == True:
            new_label = [new_label[0]+spaces_added, new_label[1]+spaces_added, new_label[2]]
        #case 1: included extra " " or punctuation at begining or end of token
        #check 1st, last text for punctuation and spaces
        prev_text = text[new_label[0]:new_label[1]]
        prev_len = len(prev_text)
        label_text = prev_text.strip(" .,;'}")
        new_len = len(label_text)
        if new_len != prev_len:
            if prev_text[0] != label_text[0]: #i.e. there was a leading space
                new_label = [new_label[0]+1, new_label[1], new_label[2]]
            if prev_text[-1] != label_text[-1]: # i.e. there was a trailing space
                new_label = [new_label[0], new_label[1]-1, new_label[2]]
        #case 2: did not include begining or ending characters of token
        #check previous char for space, check next char for space/punctuation
        else:
            if new_label[0] > 0 and text[new_label[0]-1]!=" ": #did not include first char
                precedding_char = text[new_label[0]-1]
                if precedding_char.isalpha() == True: 
                    new_label = [new_label[0]-1, new_label[1], new_label[2]]
            if new_label[1]<len(text)-1 and text[new_label[1]]!=" ": #did not include trailing chars
                procedding_char = text[new_label[1]]
                if procedding_char.isalpha() == True: 
                    new_end = text.find(" ", new_label[1])
                    new_label = [new_label[0], new_end, new_label[2]]
            #case 3: missing space -> need to update text and following labels by adding in a space
            #this occurs when the precedding or following character is punctuation
            if new_label[0] > 0 and (not text[new_label[0]-1].isalpha()) and text[new_label[0]-1]!=" ": #missing preceding space
                new_text = text[:new_label[0]] + ' ' + text[new_label[0]:]
                spaces_added += 1
                #spaces need to be added to future labels and this label
                new_label = [new_label[0]+1, new_label[1]+1, new_label[2]]
                add_spaces=True
            if new_label[1]<len(new_text)-2 and not new_text[new_label[1]:new_label[1]+2].isalpha(): #missing proceeding space
                new_text = new_text[:new_label[1]] + ' ' + new_text[new_label[1]:]
                spaces_added += 1
                #spaces need to be added to future labels
                add_spaces=True
        #update text for added spaces
        text = new_text
        new_labels.append(new_label)
    return new_labels, text

def identify_bad_annotations(text_df):
    inds_with_issues = [i for i in range(len(text_df)) if '-' in text_df.iloc[i]['tags']]
    text_df_issues = text_df.iloc[inds_with_issues][:]
    bad_tokens = []
    for ind in range(len(text_df_issues)):
        inds = [i for i, x in enumerate(text_df_issues.iloc[ind]['tags']) if x == "-"]
        [bad_tokens.append(text_df_issues.iloc[ind]['docs'][i]) for i in inds]
    return bad_tokens

def split_docs_to_sentances(text_df): 
    #split each document into one row per sentance
    sentence_tags_total = []
    sentences_in_list = []
    ids = []
    for i in range(len(text_df)):
        doc = text_df.iloc[i]['docs']
        total_sentence_tags = text_df.iloc[i]['tags']
        sentences = [sent for sent in doc.sents] #split into sentences
        sentence_tags = [[tag for tag in  total_sentence_tags[sent.start:sent.end]] for sent in doc.sents]
        for tags in sentence_tags:
            sentence_tags_total.append(tags)
        for sent in sentences:
            sentences_in_list.append(sent)
            ids.append(text_df.iloc[i]['Tracking #'])
    sentence_df = pd.DataFrame({"Tracking #":ids,
                                "sentence": sentences_in_list,
                                "tags": sentence_tags_total})
    return sentence_df

def check_doc_to_sentence_split(sentence_df):
    for i in range(len(sentence_df)):
        sent = sentence_df.iloc[i]['sentence']
        num_tokens = len([token.text for token in sent])
        num_tags = len(sentence_df.iloc[i]['tags'])
        if num_tokens != num_tags: print("error: the number of tokens does not equal the number of tags")

#read in doccano annotations
df = read_doccano_annots(file)
text_df = df[['data', 'label', 'Tracking #']]
#clean doccano annotations
text_df = clean_doccano_annots(text_df)
#convert text to spacy doc 
text_data = text_df['data'].tolist()
docs = [nlp(doc) for doc in text_data]
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
sentence_df = split_docs_to_sentances(text_df)
check_doc_to_sentence_split(sentence_df)

#convert labels to numerical tags
ner_tags = [[labels_to_ids[label] for label in labels] for labels in sentence_df['tags']]
sentence_df['ner_tags'] = ner_tags
#load tokenizer
cuda.empty_cache()
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

def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # Same word as previous token
            label = labels[word_id]
            # If the label is B-XXX we change it to I-XXX
            if label % 2 == 1:
                label += 1
            new_labels.append(label)
    return new_labels

def tokenize_and_align_labels(sentence_df):
    tokenized_inputs = tokenizer(
        sentence_df["tokens"], is_split_into_words=True
    )
    all_labels = sentence_df["ner_tags"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs

label_list = total_tags['tags'].unique()
metric = load_metric("seqeval")

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[label_list[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
    
    return {
        "precision": all_metrics["overall_precision"],
        "recall": all_metrics["overall_recall"],
        "f1": all_metrics["overall_f1"],
        "accuracy": all_metrics["overall_accuracy"],
    }

cols_to_drop = [col for col in train_dataset.columns if col not in ["tokens", 'ner_tags']]
train_dataset = train_dataset.drop(cols_to_drop, axis=1)
test_dataset = test_dataset.drop(cols_to_drop, axis=1)
train_data = Dataset.from_pandas(train_dataset).map(
    tokenize_and_align_labels,
    batched=True)
test_data = Dataset.from_pandas(test_dataset).map(
    tokenize_and_align_labels,
    batched=True)

#initiating model
model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    id2label=ids_to_labels,
    label2id=labels_to_ids,
    ignore_mismatched_sizes=True  # set to True to use custom labels
)
#training setup
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
args = TrainingArguments(
    "FMEA-ner",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=2,
    weight_decay=0.01,
    push_to_hub=False,
    per_device_train_batch_size = 4,
    per_device_eval_batch_size = 4
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_data,
    eval_dataset=test_data,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
)

train_result = trainer.train()
print(train_result)
metrics=trainer.evaluate()
print(metrics)