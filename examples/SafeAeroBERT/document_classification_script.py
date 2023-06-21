# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 13:11:29 2022

@author: srandrad
"""
from transformers import TrainingArguments, Trainer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import datasets
from datasets import Dataset, concatenate_datasets, DatasetDict
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support, accuracy_score
import pandas as pd
import numpy as np
import sys, os
from torch import cuda
import pathlib
sys.path.append(os.path.join(".."))
from mika.utils import Data

os.environ["TOKENIZERS_PARALLELISM"] = "false"
#os.environ["CUDA_VISIBLE_DEVICES"] = "2"
#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

contributing_factors = ['Human Factors', 
                        'Weather', 
                        'Procedure', 
                        'Aircraft' 
                        ]
checkpoint = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir)),"models", "SafeAeroBERT_v2", "checkpoint-13000")

model_checkpoints = ["allenai/scibert_scivocab_uncased", 
                     "bert-base-uncased", 
                     checkpoint] 

def get_most_recent_checkpoint(save_name, contributing_factor):
    rootdir = os.path.join(os.getcwd(), f"{contributing_factor}-{save_name}-finetuned")
    if os.path.isdir(rootdir) == False:
        return None
    checkpoints = []
    for subdir, dirs, files in os.walk(rootdir):
        if 'checkpoint' in subdir and 'finetuned' not in subdir: 
            checkpoints.append(int(subdir.split("-")[-1]))
    if checkpoints == []:
        return None
    most_recent_checkpoint = max(checkpoints)
    checkpoint = os.path.join(os.getcwd(), f"{contributing_factor}-{save_name}-finetuned", "checkpoint-"+str(most_recent_checkpoint))
    return checkpoint
    
def train_classifier(tokenizer, model, encoded_dataset, contributing_factor, compute_metrics, model_name, batch_size=4):
    save_name = model_name.split("/")[-1]
    if "checkpoint" in model_name:
        path = pathlib.PurePath(model_name)
        save_name = path.name
    args = TrainingArguments(
    f"{contributing_factor}-{save_name}-finetuned",
    evaluation_strategy = "epoch",
    save_strategy = "steps",
    learning_rate=1e-3,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=2,
    num_train_epochs=2,
    weight_decay=0.01,
    push_to_hub=False,
    gradient_accumulation_steps=8,
    save_steps= 10,
    gradient_checkpointing=True,
    optim="adafactor",
    save_total_limit = 2 #saves only last 2 checkpoints
    )

    trainer = Trainer(
    model,
    args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["valid"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
    )
    checkpoint = get_most_recent_checkpoint(save_name, contributing_factor)
    if checkpoint is not None:
        history = trainer.train(checkpoint)
    else:
        history = trainer.train()
    return trainer

def evaluate_test_set(trainer, encoded_dataset, data_type, average='weighted'):
    preds, label_ids, pred_metric = trainer.predict(encoded_dataset[data_type])
    labels = encoded_dataset[data_type]['label']
    y_pred = np.argmax(preds, axis=1)
    precision, recall, fscore, support = precision_recall_fscore_support(labels, y_pred, average=average)
    accuracy = accuracy_score(labels, y_pred)
    return precision, recall, fscore, accuracy

def train_test_model(ASRS_df, contributing_factors, models, train_size, test_size, val_size, compute_metrics, save_results=False, batch_size=2):
    test_results = {model: [] for model in models}
    train_results = {model: [] for model in models}
    val_results = {model: [] for model in models}
    X, y = get_X_y(ASRS_df, contributing_factors)
    X_train, y_train, X_val, y_val, X_test, y_test = split_data_for_all_categories(contributing_factors, X, y, train_size, test_size, val_size)
    save_data_counts(contributing_factors, X_train, y_train, X_val, y_val, X_test, y_test)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    for model_checkpoint in models:
        classification_model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2)
        for contributing_factor in contributing_factors:
            print(model_checkpoint, "-----", contributing_factor)
            cuda.empty_cache()
            encoded_dataset = prepare_data(X_train, y_train, X_val, y_val, X_test, y_test, contributing_factor, tokenizer)
            classification_model.to('cuda')
            trainer = train_classifier(tokenizer, classification_model, encoded_dataset, contributing_factor, compute_metrics, model_checkpoint, batch_size=batch_size)
            precision, recall, fscore, accuracy = evaluate_test_set(trainer, encoded_dataset, data_type='test', average='weighted')
            test_results[model_checkpoint].append(accuracy)
            test_results[model_checkpoint].append(precision)
            test_results[model_checkpoint].append(recall)
            test_results[model_checkpoint].append(fscore)
            precision, recall, fscore, accuracy = evaluate_test_set(trainer, encoded_dataset, data_type='train', average='weighted')
            train_results[model_checkpoint].append(accuracy)
            train_results[model_checkpoint].append(precision)
            train_results[model_checkpoint].append(recall)
            train_results[model_checkpoint].append(fscore)
            precision, recall, fscore, accuracy = evaluate_test_set(trainer, encoded_dataset, data_type='valid', average='weighted')
            val_results[model_checkpoint].append(accuracy)
            val_results[model_checkpoint].append(precision)
            val_results[model_checkpoint].append(recall)
            val_results[model_checkpoint].append(fscore)
    
    iterables = [contributing_factors, ["Accuracy", "Precision", "Recall", "F1"]]
    ind = pd.MultiIndex.from_product(iterables)
    test_results_df = pd.DataFrame(test_results, index=ind)
    train_results_df = pd.DataFrame(train_results, index=ind)
    val_results_df = pd.DataFrame(val_results, index=ind)
    dfs = [test_results_df, train_results_df, val_results_df]
    combined_results = pd.concat(dfs ,keys= ['Test', 'Train', 'Validation'], axis=1)
    if save_results == True:
        test_results_df.to_csv("document_classification_test_results.csv")
        train_results_df.to_csv("document_classification_train_results.csv")
        val_results_df.to_csv("document_classification_validation_results.csv")
        combined_results.to_csv("document_classification_combined_results.csv")
    return test_results_df, train_results_df, val_results_df, combined_results

def tokenize(text_df, tokenizer):
    tokenized_inputs = tokenizer(text_df["Combined Text"], truncation=True)
    return tokenized_inputs

def get_X_y(ASRS_df, contributing_factors):
    cat_indicators = {cat:[] for cat in contributing_factors}
    for i in range(len(ASRS_df)):
        for cat in cat_indicators:
            if cat in str(ASRS_df.at[i,'Assessments']):
                cat_indicators[cat].append(1)
            else:
                cat_indicators[cat].append(0)
    X = ASRS_df["Combined Text"]
    y = pd.DataFrame(cat_indicators)
    return X, y

def split_data(X, y, category, train_size=0.6, test_size=0.2, val_size=0.2, random_state=0):
    #need to add in handling for when there are few reports
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_size, train_size=train_size+val_size, random_state=random_state, stratify=y[category])
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_size/(train_size+val_size), random_state=random_state, stratify=y_train_val[category])
    return X_train, X_test, X_val, y_train, y_test, y_val

def split_data_for_all_categories(contributing_factors, X, y, train_size, test_size, val_size):
    X_train = {cat:[] for cat in contributing_factors}
    X_test = {cat:[] for cat in contributing_factors}
    X_val = {cat:[] for cat in contributing_factors}
    y_train = {cat:[] for cat in contributing_factors}
    y_test = {cat:[] for cat in contributing_factors}
    y_val = {cat:[] for cat in contributing_factors}
    for cat in contributing_factors: 
        X_train_temp, X_test_temp, X_val_temp, y_train_temp, y_test_temp, y_val_temp = split_data(X, y, cat, train_size, test_size, val_size)
        X_train[cat] = X_train_temp
        X_test[cat] = X_test_temp
        X_val[cat] = X_val_temp
        y_train[cat] = y_train_temp
        y_test[cat] = y_test_temp
        y_val[cat] = y_val_temp
    return X_train, y_train, X_val, y_val, X_test, y_test

def save_data_counts(contributing_factors, X_train, y_train, X_val, y_val, X_test, y_test):
    iterables = [["Train", "Validation", "Test"], ["0", "1", "total"]]

    ind = pd.MultiIndex.from_product(iterables)
    
    counts = pd.DataFrame([[len(y_train[cat].loc[y_train[cat][cat]==0]) for cat in contributing_factors],
                  [len(y_train[cat].loc[y_train[cat][cat]==1]) for cat in contributing_factors],
                  [len(X_train[cat]) for cat in contributing_factors],
                  [len(y_val[cat].loc[y_val[cat][cat]==0]) for cat in contributing_factors],
                  [len(y_val[cat].loc[y_val[cat][cat]==1]) for cat in contributing_factors],
                  [len(X_val[cat]) for cat in contributing_factors],
                  [len(y_test[cat].loc[y_test[cat][cat]==0]) for cat in contributing_factors],
                  [len(y_test[cat].loc[y_test[cat][cat]==1]) for cat in contributing_factors],
                  [len(X_test[cat]) for cat in contributing_factors]],
                  index = ind, columns = contributing_factors)
    counts.to_csv("train_test_valid_counts.csv")
    
def prepare_data(X_train, y_train, X_val, y_val, X_test, y_test, contributing_factor, tokenizer):
    x_train = X_train[contributing_factor]
    y_train = y_train[contributing_factor][contributing_factor]
    x_valid = X_val[contributing_factor]
    y_valid = y_val[contributing_factor][contributing_factor]
    x_test = X_test[contributing_factor]
    y_test = y_test[contributing_factor][contributing_factor]
    train = Dataset.from_pandas(x_train.to_frame()).map(lambda example: {'idx': example['__index_level_0__']}, remove_columns=['__index_level_0__'])
    valid = Dataset.from_pandas(x_valid.to_frame()).map(lambda example: {'idx': example['__index_level_0__']}, remove_columns=['__index_level_0__'])
    test = Dataset.from_pandas(x_test.to_frame()).map(lambda example: {'idx': example['__index_level_0__']}, remove_columns=['__index_level_0__'])
    train_labels = Dataset.from_pandas(y_train.to_frame()).map(lambda example: {'label': example[contributing_factor]}, remove_columns=['__index_level_0__', contributing_factor])
    valid_labels = Dataset.from_pandas(y_valid.to_frame()).map(lambda example: {'label': example[contributing_factor]}, remove_columns=['__index_level_0__', contributing_factor])
    test_labels = Dataset.from_pandas(y_test.to_frame()).map(lambda example: {'label': example[contributing_factor]}, remove_columns=['__index_level_0__', contributing_factor])
    test_data = concatenate_datasets([test, test_labels], axis=1)
    train_data = concatenate_datasets([train, train_labels], axis=1)
    valid_data = concatenate_datasets([valid, valid_labels], axis=1)
    dataset = DatasetDict({'train': train_data,
                      'valid': valid_data,
                      'test': test_data})

    encoded_dataset = dataset.map(tokenize, fn_kwargs={'tokenizer':tokenizer}, batched=True)
    return encoded_dataset

def compute_metrics(eval_predictions):
    predictions, labels = eval_predictions
    predictions = np.argmax(predictions, axis=1)
    precision, recall, fscore, support = precision_recall_fscore_support(labels, predictions, average='weighted')
    accuracy = accuracy_score(labels, predictions)
    return {"precision": precision,
            "recall": recall,
            "f1": fscore, 
            "accuracy": accuracy}
if __name__ == '__main__':
    #load in data
    ASRS_file = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir)),'data/ASRS/ASRS_1988_2022_cleaned.csv')
    ASRS_id_col = 'ACN'
    ASRS_text_cols = ['Report 1', 'Report 1.1', 'Report 2',	'Report 2.1', 'Report 1.2']
    ASRS = Data()
    ASRS.load(ASRS_file, id_col=ASRS_id_col, text_columns=ASRS_text_cols)
    ASRS.prepare_data(combine_columns=ASRS_text_cols, remove_incomplete_rows=False)
    ASRS_df = ASRS.data_df.loc[ASRS.data_df["Combined Text"]!=""].reset_index(drop=True)
    
    test_results_df, train_results_df, val_results_df, combined_results = train_test_model(ASRS_df, contributing_factors, model_checkpoints, train_size=8000, test_size=1000, val_size=1000, compute_metrics=compute_metrics, save_results=True, batch_size=4)