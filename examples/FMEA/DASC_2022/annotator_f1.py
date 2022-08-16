# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 13:16:13 2022

@author: srandrad
"""

from module.NER_utils import read_doccano_annots, clean_doccano_annots, split_docs_to_sentances, check_doc_to_sentence_split, tokenize_and_align_labels, compute_metrics, compute_classification_report, build_confusion_matrix, plot_eval_results

import os
import pandas as pd
from torch import cuda
from spacy.training import offsets_to_biluo_tags
import spacy
from datasets import load_metric
import numpy as np
from sklearn.metrics import f1_score
#from seqeval.scheme import BILOU
import sklearn
from sklearn.metrics import cohen_kappa_score

def update_label(mismatch_ind, label_list):
    new_labels = []
    for label in label_list:
        new_label = label
        if label[0]>mismatch_ind:
            new_label[0] = label[0]-1
            new_label[1] = label[1]-1
            new_label[2] = label[2]
        new_labels.append(new_label)
    return new_labels

def corrext_mismatch_len(text1, text2, labels1, labels2):
    while len(text1) != len(text2):
        #need to find mis matching character
        mismatch_ind = min([i for i in range(min(len(text1),len(text2))) if text1[i]!=text2[i]])
        #check to see which text has an extra space
        if text1[mismatch_ind] == " ":
            text1 = text1[0:mismatch_ind]+text1[mismatch_ind+1:]
            labels1 = update_label(mismatch_ind, labels1)
        elif text2[mismatch_ind] == " ":
            text2 = text2[0:mismatch_ind]+text2[mismatch_ind+1:]
            labels2 = update_label(mismatch_ind, labels2)
        else:
            print(text1[:mismatch_ind], text1)
            print(text2[:mismatch_ind], text2)
    return text1, text2, labels1, labels2
        
def clean_interannotations(annot1, annot2):
    if len(annot1) != len(annot2):
        print("Error: datasets do not have the same number of annotations")
        print("Annot 1: ", len(annot1), "Annot 2: ", len(annot2))
        return
    for i in range(len(annot1)):
        text1 = annot1.iloc[i]['data']
        text2 = annot2.iloc[i]['data']
        labels1 = annot1.iloc[i]['label']
        labels2 = annot2.iloc[i]['label']
        if len(text1) != len(text1):
            text1, text2, labels1, labels2 = corrext_mismatch_len(text1, text2, labels1, labels2)
        
        mismatch_inds = [i for i in range(min(len(text1),len(text2))) if text1[i]!=text2[i]]
        while mismatch_inds != []:
            #if there are still mismatching indicies with extra spaces...
            mismatch_ind = min(mismatch_inds)
            if text1[mismatch_ind] == " ":
                text1 = text1[0:mismatch_ind]+text1[mismatch_ind+1:]
                labels1 = update_label(mismatch_ind, labels1)
            elif text2[mismatch_ind] == " ":
                text2 = text2[0:mismatch_ind]+text2[mismatch_ind+1:]
                labels2 = update_label(mismatch_ind, labels2)
            else:
                print(text1[:mismatch_ind], text1)
                print(text2[:mismatch_ind], text2)
            text1, text2, labels1, labels2 = corrext_mismatch_len(text1, text2, labels1, labels2)
            mismatch_inds = [i for i in range(min(len(text1),len(text2))) if text1[i]!=text2[i]]
        annot1.at[i,'data'] = text1
        annot2.at[i, 'data'] = text2
        annot1.at[i,'label'] = labels1
        annot2.at[i,'label'] = labels2
    return annot1, annot2

def prepare_annot_dfs(file1, file2, name="safecom", file1_encode=True, file2_encode=False):
    if name == 'safecom': id_ = 'Tracking #'
    elif name == 'llis': id_= 'Lesson ID'
    
    #read in doccano annotations
    H_annot_df = read_doccano_annots(file1,encoding=file1_encode)
    ids = H_annot_df[id_].tolist()
    H_annot_df = H_annot_df.sort_values(by=[id_]).reset_index()
    H_annot_df = H_annot_df[['data', 'label']]
    
    S_annot_df =  read_doccano_annots(file2,encoding=file2_encode)
    S_annot_df = S_annot_df.loc[S_annot_df[id_].isin(ids)].reset_index(drop=True)
    S_annot_df = S_annot_df.sort_values(by=[id_]).reset_index()
    S_annot_df = S_annot_df[['data', 'label']]
    
    H_annot_df = clean_doccano_annots(H_annot_df)
    S_annot_df = clean_doccano_annots(S_annot_df)
    
    #problems between OS: extra spaces, ” and “, ’,
    S_annot_df, H_annot_df = clean_interannotations(S_annot_df, H_annot_df)
    
    bad_inds = []
    #check for bad indexes
    for i in range(len(S_annot_df)):
        text = S_annot_df.iloc[i]['data']
        ind = H_annot_df.loc[H_annot_df['data']==text]
        if len(text) != len(H_annot_df.loc[i]['data']):
        #if ind.index.values.size == 0:
            bad_inds.append(i)
                
    print("Number of bad reports: ", len(bad_inds))
    
    #need to remove and update tags
    H_annot_df = H_annot_df.drop(bad_inds).reset_index(drop=True)
    S_annot_df = S_annot_df.drop(bad_inds).reset_index(drop=True)
    
    #convert text to spacy doc 
    H_annot_data = H_annot_df['data'].tolist()
    docs = [nlp.make_doc(doc) for doc in H_annot_data]
    H_annot_df['docs'] = docs#[nlp(doc) for doc in text_data]
    
    S_annot_data = S_annot_df['data'].tolist()
    docs = [nlp.make_doc(doc) for doc in S_annot_data]
    S_annot_df['docs'] = docs
    
    #convert offset labels to biluo tags
    H_annot_df['tags'] = [offsets_to_biluo_tags(H_annot_df.at[i,'docs'], H_annot_df.at[i,'label']) for i in range(len(H_annot_df))]
    S_annot_df['tags'] = [offsets_to_biluo_tags(S_annot_df.at[i,'docs'], S_annot_df.at[i,'label']) for i in range(len(S_annot_df))]
    
    return H_annot_df, S_annot_df

#set up GPU
device = 'cuda' if cuda.is_available() else 'cpu'
cuda.empty_cache()

#load spacy sentencizer
nlp = spacy.load("en_core_web_trf")
nlp.add_pipe("sentencizer")

llis_H_files = ['data/doccano/annotations/hswalsh_LLIS_DE_IAA.jsonl','data/doccano/annotations/hswalsh_LLIS_LL_IAA.jsonl', 'data/doccano/annotations/hswalsh_LLIS_R_IAA.jsonl']
llis_S_files = ['data/doccano/annotations/srandrade_DE.jsonl','data/doccano/annotations/srandrade_LL_IAA.jsonl', 'data/doccano/annotations/srandrade_REC.jsonl']
h = []
s = []
for i in range(3):
    file1 = llis_H_files[i]
    file2 = llis_S_files[i]
    if i == 1: file2_encode=True
    else: file2_encode=False
    H_df, S_df = prepare_annot_dfs(file1, file2, name="llis", file2_encode=file2_encode)
    h.append(H_df)
    s.append(S_df)
S_annot_df = pd.concat(s).reset_index(drop=True)
H_annot_df = pd.concat(h).reset_index(drop=True)

H_safecom, S_safecom = prepare_annot_dfs(file1=os.path.join('data','doccano','annotations','hswalsh_safecom_round2.jsonl'), file2=os.path.join('data','doccano','annotations','srandrad_safecom_v2.jsonl'))

metric = load_metric("seqeval")
true_predictions = S_annot_df['tags']
true_labels = H_annot_df['tags']
all_metrics = metric.compute(predictions=true_predictions, references=true_labels, zero_division=0)
ents = [ent for ent in all_metrics if 'overall' not in ent]
f1 = [all_metrics[ent]['f1'] for ent in ents]
ents.append('Average')
f1.append(all_metrics['overall_f1'])
f1_df = pd.DataFrame({"Entity":ents, 'LLIS F1-score':f1})

true_predictions = S_safecom['tags']
true_labels = H_safecom['tags']
all_metrics = metric.compute(predictions=true_predictions, references=true_labels, zero_division=0)
ents = [ent for ent in all_metrics if 'overall' not in ent]
f1 = [all_metrics[ent]['f1'] for ent in ents]
ents.append('Average')
f1.append(all_metrics['overall_f1'])
f1_df['SAFECOM F1-score'] = f1
print(f1_df.round(3))

def clean_tags(df):
    cleaned_tags = []
    for i in range(len(df)):
        tags = df.iloc[i]['tags']
        clean_tags = [tag.split("-")[1] if "-" in tag else tag for tag in tags]
        cleaned_tags.append(clean_tags)
    df['clean_tags'] = cleaned_tags
    return df

H_annot_df = clean_tags(H_annot_df)
S_annot_df = clean_tags(S_annot_df)
H_safecom = clean_tags(H_safecom)
S_safecom = clean_tags(S_safecom)

S = [s for l in S_annot_df['clean_tags'].tolist() for s in l]
H = [h for l in H_annot_df['clean_tags'].tolist() for h in l]
labels = ['CAU', 'CON', 'EFF', 'MOD', 'REC']
f1 = f1_score(S, H, average=None, labels=labels)
f1_scores_with_labels = {label:score for label,score in zip(labels, f1)}
f1_scores_with_labels['Average'] = f1_score(S, H, average='weighted', labels=labels)
f1_df = pd.DataFrame({"Entity": f1_scores_with_labels.keys(), 
                        'LLIS F1-score': f1_scores_with_labels.values()})
S = [s for l in S_safecom['clean_tags'].tolist() for s in l]
H = [h for l in H_safecom['clean_tags'].tolist() for h in l]
labels = ['CAU', 'CON', 'EFF', 'MOD', 'REC']
f1 = f1_score(S, H, average=None, labels=labels)
f1_scores_with_labels = {label:score for label,score in zip(labels, f1)}
f1_scores_with_labels['Average'] = f1_score(S, H, average='weighted', labels=labels)
f1_df['SAFECOM F1-score'] = [f1_scores_with_labels[key] for key in labels+['Average']]
f1_df = f1_df.set_index('Entity')
print(f1_df.round(3))
print(f1_df.round(3).to_latex())
print("Cohens Kappa: ", cohen_kappa_score(S,H))