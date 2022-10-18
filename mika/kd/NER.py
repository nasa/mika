# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 12:33:13 2022

Utility functions for training custom NER models

@author: srandrad
"""
import pandas as pd
import os
import json
import numpy as np
import regex as re
# seqeval.metrics import classification_report
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support, accuracy_score
from datasets import load_metric
import seaborn as sn
import matplotlib.pyplot as plt
import copy
import matplotlib.cm as cm 
import matplotlib
matplotlib.style.use("seaborn")
plt.rcParams["font.family"] = "Times New Roman"

#device = 'cuda' if cuda.is_available() else 'cpu'

def read_doccano_annots(file, encoding=False):
    """
    Reads in a .jsonl file containing annotations from doccano
    
    Parameters
    ----------
    file : string
        File location.
    encoding : boolean, optional
        True if the file should be opened with utf-8 encoding. The default is False.

    Returns
    -------
    df : pandas dataframe
        Dataframe of the annotated document set. 

    """
    if encoding == False: f = open(file, "r") #safecom 
    else: f = open(file, "r", encoding='utf-8', errors='ignore') #LLIS hannah annots
    list_of_str_jsons = f.read().split("\n")[:-1]#removing last item which is empty
    list_of_dict_jsons = [json.loads(data) for data in list_of_str_jsons]
    df = pd.DataFrame(list_of_dict_jsons)
    return df

def clean_doccano_annots(df):
    """
    cleans annotated documents

    Parameters
    ----------
    df : pandas dataframe
        Dataframe of the annotated document set. 

    Returns
    -------
    df : pandas dataframe
        Dataframe of the cleaned annotated document set. 

    """
    for i in range(len(df)):
        text = df.iloc[i]['data']#.replace("  ", " ")
        label = df.iloc[i]['label']
        new_labels, new_text = clean_text_tags(text, label)
        df.at[i, 'label'] = new_labels
        df.at[i, 'data'] = new_text
    return df

def clean_annots_from_str(df):
    """
    
    cleans annotated documents by removing excess symbols
    use if the text is a string from a list of words
    
    Parameters
    ----------
    df : pandas dataframe
        Dataframe of the annotated document set. 
    
    Returns
    -------
    df : pandas dataframe
        Dataframe of the cleaned annotated document set. 

    """
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

def clean_text_tags(text, labels): 
    """
    cleans an individual text and label set by removing extra spaces or punctuation at
    the beginning or end of a string, fixing annotations that are missing a proceeding
    or preceeding character, and adding spaces into text that is misisng a space

    Parameters
    ----------
    text : string
        text accosicated with the labels
    labels : list
        list of labels from annotation where each label in the list is a tuple with
        structure (1,2,3), where 1 is the beginning character location, 2 is the end character
        location, and 3 is the label name

    Returns
    -------
    new_labels : list
        cleaned list of labels
    text : string
        cleaned string corresponding to the labels.

    """
    #input single text, list of labels [beg, end, tag]
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
        #else:
        if new_label[0] > 0 and text[new_label[0]-1]!=" ": #did not include first char
            preceeding_char = text[new_label[0]-1]
            if preceeding_char.isalpha() == True: 
                new_label = [new_label[0]-1, new_label[1], new_label[2]]
        if new_label[1]<len(text)-1 and text[new_label[1]] not in [" ", ".", ",", "?", "!"]: #did not include trailing chars
            proceeding_char = text[new_label[1]]
            if proceeding_char.isalpha() == True: 
                new_end = re.search(r'[.!?\S]', text[new_label[0]:]).end() + new_label[1]
                new_label = [new_label[0], new_end, new_label[2]]
        #case 3: missing space -> need to update text and following labels by adding in a space
        #this occurs when the preceeding or following character is punctuation
        if new_label[0] > 0 and (not text[new_label[0]-1].isalpha()) and text[new_label[0]-1]!=" ": #missing preceding space
            new_text = text[:new_label[0]] + ' ' + text[new_label[0]:]
            spaces_added += 1
            #spaces need to be added to future labels and this label
            new_label = [new_label[0]+1, new_label[1]+1, new_label[2]]
            add_spaces=True
        if new_label[1]<len(new_text)-2 and (not new_text[new_label[1]:new_label[1]+1].isalpha()) and new_text[new_label[1]:new_label[1]+1]!=" ": #missing proceeding space
            new_text = new_text[:new_label[1]] + ' ' + new_text[new_label[1]:]
            spaces_added += 1
            #spaces need to be added to future labels
            add_spaces=True
        #update text for added spaces
        text = new_text
        new_labels.append(new_label)
    return new_labels, text

def identify_bad_annotations(text_df):
    """
    Used for identifying invalid annotations which typically occur from inaccurate tagging,
    for example, tagging an extra space or punctuation. Must be ran prior to any model training
    or training will fail on bad annotations.

    Parameters
    ----------
    text_df : pandas DataFrame
        Dataframe storing the documents. Must contain 'docs' column and a 'tags' column.

    Returns
    -------
    bad_tokens : list
        list of tokens that are invalid.

    """
    inds_with_issues = [i for i in range(len(text_df)) if '-' in text_df.iloc[i]['tags']]
    text_df_issues = text_df.iloc[inds_with_issues][:]
    bad_tokens = []
    for ind in range(len(text_df_issues)):
        inds = [i for i, x in enumerate(text_df_issues.iloc[ind]['tags']) if x == "-"]
        [bad_tokens.append(text_df_issues.iloc[ind]['docs'][i]) for i in inds]
    return bad_tokens

def split_docs_to_sentances(text_df, id_col = 'Tracking #', tags=True): 
    """
    splits a dataframe of documents into a new dataframe where each sentance from
    each document is in its own row. This is useful for using BERT models with
    maximum character limits.

    Parameters
    ----------
    text_df : pandas DataFrame
        Dataframe storing the documents. Must contain 'docs' column and an id column.
    id_col : string, optional
        The column in the dataframe storing the doucment ids. The default is 'Tracking #'.
    tags : boolean, optional
        True if the dataframe also contains tags or annotations. False if no annotations
        are present. The default is True.

    Returns
    -------
    sentence_df : pandas DataFrame
        New dataframe where each row is a sentance.

    """
    #split each document into one row per sentance
    sentence_tags_total = []
    sentences_in_list = []
    ids = []
    for i in range(len(text_df)):
        doc = text_df.iloc[i]['docs']
        sentences = [sent for sent in doc.sents] #split into sentences
        if tags == True:
            total_sentence_tags = text_df.iloc[i]['tags']
            sentence_tags = [[tag for tag in  total_sentence_tags[sent.start:sent.end]] for sent in doc.sents]
            for tags_ in sentence_tags:
                sentence_tags_total.append(tags_)
        for sent in sentences:
            sentences_in_list.append(sent)
            ids.append(text_df.iloc[i][id_col])
    results_dict = {id_col:ids,"sentence": sentences_in_list}
    if tags == True:
        results_dict.update({"tags": sentence_tags_total})
    sentence_df = pd.DataFrame(results_dict)#{id_col:ids,
                                #"sentence": sentences_in_list,
                                #"tags": sentence_tags_total})
    return sentence_df


def check_doc_to_sentence_split(sentence_df):
    """
    tests that tags are preserved during document to sentence split

    Parameters
    ----------
    sentence_df : pandas DataFrame
        Dataframe where each row is a sentance. Has columns 'sentence' and 'tags'

    Returns
    -------
    None.

    """
    for i in range(len(sentence_df)):
        sent = sentence_df.iloc[i]['sentence']
        num_tokens = len([token.text for token in sent])
        num_tags = len(sentence_df.iloc[i]['tags'])
        if num_tokens != num_tags: print("error: the number of tokens does not equal the number of tags")


def align_labels_with_tokens(labels, word_ids):
    """
    Aligns labels and tokens for model training. 
    Adds special tokens to identify tokens not in the tokenizer and the begining of new words.    

    Parameters
    ----------
    labels : list
        List of NER labels where each label is a string
    word_ids : list
        List of work ids generated from a tokenizer

    Returns
    -------
    new_labels : list
        List of new labels with special token labels added.

    """
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

def tokenize_and_align_labels(sentence_df, tokenizer, align_labels=True):
    """
    tokenizes text and aligns with labels. Necessary due to subword tokenization with BERT.

    Parameters
    ----------
    sentence_df : pandas DataFrame
        Dataframe where each row is a sentance. Has columns 'sentence' and 'tags'
    tokenizer : tokenizer object
        tokenizer object, usually an autotokenizer Transformers BERT model
    align_labels : Boolean, optional
        True if labels should be aligned. The default is True.

    Returns
    -------
    tokenized_inputs : Object
        Tokenized inputw with new labels alligned to corresponding tokens

    """
    tokenized_inputs = tokenizer(sentence_df["tokens"], is_split_into_words=True)#, padding=True, truncation=True)
    if align_labels==True:
        all_labels = sentence_df["ner_tags"]
        new_labels = []
        for i, labels in enumerate(all_labels):
            word_ids = tokenized_inputs.word_ids(i)
            new_labels.append(align_labels_with_tokens(labels, word_ids))
    
        tokenized_inputs["labels"] = new_labels
    return tokenized_inputs

def tokenize(sentence_df, tokenizer):
    """
    Tokenizer used during dataset creation when the input data has already been tokenized.
    Currently unused, can be used to get token ids.
    Parameters
    ----------
    sentence_df : pandas DataFrame
        Dataframe where each row is a sentance. Has columns 'sentence' and 'tags'
    tokenizer : tokenizer object
        tokenizer object, usually an autotokenizer Transformers BERT model

    Returns
    -------
    tokenized_inputs : list
        list of tokens with corresponding IDs from the tokenizer

    """
    tokenized_inputs = tokenizer(sentence_df["tokens"], is_split_into_words=True)
    return tokenized_inputs

def compute_metrics(eval_preds, id2label):
    """
    Calculates sequence classification metrics. Used during trainig.
    
    Parameters
    ----------
    eval_preds : Tensor
        Pytorch tensor generated during training/evaluation.
    id2label : Dict
        Dict mapping numeric ids to labels.

    Returns
    -------
    Dict of metrics.

    """
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[id2label[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)]
    true_labels = [get_cleaned_label(l) for list_ in true_labels for l in list_]
    true_predictions = [get_cleaned_label(l) for list_ in true_predictions for l in list_]
    #metric = load_metric("seqeval")
    #all_metrics = metric.compute(predictions=true_predictions, references=true_labels, zero_division=0)
    labels=[lab for lab in set(true_predictions) if lab!='O']
    precision, recall, fscore, support = precision_recall_fscore_support(true_labels, true_predictions, average='weighted',  labels=labels)
    accuracy = accuracy_score(true_labels, true_predictions)
    return {"precision": precision, #all_metrics["overall_precision"],
            "recall": recall, #all_metrics["overall_recall"],
            "f1": fscore, #all_metrics["overall_f1"],
            "accuracy": accuracy} #all_metrics["overall_accuracy"]}


def compute_classification_report(labels, preds, pred_labels, label_list):
    """
    

    Parameters
    ----------
    labels : list
        true labels
    preds : Numpy Array
        predictions output from model
    pred_labels : List
        token labels corresponding to predictions. Used for removing special tokens
    label_list : Dict
        Maps label ids to NER labels.

    Returns
    -------
    Classification Report
        Output of classification results

    """
    true_labels = [[label_list[l] for l in label if l != -100] for label in labels]
    predictions = np.argmax(preds, axis=-1)
    labels = pred_labels
    true_predictions = [[label_list[p] for (p, l) in zip(prediction, label) if l != -100]
                        for prediction, label in zip(predictions, labels)]
    true_labels = [get_cleaned_label(l) for list_ in true_labels for l in list_]
    true_predictions = [get_cleaned_label(l) for list_ in true_predictions for l in list_]
    labels=[lab for lab in set(true_predictions) if lab!='O']
    return classification_report(true_labels, true_predictions, labels=labels)

def get_cleaned_label(label):
    """
    removes BILOU component of tag

    Parameters
    ----------
    label : string
        NER label potentially with BILOU tag.

    Returns
    -------
    label : string
        Label with BILOU component removed.

    """
    if "-" in label:
        return label.split("-")[1]
    else:
        return label
    
def build_confusion_matrix(labels, preds, pred_labels, label_list, save=False, savepath=""):
    """
    Creates confusion matrix for the muliclass NER classification

    Parameters
    ----------
    labels : list
        true labels
    preds : Numpy Array
        predictions output from model
    pred_labels : List
        token labels corresponding to predictions. Used for removing special tokens
    label_list : Dict
        Maps label ids to NER labels.
    save : Boolean, optional
        True to save the figure as pdf, false to not save. The default is False.
    savepath : string, optional
        Path to save the figure to. The default is "".

    Returns
    -------
    conf_mat : pandas DataFrame
        confusion matrix object
    true_predictions : list
        List of model prediction labels
    true_labels : list
        List of true labels

    """
    FONT=14
    true_labels = [get_cleaned_label(label_list[l]) for label in labels for l in label if l != -100 ]
    predictions = np.argmax(preds, axis=-1)
    labels = pred_labels
    true_predictions = [get_cleaned_label(label_list[p]) for prediction, label in zip(predictions, labels) for (p, l) in zip(prediction, label) if l != -100]
    entities = [str(l) for l in set([get_cleaned_label(l) for l in label_list.values()])]
    num_true_per_entity = np.unique(true_labels, return_counts=True)
    conf_mat = confusion_matrix(true_labels, true_predictions, labels=entities)#, normalize='true')
    cm_row_counts = [sum(row) for row in conf_mat]
    labels = [num_true_per_entity[0][np.where(num_true_per_entity[1]==row_sum)][0] for row_sum in cm_row_counts]
    my_cmap = copy.copy(cm.get_cmap('viridis')) # copy the default cmap
    conf_mat = confusion_matrix(true_labels, true_predictions, labels=entities, normalize='true')
    conf_mat = pd.DataFrame(conf_mat, index=labels, columns=labels)
    sn.heatmap(conf_mat, annot=True, annot_kws={"size": FONT}, cmap=my_cmap) #norm=LogNorm(), cmap=my_cmap, fmt='d') # font size
    plt.ylabel('True label', fontsize=FONT)
    plt.xlabel('Predicted label', fontsize=FONT)
    if save==True:
        plt.savefig(savepath+"confusion_matrix.pdf", bbox_inches="tight")
    plt.show()
    return conf_mat, true_predictions, true_labels


def read_trainer_logs(filepath, final_train_metrics, final_eval_metrics):
    """
    Reads training logs stored at a checkpoint to get training metrics

    Parameters
    ----------
    filepath : string
        Path to the checkpoint logs.
    final_train_metrics : Dict
        Dictionary of final training metrics in case they are not recorded in the log
    final_eval_metrics : Dict
        Dictionary of final eval metrics in case they are not recorded in the log

    Returns
    -------
    eval_df : pandas DataFrame
        Dataframe of evaluation metrics 
    training_df : pandas DataFrame
        Dataframe of training metrics 

    """
    df = pd.read_json(filepath)
    eval_dicts = []
    training_dicts = []
    for i in range(len(df)):
        if i%2 == 0: #training
            training_dicts.append(df.iloc[i]['log_history'])
        else:
            eval_dicts.append(df.iloc[i]['log_history'])
    if final_eval_metrics != {}:
        final_eval_metrics['steps'] = eval_dicts[0]['step']
        eval_dicts.append(final_eval_metrics)
    if final_train_metrics != {}:
        final_train_metrics = {"loss": final_train_metrics['train_loss'],#['training_loss'],
                               "epoch": final_train_metrics['epoch'],
                               "learning rate": "n/a"}
        training_dicts.append(final_train_metrics)
    eval_df = pd.DataFrame(eval_dicts)
    training_df = pd.DataFrame(training_dicts)
    return eval_df, training_df

def plot_loss(eval_df, training_df, save, savepath):
    """
    Plots the validation and training loss over training

    Parameters
    ----------
    eval_df : pandas DataFrame
        Dataframe of evaluation metrics 
    training_df : pandas DataFrame
        Dataframe of training metrics
    save : Boolean, optional
        True to save the figure as pdf, false to not save. The default is False.
    savepath : string, optional
        Path to save the figure to. The default is "".

    Returns
    -------
    None.

    """
    eval_loss = eval_df['eval_loss']
    training_loss = training_df['loss']
    epochs = training_df['epoch'].tolist()
    plt.figure(figsize=(15, 6))
    FONT=14
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, training_loss, label='training loss')
    plt.plot(epochs, eval_loss, label='validation loss')
    plt.xlabel('Number of epochs', fontsize=FONT)
    plt.ylabel('Binary cross entropy loss', fontsize=FONT)
    plt.legend(fontsize=FONT)
    plt.tick_params(axis='both', which='major', labelsize=FONT)
    if save:
        plt.savefig(savepath+"training_val_loss.pdf", bbox_inches="tight")
    plt.show()
    return

def plot_eval_metrics(eval_df, save, savepath):
    """
    plots classification metrics on the evaluation set over training

    Parameters
    ----------
    eval_df : pandas DataFrame
        Dataframe of evaluation metrics 
    save : Boolean, optional
        True to save the figure as pdf, false to not save. The default is False.
    savepath : string, optional
        Path to save the figure to. The default is "".

    Returns
    -------
    None.

    """
    metrics = ["eval_accuracy", "eval_f1", "eval_precision", "eval_recall"]
    epochs = eval_df['epoch'].tolist()
    plt.figure(figsize=(15, 6))
    FONT=14
    
    plt.subplot(1, 2, 1)
    for metric in metrics:
        plt.plot(epochs, eval_df[metric].tolist(), label=metric.split("_")[1])
    plt.xlabel('Number of epochs', fontsize=FONT)
    plt.ylabel('Score', fontsize=FONT)
    plt.legend(fontsize=FONT)
    plt.tick_params(axis='both', which='major', labelsize=FONT)
    if save:
        plt.savefig(savepath+"val_metrics.pdf", bbox_inches="tight")
    plt.show()
    return

def plot_eval_results(filepath, final_train_metrics={}, final_eval_metrics={}, save=False, savepath=None, loss=True, metrics=True):
    """
    PLots evaluation and training metrics as specified using other funcitons.

    Parameters
    ----------
    filepath : string
        Path to the checkpoint logs.
    final_train_metrics : Dict, optional
        Dictionary of final training metrics in case they are not recorded in the log. The default is {}.
    final_eval_metrics : Dict, optional
        Dictionary of final eval metrics in case they are not recorded in the log. The default is {}.
    save : Boolean, optional
        True to save the figure as pdf, false to not save. The default is False.
    savepath : string, optional
        Path to save the figure to. The default is "".
    loss : Boolean, optional
        True to plot loss False to not plot loss. The default is True.
    metrics : Boolean, optional
        True to plot evaluation metrics. The default is True.

    Returns
    -------
    None.

    """
    eval_df, training_df = read_trainer_logs(filepath, final_train_metrics, final_eval_metrics)
    if loss == True:
        plot_loss(eval_df, training_df, save, savepath)
    if metrics == True:
        plot_eval_metrics(eval_df, save, savepath)