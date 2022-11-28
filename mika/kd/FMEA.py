# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 16:46:38 2022

requires CUDA GPU set up for transformers and spacy

@author: srandrad
"""
import os
import pandas as pd
import numpy as np
from transformers import Trainer, pipeline, TrainingArguments, AutoTokenizer, DataCollatorForTokenClassification, BertForTokenClassification
from mika.kd.NER import build_confusion_matrix, compute_classification_report, split_docs_to_sentences, read_doccano_annots, clean_doccano_annots
#from models import FMEA_NER
from datasets import load_from_disk, Dataset
from torch import cuda, tensor
import torch
import spacy
from spacy.training import offsets_to_biluo_tags
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from pathlib import Path
import random
from nltk.corpus import words
spacy.prefer_gpu()
#spacy.require_gpu()

device = 'cuda' if cuda.is_available() else 'cpu'

class FMEA():
    """
    A class used for generated FMEAs from a data set of reports
    
    Attributes
    -----------
    
    Methods
    -----------
    load_model(self, model_checkpoint=None):
        loads custom NER model
    load_data(self, filepath='', df=None, formatted=False, text_col="Narrative", id_col="Tracking #", label_col="labels"):
        loads are preprocesses data for NER
    predict(self):
        performs NER
    get_entities_per_doc(self, pred=True):
        gets all the entities for each document
    group_docs_with_meta(self, grouping_col='UAS_cleaned', cluster_by=['CAU', 'MOD'], additional_cols=['Mission Type']):
        groups docs into fmea rows using metadata. Under development.
    group_docs_manual(self, filename='', grouping_col='Mode', additional_cols=['Mission Type'], sample=1):
        groups docs into fmea rows according to a file manually defining which docs go in which row
    post_process_fmea(self, rows_to_drop=[], id_name='SAFECOM', phase_name='Mission Type', max_words=20):
        post processes fmea by including the operational phase, cleaning sub-word tokens, and truncating the number of words per cell in the FMEA
    calc_frequency(self, year_col=''):
        calculates the frequency for each FMEA row and assigns it a category
    calc_severity(self, severity_func, from_file=False, file_name='', file_kwargs={}):
        calculates the severity for each FMEA row
    calc_risk(self):
        calculates the risk for each FMEA row as the product of severity and frequency
    build_fmea(self, severity_func, group_by, group_by_kwargs={}, post_process_kwargs={}):
        builds FMEA using previous functions
    display_doc(self, doc_id, save=True, output_path="", colors_path=None, pred=True):
        displays an annotated document, either from NER outputs or manually labeled data.
    """
    
    __english_vocab = set([w.lower() for w in words.words()])
    
    def __init__(self):
        """
        initializes FMEA object

        Returns
        -------
        None.

        """
        return
    
    def load_model(self, model_checkpoint=None):
        """
        Loads in a fine-tuned custom NER model trained to extract FMEA entities
        If no checkpoint is passed, the custom model from MIKA is used
        Parameters
        ----------
        model_checkpoint : string, optional
            model check point, can be from huggingface or a path from personal device

        Returns
        -------
        None.

        """
        if model_checkpoint:
            self.token_classifier = pipeline("token-classification", model=model_checkpoint, aggregation_strategy="simple", device=-1)#sets model on cpu
        #else:
        #    self.token_classifier = FMEA_NER
    def load_data(self, text_col, id_col, filepath='', df=None, formatted=False, label_col="labels"):
        """
        Loads data to prepare for FMEA extraction. 
        Sentence tokenization is performed for preprocessing, and the raw data is also saved.
        Can input a filepath for a .jsonl (annotations from doccano) or .csv file.
        Can also instead input a dataframe already loaded in.
        Can also instead input a huggingface dataset object location.
        Saves data formatted to input into the NER model.
        Parameters
        ----------
        text_col : string
            The column where the text used for FMEA extraction is stored.
        id_col : string
            The id column in the dataframe. 
        filepath : string, optional
            Can input a filepath for a .jsonl (annotations from doccano) or .csv file. The default is ''.
        df : pandas DataFrame, optional
            Can instead input a pandas DataFrame already loaded in, with one column of text. The default is None.
        formatted : Bool, optional
            True if the input in filepath is a formatted dataset object. The default is False.
        
        label_col : string, optional
            The column containing annotation labels if the data is annotated. The default is "labels".

        Returns
        -------
        None.

        """
        self.id_col = id_col
        self.text_col = text_col
        if formatted == True:
            self.input_data = load_from_disk(filepath)
            self.true_labels = self.input_data[label_col]
        elif formatted == False: 
            if '.csv' in filepath or df is not None: #replace with Data class to limit testing needs?
                if df is None:
                    data = pd.read_csv(filepath)#, index_col=0)
                    data = data.dropna(subset=[text_col])
                else:
                    data = df
                    data = data.dropna(subset=[text_col])
                #sentences
                self.nlp = spacy.load("en_core_web_trf")
                self.nlp.add_pipe("sentencizer")
                docs = data[text_col].tolist()
                docs = [self.nlp(doc) for doc in docs]
                data['docs'] = docs
                self.raw_df = data
                sentence_df = split_docs_to_sentences(data, id_col=id_col,tags=False)
                self.data_df = sentence_df 
                self.data_df['sentence'] = [sent.text for sent in sentence_df["sentence"].tolist()]
                self.input_data = self.data_df['sentence'].tolist()
            elif '.jsonl' in filepath:
                test_data = read_doccano_annots(filepath)
                test_data = clean_doccano_annots(test_data)
                #break into sentences and tags
                self.nlp = spacy.load("en_core_web_trf")
                self.nlp.add_pipe("sentencizer")
                test_docs = test_data[text_col].tolist()
                docs = [self.nlp(doc) for doc in test_docs]
                test_data['docs'] = docs
                self.raw_df = test_data
                test_data['tags'] = [offsets_to_biluo_tags(test_data.at[i,'docs'], test_data.at[i,'label']) for i in range(len(test_data))]
                sentence_df = split_docs_to_sentences(test_data, id_col=id_col, tags=True)
                self.data_df = sentence_df 
                self.data_df['sentence'] = [sent.text for sent in sentence_df["sentence"].tolist()]
                self.input_data = self.data_df['sentence'].tolist()
                self.true_labels = self.data_df['tags']
            
    def predict(self):
        """
        Performs named entity recognition on the input data
        Returns
        -------
        Preds: 
            Predicted entities per each document

        """
        self.preds = self.token_classifier(self.input_data)
        return self.preds
    
    def evaluate_preds(self, cm=True, class_report=True): 
        """
        
        Can only be used if the input data is labeled. 
        Evaluates the performance of the NER model against labeled data.
        Parameters
        ----------
        cm : Boolean, optional
            Creates a confusion matrix if True. The default is True.
        class_report : Boolean, optional
            Creates a classification report if True. The default is True.

        Returns
        -------
        return_vals : Dictionary
            Dict containing confusion matrix and classification report if specified.

        """
        #probably wont work with the pipeline... do this in training
        
        return_vals = {}
        if cm == True:
            cm, true_predictions, labels = build_confusion_matrix(self.true_labels, self.raw_pred, self.pred_labels, self.id2label)
            return_vals['Confusion Matrix'] = cm
        if class_report == True:
            classification_report = compute_classification_report(self.true_labels, self.raw_pred, self.pred_labels, self.id2label)
            return_vals['Classification Report'] = classification_report
        return return_vals
    
    def __update_entities_per_sentence(self):
        """
        
        Returns
        -------
        None.

        """
        #for each id: get len of text, add on to stop and start for each new sentence. 
        ids = self.raw_df[self.id_col].tolist()
        doc_ents = []
        for i in ids:
            id_df = self.data_df.loc[self.data_df[self.id_col]==i].reset_index(drop=True)
            text_len = len(id_df.iloc[0]['sentence']) +1 #extra for the added space
            for j in range(len(id_df)): #update all entities after the first sentence
                if j != 0:
                    ents = id_df.iloc[j]['predicted entities']
                    new_ents = []
                    for ent in ents:
                        new_ent = ent
                        new_ent['start'] += text_len
                        new_ent['end'] += text_len 
                        new_ents.append(new_ent)
                    text_len += len(id_df.iloc[j]['sentence']) +1 #extra for the added space
                else:
                    new_ents = id_df.iloc[j]['predicted entities']
                doc_ents.append(new_ents)
        self.data_df['document entities'] = doc_ents
    
    def get_entities_per_doc(self, pred=True):
        """
        Gets all entites for each document. 
        Note that this is required because the NER model is run on sentences.
        This function reconstructs the documents from the sentences, while preserving the entities.

        Parameters
        ----------
        pred : Boolean, optional
            True if the entities per doc are from predicted entities.
            False if the entities per doc are from labels.
            The default is True.

        Returns
        -------
        data_df
            pandas DataFrame with documents as rows, entities as columns

        """
        if pred==True:
            self.data_df['predicted entities'] = self.preds
        else: 
            self.data_df['predicted entities'] = self.data_df['tags']
        self.__update_entities_per_sentence()
        self.data_df = self.data_df.groupby(self.id_col).agg({'document entities': 'sum', 'sentence': lambda x: ' '.join(x)})
        #go through predicted entities
        #for each entity add to the corresponding list
        total_entities = {'CAU': [], "MOD": [], "EFF": [], "CON": [], "REC": []}
        for i in range(len(self.data_df)):
            entity_list = self.data_df.iloc[i]['document entities']
            entities_per_doc = {'CAU': "", "MOD": "", "EFF": "", "CON": "", "REC": ""}
            for entity_dict in entity_list:
                entity_group = entity_dict['entity_group']
                if entities_per_doc[entity_group] == "":
                    entities_per_doc[entity_group] += entity_dict['word']
                else:
                    entities_per_doc[entity_group] += ", " + entity_dict['word']
            for ent in total_entities:
                total_entities[ent].append(entities_per_doc[ent])  
        for ent in total_entities:
            self.data_df[ent] = total_entities[ent]
        self.data_df[self.id_col] = self.data_df.index.tolist()
        return self.data_df
    
    def group_docs_with_meta(self, grouping_col, additional_cols=[], sample=1):
        """
        Currently unused and in operable
        Intented function is to group documents into an FMEA using a grouping column,
        which is metadata from the initial dataset.

        Parameters
        ----------
        grouping_col : string
            The column in the original dataset used to group documents into FMEA rows.
        additional_cols : list of strings, optional
            additional columns in a dataset to include in the FMEA. The default is [].
        sample : int, optional
            Number of samples to pull for each FMEA row. The default is 1.

        Returns
        -------
        grouped_df : DataFrame
            The grouped FMEA dataframe

        """
        if additional_cols != []:
            for col in additional_cols:
                self.data_df[col] = self.raw_df[col].tolist()
        temp_grouped_df = self.data_df.copy()
        #group reports by category/mission type/phase
        clusters = []
        for id in temp_grouped_df[self.id_col].tolist(): #could instead reindex so ids in raw are ids in data_df
            cluster = self.raw_df.loc[self.raw_df[self.id_col]==id].reset_index(drop=True).at[0,grouping_col]
            clusters.append(cluster)
        temp_grouped_df['cluster'] = clusters#self.raw_df[grouping_col].tolist()
        #cluster = self.raw_df[grouping_col].tolist()
        agg_dict = {'CAU': lambda x: '; '.join([i for i in x if i!="" and type(i)==str]),
                    'MOD': lambda x: '; '.join([i for i in x if i!="" and type(i)==str]),
                    'EFF': lambda x: '; '.join([i for i in x if i!="" and type(i)==str]),
                    'CON': lambda x: '; '.join([i for i in x if i!="" and type(i)==str]),
                    'REC': lambda x: '; '.join([i for i in x if i!="" and type(i)==str]),
                    self.id_col: lambda x: '; '.join([str(i) for i in x])}#str(x))} #this may not work for all data sets
        ad_col_dict = {col: lambda x: '; '.join(set([str(i) for i in x])) for col in additional_cols}
        agg_dict.update(ad_col_dict)
        self.grouped_df = temp_grouped_df.groupby('cluster').agg(agg_dict)

        sampled_ids = []
        for i in range(len(self.grouped_df)):
            ids = self.grouped_df.iloc[i][self.id_col].split("; ")
            sampled = random.sample(ids, min(len(ids),sample))
            sampled = "; ".join(sampled)
            sampled_ids.append(sampled)
        self.grouped_df['Sampled '+self.id_col] = sampled_ids
        return self.grouped_df
        
    
    def group_docs_manual(self, filename, grouping_col, additional_cols=[], sample=1):
        """
        Creates FMEA rows by grouping together documents according to values manually defined in a separate file.
        Loads in the file and then aggregates the data. Sample IDs for documents in each row are created as well.
        Parameters
        ----------
        filename : string
            filepath to the spreadsheet defining the rows.
        grouping_col : string
            The column within the spreadsheet that defines the rows.
        additional_cols : list, optional
            Additional columns to include in the FMEA.
        sample : int, optional
            Number of samples to pull for each FMEA row. The default is 1.

        Returns
        -------
        grouped_df : DataFrame
            The grouped FMEA dataframe

        """
        if '.xlsx' in filename: 
            manual_groups = pd.read_excel(filename)
        elif '.csv' in filename:
            manual_groups = pd.read_csv(filename)
        if additional_cols != []:
            for col in additional_cols:
                self.data_df[col] = self.raw_df[col].tolist()
        cluster = []
        new_data_df = self.data_df.copy()
        rows_added = 0
        for i in range(len(self.data_df)):
            id_ = self.data_df.iloc[i][self.id_col]
            group = manual_groups.loc[manual_groups[self.id_col]==id_].reset_index(drop=True)
            if len(group) == 1: #reports with one hazard
                cluster.append(group.at[0,grouping_col])
            elif len(group) == 0: #reports with no hazards
                cluster.append('misc')
            elif len(group) >= 2: #reports with 2 or more hazards #something is wrong here!!
                for j in range(len(group)):
                    cluster.append(group.at[j,grouping_col])
                    if j>0:
                        new_data_df = pd.concat([new_data_df.iloc[:i+rows_added][:],self.data_df.iloc[i:i+1][:], new_data_df.iloc[i+rows_added:][:]]).reset_index(drop=True)
                        rows_added += 1
        self.data_df_all_rows = new_data_df
        self.data_df_all_rows['cluster'] = cluster #need to add extra rows to data df for documents in multiple clusters
        agg_dict = {'CAU': lambda x: '; '.join([i for i in x if i!="" and type(i)==str]),
                    'MOD': lambda x: '; '.join([i for i in x if i!="" and type(i)==str]),
                    'EFF': lambda x: '; '.join([i for i in x if i!="" and type(i)==str]),
                    'CON': lambda x: '; '.join([i for i in x if i!="" and type(i)==str]),
                    'REC': lambda x: '; '.join([i for i in x if i!="" and type(i)==str]),
                    self.id_col: lambda x: '; '.join([str(i) for i in x])}#str(x))} #this may not work for all data sets
        ad_col_dict = {col: lambda x: '; '.join(set([i.replace("Fire, ","") for i in x])) for col in additional_cols}
        agg_dict.update(ad_col_dict)
        self.grouped_df = self.data_df_all_rows.groupby('cluster').agg(agg_dict)

        sampled_ids = []
        for i in range(len(self.grouped_df)):
            ids = self.grouped_df.iloc[i][self.id_col].split("; ")
            sampled = random.sample(ids, min(len(ids),sample))
            sampled = "; ".join(sampled)
            sampled_ids.append(sampled)
        self.grouped_df['Sampled '+self.id_col] = sampled_ids
        return self.grouped_df
    
    def post_process_fmea(self, id_name='ID', phase_name='Mission Type', max_words=20):
        """
        Post processes the FMEA to identify the column that contains the phase name,
        clean sub-word tokens, and limit the number of words per cell.

        Parameters
        ----------
        id_name : string, optional
            Name of dataset used/name over the id column. The default is 'SAFECOM'.
        phase_name : string, optional
            Column that can be used to find the phase of operation. The default is 'Mission Type'.
        max_words : int, optional
            Maximum number of words in a cell in the FMEA. The default is 20.

        Returns
        -------
        fmea_df : DataFrame
            FMEA post processed DataFrame

        """
        #clean data in NER columns: remove duplicate words, form tokens from token pieces
        for i in self.grouped_df.index:
            for ent in ['CAU', 'MOD', 'EFF', 'CON', 'REC']:
                text = self.grouped_df.loc[i, ent]
                docs = text.split("; ")
                new_docs = []
                for doc_text in docs:
                    doc_text = doc_text.split(", ")
                    new_text = []
                    for word in doc_text:
                        if "##" in word:
                            #check prev word
                            if doc_text.index(word) != 0 and len(new_text)>0:
                                new_word = new_text[-1] + word.strip("##") #combines with previous word
                                del new_text[-1] #deletes previous word
                                if new_word in self.__english_vocab:
                                    new_text.append(new_word) # updates with combined word if its a real word
                                elif word.strip("##") == 'ua':
                                    new_text.append('uas')
                            else:
                                # do not append word
                                continue
                        else:
                            new_text.append(word)
                    new_docs.append(", ".join(new_text))
                new_docs = ", ".join(new_docs)
                seen = set()
                new_docs = ', '.join(seen.add(i) or i for i in new_docs.split(", ") if i not in seen)
                self.grouped_df.at[i, ent] = new_docs
        #format columns
        col_to_label = {'CAU': "Cause", 'MOD': "Failure Mode", 'EFF': "Effect",
                        "CON": "Control Process", "REC": "Recommendations",
                        "frequency": 'Frequency', 'severity':"Severity",
                        "Sampled "+self.id_col: id_name,"risk": "Risk"}
        fmea_cols = ['CAU', 'MOD', 'EFF', 'CON', 'REC', 'frequency', 'severity', "risk", "Sampled "+self.id_col]
        if phase_name != '':
            col_to_label.update({phase_name: 'Phase'})
            fmea_cols = [phase_name] + fmea_cols 
            
        self.fmea_df = self.grouped_df[fmea_cols]
        self.fmea_df.columns = [col_to_label[col] for col in self.fmea_df.columns]
        if max_words is not None:
            for col in ['Cause', 'Failure Mode', 'Effect', 'Control Process', "Recommendations"]:
                for i in self.fmea_df.index:
                    self.fmea_df.at[i, col] = " ".join(self.fmea_df.at[i,col].split(" ")[:max_words])
        return self.fmea_df
    
    def get_year_per_doc(self, year_col, config='/'):
        """
        Used to convert dates to years prior to calculating frequency

        Parameters
        ----------
        year_col : string,
            The colomn in the raw dataframe with the date information.
        config : TYPE, optional
            DESCRIPTION. The default is '/'.

        Returns
        -------
        None.

        """
        if config == '/':
            self.raw_df['Year'] = [self.raw_df.at[i,year_col].split('/')[-1].split(" ")[0] for i in range(len(self.raw_df))]
        elif config == 'id':
            self.raw_df['Year'] = [num.split("-")[0] for num in self.raw_df[self.id_col]]
            
    def calc_frequency(self, year_col): 
        """
        Calculates the frequency for each row and assigns it a category

        Parameters
        ----------
        year_col : string
            The column the year for the report is stored in. 

        Returns
        -------
        grouped_df : DataFrame
            Grouped df with requency column added

        """
        #something is wrong with the frequency
        
        self.grouped_df[self.id_col]
        #total frequency
        frequency = [len(ids.split("; ")) for ids in self.grouped_df[self.id_col]]
        self.grouped_df['Total Frequency'] = frequency
        #frequency per each year
        years = list(set([y for y in self.raw_df[year_col]]))
        years.sort()
        years_frequency = {year:[] for year in years}
        for i in range(len(self.grouped_df)):
            ids = self.grouped_df.iloc[i][self.id_col].split("; ")
            year_ids = self.raw_df.loc[self.raw_df[self.id_col].isin(ids)][year_col].tolist()
            for year in years_frequency:
                years_frequency[year].append(year_ids.count(year))
        for year in years_frequency:
            self.grouped_df[str(year)+" Frequency"] = years_frequency[year]
        # FAA frequency: 
        self.grouped_df['rate'] = self.grouped_df['Total Frequency']/len(years)
        FAA_freqs = []
        for i in range(len(self.grouped_df)):
            rate = self.grouped_df.iloc[i]['rate']
            max_year_freq = max([self.grouped_df.iloc[i][str(year)+" Frequency"] for year in years])
            if rate > 100 or max_year_freq > 100:
                freq = 5 # occurs 100 times a year
            elif rate > 10 or max_year_freq >10:
                freq = 4 # occurs between 10 and a hundered times per year
            elif rate > 1 or max_year_freq >1:
                freq = 3 # occurs less than 10 times per year
            elif max_year_freq == 1 and rate > 1/len(years):
                freq = 2 # occurs once per year, occurs more than once over the years in the data
            elif rate == 1/len(years) or rate < 0.1: 
                freq = 1 # occurs only once, or once every 10 years
            FAA_freqs.append(freq)
        self.grouped_df['frequency'] = FAA_freqs
        return self.grouped_df
    
    def calc_severity(self, severity_func, from_file=False, file_name='', file_kwargs={}):
        """
        Calculates the severity for each row according to a defined severity function.

        Parameters
        ----------
        severity_func : function
            User defined function for calculating severity. Usually a linear combination of other values.
        from_file : Boolean, optional
            True if the severity value is already stored in a file, false if calculated from a severiy function. The default is False.
        file_name : string, optional
            filepath to a spread sheet containing the severity value for each document. The default is ''.
        file_kwargs : dict, optional
            any kwargs needed to read the file. Typically needed for .xlsx workbooks with multiple sheets. The default is {}.

        Returns
        -------
        None.

        """
        #severity func calculates the severity for each report in the raw df
        if from_file == False:
            self.raw_df = severity_func(self.raw_df)
            #need to get average severity for row in grouped df
            severity_per_row = []
            for i in range(len(self.grouped_df)):
                ids = self.grouped_df.iloc[i][self.id_col].split("; ")
                temp_df = self.raw_df.loc[self.raw_df[self.id_col].astype(str).isin(ids)]
                severities = temp_df['severity'].tolist()
                severity_per_row.append(np.average(severities))
            self.grouped_df['severity'] = severity_per_row
        else:
            if '.xlsx' in file_name:
                file_df = pd.read_excel(file_name, **file_kwargs)[file_kwargs['sheet_name'][0]] 
            elif '.csv' in file_name:
                file_df = pd.read_csv(file_name, **file_kwargs)
            #make sure it is in same order as grouped df
            file_df = severity_func(file_df, self.grouped_df)
            self.grouped_df['severity'] = file_df['severity'].tolist() 
        return
    
    def calc_risk(self):
        """
        
        Calculates risk as the product of severity and frequency.
        Adds risk column to the grouped df
        Returns
        -------
        None.

        """
        self.grouped_df['risk'] = self.grouped_df['severity'] * self.grouped_df['frequency']
        return
    
    def build_fmea(self, severity_func, group_by, year_col, group_by_kwargs={}, post_process_kwargs={}, save=True):
        """
        Builds the FMEA using the above functions, all in one call.
        Less customizable, but useful for a quick implementation.

        Parameters
        ----------
        severity_func : function
            DESCRIPTION.
        group_by : 'string'
            method to group together the FMEA rows, either manual file or by meta data.
        year_col : string
            The column the year for the report is stored in. 
        group_by_kwargs : dict, optional
            dictionary containing all inputs for the group_by function. The default is {}.
        post_process_kwargs : dict, optional
            dictionary containing all inputs for the post_process_fmea function. The default is {}.

        Returns
        -------
        None.

        """
        self.get_entities_per_doc()
        if group_by == 'manual':
            self.group_docs_manual(**group_by_kwargs)
        elif group_by == 'meta':
            self.group_docs_with_meta(**group_by_kwargs)
        self.calc_frequency(year_col)
        self.calc_severity(severity_func)
        self.calc_risk()
        #prune excess columns, add mission phase
        self.post_process_fmea(**post_process_kwargs)
        #save table
        if save == True:
            self.fmea_df.to_csv(os.path.join(os.getcwd(),"fmea.csv"))
    
    def display_doc(self, doc_id, save=True, output_path="", colors_path=None, pred=True):
        """
        Displays an annotated document with entities highlighted accordingly

        Parameters
        ----------
        doc_id : string
            The id of the document to be dispalyed
        save : Boolean, optional
            Saves as html if true. Displays if False. The default is True.
        output_path : string, optional
            The filepath the display will be saved to. The default is "".
        colors_path : string, optional
            The path to a file that defines the colors to be used for each entity. The default is None.
        pred : Boolean, optional
            True if the displayed document is from predictions, False if from manual annotations. The default is True.

        Returns
        -------
        html : TYPE
            DESCRIPTION.

        """
        #see https://spacy.io/usage/visualizers
        #doc = self.data_df.loc[self.data_df[self.id_col]==str(doc_id)].reset_index(drop=True)
        if pred == False:
            text_col = self.text_col
            ent_col = 'label'
            doc = self.raw_df.loc[self.raw_df[self.id_col]==str(doc_id)].reset_index(drop=True)
        else: 
            text_col = 'sentence'
            ent_col = 'document entities'
            doc = self.data_df.loc[self.data_df[self.id_col]==str(doc_id)].reset_index(drop=True)
        text = doc.iloc[0][text_col]
        ents = doc.iloc[0][ent_col]
        if pred == True:
            cleaned_ents = []
            for ent in ents:
                ent['label'] = ent.pop('entity_group')
                cleaned_ents.append(ent)
        elif pred == False:
            cleaned_ents = []
            for ent in ents:
                ent_dict = {"start":ent[0], "end":ent[1], "label":ent[2]}
                cleaned_ents.append(ent_dict)
            
        ent_input = {"text": text,
                     "ents": cleaned_ents,
                     "title": doc_id}
        if colors_path:
            colors = pd.read_json(colors_path)
            colors_dict = {colors.iloc[i]['text']: colors.iloc[i]['backgroundColor'] for i in range(len(colors))}
        else:
            colors_dict = {}
        options = {"colors": colors_dict}
        html = spacy.displacy.render(ent_input, options=options, style="ent", manual=True, jupyter=False, page=True)
        if save == True:
            if output_path == "":
                output_path = os.path.join(os.getcwd(), str(doc_id)+"_display")
                pdf_path = output_path+".pdf"
            output = Path(output_path+".html")
            output.open("w", encoding="utf-8").write(html)
            #doc = aw.Document(output_path+'.html')
            #for page in range(0, doc.page_count):
            #    extractedPage = doc.extract_pages(page, 1)
            #    extractedPage.save(output_path+".svg")
            #path_wkhtmltopdf = r"C:\Users\srandrad\Anaconda3\Lib\site-packages\wkhtmltox-0.12.6-1.msvc2015-win32.exe"
            #config = pdfkit.configuration(wkhtmltopdf=path_wkhtmltopdf)
            #pdfkit.from_string(svg, output_path+".pdf", configuration=config)
            #svg2pdf(url=output_path+".svg", write_to=pdf_path,output_width=4, output_height=2)
        elif save == False:
            spacy.displacy.serve(ent_input, options=options, style="ent", manual=True)
        return html

#if __name__ == '__main__':
#    freeze_support()
