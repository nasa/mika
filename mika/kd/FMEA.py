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
from mika.kd.NER import build_confusion_matrix, compute_classification_report, split_docs_to_sentances, tokenize, tokenize_and_align_labels, read_doccano_annots, clean_doccano_annots
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
    __english_vocab = set([w.lower() for w in words.words()])
    
    def __init__(self):
        return
    
    def load_model(self, model_checkpoint):
        self.token_classifier = pipeline("token-classification", model=model_checkpoint, aggregation_strategy="simple", device=-1)#sets model on cpu
        
    def load_data(self, filepath, formatted=False, text_col="Narrative", id_col="Tracking #", label_col="labels"):
        self.id_col = id_col
        self.text_col = text_col
        if formatted == True:
            self.input_data = load_from_disk(filepath)
            self.true_labels = self.input_data[label_col]
        elif formatted == False:
            if '.csv' in filepath:
                test_data = pd.read_csv(filepath)#, index_col=0)
                test_data = test_data.dropna(subset=[text_col])
                #sentences
                self.nlp = spacy.load("en_core_web_trf")
                self.nlp.add_pipe("sentencizer")
                test_docs = test_data[text_col].tolist()
                docs = [self.nlp(doc) for doc in test_docs]
                test_data['docs'] = docs
                self.raw_df = test_data
                sentence_df = split_docs_to_sentances(test_data, id_col=id_col,tags=False)
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
                sentence_df = split_docs_to_sentances(test_data, id_col=id_col, tags=True)
                self.data_df = sentence_df 
                self.data_df['sentence'] = [sent.text for sent in sentence_df["sentence"].tolist()]
                self.input_data = self.data_df['sentence'].tolist()
                self.true_labels = self.data_df['tags']
            
    def predict(self):
        self.preds = self.token_classifier(self.input_data)
        return self.preds
    
    def evaluate_preds(self, cm=True, class_report=True): #probably wont work with the pipeline... do this in training
        return_vals = {}
        if cm == True:
            cm, true_predictions, labels = build_confusion_matrix(self.true_labels, self.raw_pred, self.pred_labels, self.id2label)
            return_vals['Confusion Matrix'] = cm
        if class_report == True:
            classification_report = compute_classification_report(self.true_labels, self.raw_pred, self.pred_labels, self.id2label)
            return_vals['Classification Report'] = classification_report
        return return_vals
    
    """
    for each id:
        get len of text, add on to stop and start for each new sentence. 
    """
    def update_entities_per_sentence(self):
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
        if pred==True:
            self.data_df['predicted entities'] = self.preds
        else: 
            self.data_df['predicted entities'] = self.data_df['tags']
        self.update_entities_per_sentence()
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
    
    def group_docs_with_meta(self, grouping_col='UAS_cleaned', cluster_params={}, cluster_by=['CAU', 'MOD'], additional_cols=['Mission Type']):
        #Currently unused
        self.data_df[self.id_col] = self.data_df.index.tolist()
        temp_grouped_df = self.data_df.copy()
        #group reports by category/mission type/phase
        temp_grouped_df[grouping_col] = self.raw_df[grouping_col].tolist()
        for col in additional_cols:
            temp_grouped_df[col] = self.raw_df[col].tolist()
        grouping_col_vals = temp_grouped_df[grouping_col].unique()
        grouped_dfs = {val: temp_grouped_df.loc[temp_grouped_df[grouping_col]==val].reset_index(drop=True) for val in grouping_col_vals if val is not np.nan}
        if len(grouping_col_vals) > len(grouped_dfs):
            grouped_dfs['nan'] = temp_grouped_df.loc[temp_grouped_df[grouping_col].isnull()].reset_index(drop=True)
        #within each category/mission type/phase, group by mode/cause
        clustered_dfs = []
        for val in grouped_dfs:
            df = grouped_dfs[val]
            grouped_df = self.cluster(df, cluster_by, additional_cols)
            grouped_df[grouping_col] = [val for i in range(len(grouped_df))]
            clustered_dfs.append(grouped_df)
        self.grouped_df = pd.concat(clustered_dfs)
        return self.grouped_df
    
    def cluster(self, df, cluster_by=[], additional_cols=[]):
        #Currently unused
        text_list = []
        for i in range(len(df)):
            text = ""
            for col in cluster_by:
                text += ". " + df.iloc[i][col]
            text_list.append(text)
        vecs = []
        for doc in self.nlp.pipe(text_list):
            #print(doc.text)
            if doc.text != "" and len(doc._.trf_data.tensors)>0:
                vecs.append(doc._.trf_data.tensors[1][0])
            else: 
                vecs.append([])
        correct_length = max([len(vecs[i]) for i in range(len(vecs))])
        empty_vals = np.zeros(correct_length)
        vecs = [v if v != [] else empty_vals for v in vecs]
        vecs = np.stack(vecs)
        #cluster
        clustering_model = DBSCAN(eps=3.00, min_samples=1).fit(vecs)
        #clustering_model = AgglomerativeClustering(n_clusters=None, affinity='cosine', linkage='average', distance_threshold=1).fit(vecs)
        labels = clustering_model.labels_
        df['cluster'] = labels
        #combine
        agg_function = {'CAU': lambda x: '; '.join(x),
                        'MOD': lambda x: '; '.join(x),
                        'EFF': lambda x: '; '.join(x),
                        'CON': lambda x: '; '.join(x),
                        'REC': lambda x: '; '.join(x),
                        self.id_col: lambda x: '; '.join(x)}
        for col in additional_cols:
            agg_function[col] = lambda x: '; '.join(x)
        grouped_df = df.groupby('cluster').agg(agg_function)
        return grouped_df
        
    def group_docs(self, grouping_col='Mission Type', db_params={'eps':1.00, 'min_samples':1}, cluster_by=['CAU', 'MOD']):
        #Currently unused
        #ideas: cluster by cause, then mode, then combine all effects, controls, and recs
        #issues: identifying number of clusters - DBSCAN clustering
        causes = self.data_df['CAU'].tolist()
        causes_modes = self.data_df['sentence']#['MOD'] #+". "+ self.data_df['MOD']#+". "+ self.data_df['EFF']+". "+self.data_df['CON']
        causes_modes = causes_modes.tolist()
        vecs = []
        for doc in self.nlp.pipe(causes_modes):
            if doc.text != "" and len(doc._.trf_data.tensors)>0:
                vecs.append(doc._.trf_data.tensors[1][0])
            else: 
                vecs.append([])
        correct_length = max([len(vecs[i]) for i in range(len(vecs))])
        empty_vals = np.zeros(correct_length)
        vecs = [v if v != [] else empty_vals for v in vecs]
        vecs = np.stack(vecs)
        #get mode vectors
        mode_vecs = []
        modes = self.data_df['MOD'].tolist()
        for doc in self.nlp.pipe(modes):
            if doc.text != "" and len(doc._.trf_data.tensors)>0:
                mode_vecs.append(doc._.trf_data.tensors[1][0])
            else: 
                mode_vecs.append([])
        #perform clustering: DBSCAN
        clustering_model = DBSCAN(**db_params).fit(vecs)
        clustering_model = AgglomerativeClustering(n_clusters=None, affinity='cosine', linkage='average', distance_threshold=0.01).fit(vecs)
        labels = clustering_model.labels_
        
        self.data_df['cluster'] = labels
        self.data_df[self.id_col] = self.data_df.index.tolist()
        self.grouped_df = self.data_df.groupby('cluster').agg({'CAU': lambda x: '; '.join(x),
                                                               'MOD': lambda x: '; '.join(x),
                                                               'EFF': lambda x: '; '.join(x),
                                                               'CON': lambda x: '; '.join(x),
                                                               'REC': lambda x: '; '.join(x),
                                                               self.id_col: lambda x: '; '.join(x)})
        #get most representatice doc somehow?
        return self.grouped_df
    
    def group_docs_manual(self, filename='', grouping_col='Mode', additional_cols=['Mission Type'], sample=1):
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
    
    def post_process_fmea(self, rows_to_drop=[], id_name='SAFECOM', phase_name='Mission Type', max_words=20):
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
    
    def get_year_per_doc(self, year_col='', config='/'):
        if config == '/':
            self.raw_df['Year'] = [self.raw_df.at[i,year_col].split('/')[-1].split(" ")[0] for i in range(len(self.raw_df))]
        elif config == 'id':
            self.raw_df['Year'] = [num.split("-")[0] for num in self.raw_df[self.id_col]]
            
    def calc_frequency(self, year_col=''): #something is wrong with the frequency
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
            year_ids = self.raw_df.loc[self.raw_df[self.id_col].isin(ids)][year_col].tolist()#[num.split("-")[0] for num in ids]
            for year in years_frequency:
                years_frequency[year].append(year_ids.count(year))
        for year in years_frequency:
            self.grouped_df[year+" Frequency"] = years_frequency[year]
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
        #severity func calculates the severity for each report in the raw df
        if from_file == False:
            self.raw_df = severity_func(self.raw_df)
            #need to get average severity for row in grouped df
            severity_per_row = []
            for i in range(len(self.grouped_df)):
                ids = self.grouped_df.iloc[i][self.id_col].split("; ")
                temp_df = self.raw_df.loc[self.raw_df[self.id_col ].isin(ids)]
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
        self.grouped_df['risk'] = self.grouped_df['severity'] * self.grouped_df['frequency']
        return
    
    def build_fmea(self, severity_func):
        self.get_entities_per_doc()
        self.group_docs()
        self.calc_frequency()
        self.calc_severity(severity_func)
        self.calc_risk()
        #prune excess columns
        #add mission phase
        #save table
        return
    
    def display_doc(self, doc_id, save=True, output_path="", colors_path=None, pred=True):
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
                output_path = os.path.join(os.getcwd(),"results", str(doc_id)+"_display")
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