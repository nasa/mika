# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 14:30:33 2021

@author: srandrad
"""
import pickle
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
import pandas as pd
import seaborn as sn
import numpy as np
from tensorflow.keras import preprocessing as kprocessing

class In_Time_Risk_Matrix():
    def __init__(self, num_severity_models=1, num_likelihood_models=1, hazards=[]):
        self.severity_model_count = num_severity_models
        self.likelihood_model_count = num_likelihood_models
        self.hazards = hazards
    def load_model(self,model_location=[], model_type=None, model_inputs=[]):
        """
        Loads models and saves as object attributes

        Parameters
        ----------
        model_location : list of strings, optional
            list of model locations. The default is [].
        model_type : string, optional
            likelihood or severity. The default is None.
        model_inputs : list of lists, optional
            list of lists of model predictors. The default is [].
            
        Returns
        -------
        None.

        """
        if model_location and model_type:
            for i in range(len(model_location)):
                setattr(self, str(model_type)+'_model_'+str(i), pickle.load(open(model_location[i], 'rb')))
                setattr(self, str(model_type)+'_model_inputs_'+str(i), model_inputs[i])
    
    def load_nlp_model(self, model_location=[], model_type=None, model_number=[], model_input=[]):
        """
        for tfidf -> only need to load the tfidf model
        for word2vec -> need to load bigrams detector, trigrams detector, and tokenizer
        
        saves nlp models as object attributes 
        
        Parameters
        ----------
        model_location : List of strings, optional
            list of locations of models. The default is [].
        model_type : string, optional
            likelihood or severity model. The default is None.
        model_number : List, optional
            the model number for tfidf, model type (bigram/trigram/tokenizer) for word2vec.
            The default is [].

        Returns
        -------
        None.

        """
        self.nlp_type = model_type
        if model_location and model_type:
            for i in range(len(model_location)):
                setattr(self, str(model_type)+'_nlp_model_'+str(model_number[i]), pickle.load(open(model_location[i], 'rb')))
                setattr(self, str(model_type)+'_nlp_model_input_'+str(model_number[i]), model_input[i])
    def prepare_input_data(self, df, target):
        cleaned_combined_text = []
        for text in df[target]:
            cleaned_text = self.remove_quote_marks(text)
            cleaned_combined_text.append(cleaned_text)
        df[target] = cleaned_combined_text
        return df
    
    def remove_quote_marks(self, word_list):
        word_list = word_list.strip("[]").split(", ")
        word_list = [w.replace("'","") for w in word_list]
        word_list = " ".join(word_list)
        return word_list
    
    def vectorize_inputs(self, input_df, target, model_type, nlp_model_type, nlp_model_number):
        if nlp_model_type == "tfidf":
            vectorizer = getattr(self, str(model_type)+'_nlp_model_'+str(nlp_model_number))
            vect_df =  vectorizer.transform(input_df[target])
        elif nlp_model_type == 'word2vec':
            #saves bigram and trigram dectector and tokenizer
            bigrams_detector = getattr(self, str(model_type)+'_nlp_model_bigram'+str(nlp_model_number))
            trigrams_detector = getattr(self, str(model_type)+'_nlp_model_trigram'+str(nlp_model_number))
            tokenizer = getattr(self, str(model_type)+'_nlp_model_tokenizer'+str(nlp_model_number))
            ## create list of n-grams
            lst_corpus = input_df[target]
            ## detect common bigrams and trigrams using the fitted detectors
            lst_corpus = list(bigrams_detector[lst_corpus])
            lst_corpus = list(trigrams_detector[lst_corpus])
            ## text to sequence with the fitted tokenizer
            lst_text2seq = tokenizer.texts_to_sequences(lst_corpus)
            ## padding sequence
            vect_df = kprocessing.sequence.pad_sequences(lst_text2seq, maxlen=30,
                         padding="post", truncating="post")
        return vect_df
    
    def predict_likelihoods(self, report_df):
        #complicated - depends on the models used
        self.vectorize_input( )#TODO: implement this properly -> where does this go in the work flow?? -> in likelihood predictions
        return 
    def predict_severity(self, report_df):
        #for each hazard, vary its occurrence
        self.hazards = ['Traffic','Command_Transitions', 'Evacuations', 'Inaccurate_Mapping',
                   'Aerial_Grounding', 'Resource_Issues', 'Injuries', 'Cultural_Resources',
                   'Livestock', 'Law_Violations', 'Military_Base', 'Infrastructure',
                   'Extreme_Weather', 'Ecological', 'Hazardous_Terrain', 'Floods','Dry_Weather']
        severities = {hazard:[] for hazard in self.hazards}
        #for i in input_df:
        for hazard in self.hazards:
            temp_input = report_df
            temp_input.at[0,hazard] = 1
            for other_hazard in self.hazards:
                if other_hazard != hazard:
                    temp_input.at[0,other_hazard] = 0
            severities[hazard] = self.severity_model_0.predict(temp_input[self.severity_model_inputs_0])
            preds_df = pd.DataFrame(severities, 
                                index=['Diff_Injuries', 'Diff_Structures_Damages', 
                                         'Diff_Structures_Destroyed','Diff_Fatalities'])
        return preds_df
    def get_likelihoods(self):
        self.curr_likelihoods = {} #hazard: (likelihood, severity)
        self.predict_likelihoods()
        
    def get_severities(self):
        self.curr_severities = {}
        preds_df = self.predict_severity()
        for hazard in preds_df.columns:
             injuries = preds_df.at['Diff_Injuries',hazard]
             str_dam = preds_df.at['Diff_Structures_Damages', hazard]
             str_des = preds_df.at['Diff_Structures_Destroyed', hazard]
             fatalities = preds_df.at['Diff_Fatalities', hazard]
             if injuries == 0 and fatalities == 0 and str_des == 0 and str_dam ==0:
                 impact = "Minimal Impact"
             elif injuries <= 2 and fatalities == 0 and str_des == 0 and str_dam <= 10:
                 impact = "Minor Impact"
             elif injuries <= 2 and fatalities == 0 and str_des <= 10 and str_dam <= 10:
                 impact = "Major Impact"
             else:
                if fatalities<2:
                    impact = 'Significant Critical Impact'
                else: #fatalities>2
                    impact = 'Catastrophic Impact'
             self.curr_severities[hazard] = impact
        
    def build_risk_matrix(self, input_reports, clean=False):
        plt.rcParams["font.family"] = "Times New Roman"
        if clean: 
            input_reports = self.prepare_input_data(input_reports)
        
        for report in input_reports:
            annotation_df = pd.DataFrame([[[] for i in range(5)] for j in range(5)],
                             columns=['Minimal Impact', 'Minor Impact', 'Major Impact', 'Significant Critical Impact', 'Catastrophic Impact'],
                              index=['Highly Likely', 'Likely', 'Possible','Improbable', 'Extremely Improbable'])
            
            self.get_likelihoods(report)
            self.get_severities(report)
            for hazard in self.hazards:
                annotation_df.at[self.curr_likelihoods[hazard],self.curr_severities[hazard]].append(hazard)
                
            df = pd.DataFrame([[0, 5, 10, 10, 10], [0, 5, 5, 10, 10], [0, 0, 5, 5, 10],
                    [0, 0, 0, 5, 5], [0, 0, 0, 0, 5]],
                  columns=['Minimal Impact', 'Minor Impact', 'Major Impact', 'Significant Critical Impact', 'Catastrophic Impact'],
                  index=['Highly Likely', 'Likely', 'Possible','Improbable', 'Extremely Improbable']
                  )
            fig,ax = plt.subplots(figsize=(10,8))
            #annot df has hazards in the cell they belong to #annot=annotation_df
            sn.heatmap(df, annot=annotation_df, annot_kws={'fontsize':16},cbar=False,cmap='RdYlGn_r')
            plt.title("Risk Matrix", fontsize=16)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)#, ha="right")
            plt.tick_params(labelsize=16)
            plt.ylabel("Likelihood", fontsize=16)
            plt.xlabel("Severity", fontsize=16)
            minor_ticks = np.arange(1, 6, 1)
            ax.set_xticks(minor_ticks, minor=True)#, tick_params={'length':0})
            ax.set_yticks(minor_ticks, minor=True)#, length=0)
            ax.tick_params(which='minor',length=0, grid_color='black', grid_alpha=1)
            ax.grid(which='minor', alpha=1)
            plt.show()
    
            