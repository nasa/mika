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
from tensorflow.keras import models
from sklearn.metrics import hamming_loss

class In_Time_Risk_Matrix():
    def __init__(self, num_severity_models=1, num_likelihood_models=1, hazards=[]):
        self.severity_model_count = num_severity_models
        self.likelihood_model_count = num_likelihood_models
        self.hazards = hazards
        self.NN = False
    def load_model(self,model_location=[], model_type=None, model_inputs=[], NN=False):
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
            if not NN :
                for i in range(len(model_location)):
                    setattr(self, str(model_type)+'_model_'+str(i), pickle.load(open(model_location[i], 'rb')))
                    setattr(self, str(model_type)+'_model_inputs_'+str(i), model_inputs[i])
            else:
                self.NN = True
                for i in range(len(model_location)):
                    if ".sav" in model_location[i]:
                        setattr(self, str(model_type)+'_model_'+str(i), pickle.load(open(model_location[i], 'rb')))
                        setattr(self, str(model_type)+'_model_inputs_'+str(i), model_inputs[i])
                    else:
                        setattr(self, str(model_type)+'_model_'+str(i), models.load_model(model_location[i], custom_objects={'Hamming_loss':hamming_loss}))
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
            vect_df =  vectorizer.transform(input_df[target]).todense()
            new_cols = vectorizer.get_feature_names()
            vect_df = pd.DataFrame(vect_df, columns=new_cols)
            setattr(self, 'likelihood_model_inputs_'+str(nlp_model_number), new_cols)
            
        elif nlp_model_type == 'word2vec':
            #saves bigram and trigram dectector and tokenizer
            bigrams_detector = getattr(self, str(model_type)+'_nlp_model_bigram_'+str(nlp_model_number))
            trigrams_detector = getattr(self, str(model_type)+'_nlp_model_trigram_'+str(nlp_model_number))
            tokenizer = getattr(self, str(model_type)+'_nlp_model_tokenizer_'+str(nlp_model_number))
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
        elif nlp_model_type == 'sBERT':
            vectorizer = getattr(self, str(model_type)+'_nlp_model_'+str(nlp_model_number))
            vect_df =  vectorizer.encode(input_df[target])
            self.bert=True
    
        return vect_df
    
    def predict_likelihoods(self, report_df):
        for i in range(self.likelihood_model_count):
            mdl = getattr(self, 'likelihood_model_'+str(i))
            if i == 0:
                if self.NN or self.bert:
                    probs = mdl.predict(report_df[getattr(self, 'likelihood_model_inputs_'+str(i))].values[0].reshape(1,-1))
                else: 
                    probs = mdl.predict_proba(report_df[getattr(self, 'likelihood_model_inputs_'+str(i))].values.reshape(1,-1))
                probs_df = pd.Series(probs[0], index=self.hazards)
            if i>0:
                input_df = pd.concat([report_df[getattr(self, 'likelihood_model_inputs_'+str(i))], probs_df])
                probs = mdl.predict_proba(input_df.values.reshape(1,-1))
                probs_df = pd.DataFrame(probs, columns=self.hazards)
        return probs_df
    
    def predict_severity(self, report_df):
        #for each hazard, vary its occurrence
        self.hazards = ['Traffic','Command_Transitions', 'Evacuations', 'Inaccurate_Mapping',
                   'Aerial_Grounding', 'Resource_Issues', 'Injuries', 'Cultural_Resources',
                   'Livestock', 'Law_Violations', 'Military_Base', 'Infrastructure',
                   'Extreme_Weather', 'Ecological', 'Hazardous_Terrain', 'Floods','Dry_Weather']
        severities = {hazard:[] for hazard in self.hazards}
        for hazard in self.hazards:
            temp_input = report_df
            temp_input.at[hazard] = 1
            for other_hazard in self.hazards:
                if other_hazard != hazard:
                    temp_input.at[other_hazard] = 0
            severities[hazard] = np.around(self.severity_model_0.predict(temp_input[self.severity_model_inputs_0].values.reshape(1,-1))[0])
        preds_df = pd.DataFrame(severities, 
                            index=['Diff_Injuries', 'Diff_Structures_Damages', 
                                     'Diff_Structures_Destroyed','Diff_Fatalities'])
        return preds_df
    
    def get_likelihoods(self, report_df):
        self.curr_likelihoods = {hazard:0 for hazard in self.hazards}
        probs_df = self.predict_likelihoods(report_df)
        for hazard in self.hazards:
            p = probs_df.at[0, hazard]
            if p>=0.5:
                likelihood = 'Highly Likely'
            elif p>=0.05 and p<0.5:
                likelihood = 'Likely'
            elif p>=0.005 and p<0.05:
                likelihood = 'Possible'
            elif p>=0.0005 and p<0.005:
                likelihood = 'Improbable'
            elif p<0.0005:
                likelihood = 'Extremely Improbable'
            self.curr_likelihoods[hazard] = likelihood
        
    def get_severities(self, report_df):
        self.curr_severities = {}
        preds_df = self.predict_severity(report_df)
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
        
    def build_risk_matrix(self, input_reports, clean=False, vectorize=False, condense_text=True, target=None, model_type=None, nlp_model_type=None, nlp_model_number=None,figsize=(9,5)):
        plt.rcParams["font.family"] = "Times New Roman"
        if clean: 
            input_reports = self.prepare_input_data(input_reports, target)
        if vectorize:
            vect_df = self.vectorize_inputs(input_reports, target, model_type, nlp_model_type, nlp_model_number)
            if nlp_model_type == "tfidf":
                input_reports = pd.concat([input_reports, vect_df], axis=1)
            elif nlp_model_type == "word2vec" or nlp_model_type == 'sBERT':
                input_reports[target] = [np.asarray(a).astype('float32') for a in vect_df]
            
        if condense_text: 
            condense_dict = {self.hazards[i]:"H"+str(i+1) for i in range(len(self.hazards))}
            
        for i in range(len(input_reports)):
            report = input_reports.iloc[i][:]
            
            annotation_df = pd.DataFrame([["" for i in range(5)] for j in range(5)],
                             columns=['Minimal Impact', 'Minor Impact', 'Major Impact', 'Significant Critical Impact', 'Catastrophic Impact'],
                              index=['Highly Likely', 'Likely', 'Possible','Improbable', 'Extremely Improbable'])
            self.get_likelihoods(report)
            self.get_severities(report)
            hazards_per_row_df = pd.DataFrame([[0 for i in range(5)] for j in range(5)],
                             columns=['Minimal Impact', 'Minor Impact', 'Major Impact', 'Significant Critical Impact', 'Catastrophic Impact'],
                              index=['Highly Likely', 'Likely', 'Possible','Improbable', 'Extremely Improbable'])
            for hazard in self.hazards:
                if annotation_df.at[self.curr_likelihoods[hazard],self.curr_severities[hazard]]!= "":
                    annotation_df.at[self.curr_likelihoods[hazard],self.curr_severities[hazard]] += ", "
                    hazards_per_row_df.at[self.curr_likelihoods[hazard],self.curr_severities[hazard]]+=1
                    if hazards_per_row_df.at[self.curr_likelihoods[hazard],self.curr_severities[hazard]]>2:
                        annotation_df.at[self.curr_likelihoods[hazard],self.curr_severities[hazard]]+="\n"
                        hazards_per_row_df.at[self.curr_likelihoods[hazard],self.curr_severities[hazard]]=0
                if condense_text: hazard_annot = condense_dict[hazard]
                else: hazard_annot = hazard
                annotation_df.at[self.curr_likelihoods[hazard],self.curr_severities[hazard]] += (str(hazard_annot))
                
            df = pd.DataFrame([[0, 5, 10, 10, 10], [0, 5, 5, 10, 10], [0, 0, 5, 5, 10],
                    [0, 0, 0, 5, 5], [0, 0, 0, 0, 5]],
                  columns=['Minimal Impact', 'Minor Impact', 'Major Impact', 'Significant Critical Impact', 'Catastrophic Impact'],
                  index=['Highly Likely', 'Likely', 'Possible','Improbable', 'Extremely Improbable']
                  )
            fig,ax = plt.subplots(figsize=figsize)
            #annot df has hazards in the cell they belong to #annot=annotation_df
            sn.heatmap(df, annot=annotation_df, fmt='s',annot_kws={'fontsize':16},cbar=False,cmap='RdYlGn_r')
            plt.title("Risk Matrix", fontsize=16)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)#, ha="right")
            plt.tick_params(labelsize=16)
            plt.ylabel("Likelihood", fontsize=16)
            plt.xlabel("Severity", fontsize=16)
            minor_ticks = np.arange(1, 6, 1)
            ax.set_xticks(minor_ticks, minor=True)
            ax.set_yticks(minor_ticks, minor=True)
            ax.tick_params(which='minor',length=0, grid_color='black', grid_alpha=1)
            ax.grid(which='minor', alpha=1)
            plt.show()
    
            