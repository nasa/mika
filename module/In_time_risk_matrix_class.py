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
from tqdm import tqdm
from tensorflow.keras import preprocessing as kprocessing
from tensorflow.keras import models
from sklearn.metrics import hamming_loss
import tensorflow_addons as tfa
Hamming_loss=tfa.metrics.HammingLoss(mode='multilabel', threshold=0.6)
import os

#TODO: remove old sbert and word2vec functionality
class In_Time_Risk_Matrix():
    def __init__(self, num_severity_models=1, num_likelihood_models=1, hazards=[]):
        self.severity_model_count = num_severity_models
        self.likelihood_model_count = num_likelihood_models
        self.hazards = hazards
        self.NN = False
        self.bert = False
        self.static_rm = None
    def load_model(self,model_location=[], model_type=None, model_inputs=[], NN=False):
        """
        Loads models and saves as object attributes

        Parameters
        ----------
        model_location : list of strings, optional
            list of model locations. The default is [].
        model_type : string, optional
            likelihood or severity. The default is None.
        model_inputs : list of lists/dict, optional
            list of lists.dict of model predictors. The default is [].
        NN : boolean, optional
            True if using a neural network, false if not
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
                        setattr(self, str(model_type)+'_model_'+str(i), models.load_model(model_location[i], custom_objects={'Hamming_loss':Hamming_loss}, compile=False))
                        
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
        model_input: List, optional
            The model inputs for the nlp model (preprocessed or raw usually)
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
        """
        Prepares text input data by cleaning

        Parameters
        ----------
        df : pandas DataFrame
            dataframe containing the input data
        target : string
            column in df where the text target is
        Returns
        -------
        df : pandas DataFrame
            Cleaned version of input df
        """
        cleaned_combined_text = []
        for text in df[target]:
            cleaned_text = self.remove_quote_marks(text)
            cleaned_combined_text.append(cleaned_text)
        df[target] = cleaned_combined_text
        return df
    
    def remove_quote_marks(self, word_list):
        """
        Removes extra quotation marks and brackets from raw text.
        
        Parameters
        ----------
        word_list : string
            A list of words in string format. 
            ex: "["this", "is", "the", "example"]"
        Returns
        -------
        word_list : string
           A list of words in string format that reads like a sentence
           ex: "this is the example"
        """
        if type(word_list) is str:
            word_list = word_list.strip("[]").split(", ")
        word_list = [w.replace("'","") for w in word_list]
        word_list = " ".join(word_list)
        return word_list
    
    def vectorize_inputs(self, input_df, target, model_type, nlp_model_type, nlp_model_number):
        """
        Vectorizes text inputs according to vectorization method chosen

        Parameters
        ----------
        input_df : pandas DataFrame
            Dataframe containing the input data
        target : string
            Column in df where the text target is
        model_type : string
            Type of machine learning model this is an input for. Currently only uses "likelihood"
        nlp_model_type : string
            Vectorization method. Currently supports "tfidf", "word2vec", or "sBERT".
        nlp_model_number : int
            The number in the pipeline for the model. Most cases this is 0.

        Returns
        -------
        vect_df : pandas DataFrame
            Dataframe containing the vectorized input data

        """
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
            
        self.nlp_model_type = nlp_model_type
        return vect_df
    
    def predict_likelihoods(self, report_df):
        """
        Predicts the likelihood of hazard occurent for an individual report using
        a pipeline of models (i.e. text-based and meta data based classification)

        Parameters
        ----------
        report_df : pandas DataFrame (single row)
            a single report (row) of a pandas dataframe

        Returns
        -------
        probs_df : pandas DataFrame
            a single row of a dataframe where columns are hazards and the row contains the probability of occurence

        """
        for i in range(self.likelihood_model_count):
            mdl = getattr(self, 'likelihood_model_'+str(i))
            if self.NN: mdl.compile()
            if i == 0:
                if self.NN:
                    x = {input_key: report_df[getattr(self, 'likelihood_model_inputs_'+str(i))[input_key]] for input_key in getattr(self, 'likelihood_model_inputs_'+str(i))}
                    for key in x:
                        if type(x[key].iloc[0][0]) is not str:
                            x[key] = np.array(x[key]).astype('float32')
                    probs = mdl.predict(x)
                elif self.bert:
                    probs = mdl.predict(report_df[getattr(self, 'likelihood_model_inputs_'+str(i))].values[0].reshape(1,-1))
                else: 
                    probs = mdl.predict_proba(report_df[getattr(self, 'likelihood_model_inputs_'+str(i))].values)
                probs_df = pd.DataFrame(probs, columns=self.hazards)
            if i>0:
                input_df = pd.concat([report_df[getattr(self, 'likelihood_model_inputs_'+str(i))], probs_df])
                probs = mdl.predict_proba(input_df.values.reshape(1,-1))
                probs_df = pd.DataFrame(probs, columns=self.hazards)
        return probs_df
    
    def predict_severity(self, reports):
        """
        predicts the severity for a set of reports

        Parameters
        ----------
        reports : pd.DataFrame
            ICS-209-PLUS situation reports
        Returns
        -------
        preds_dfs : list of pd.DataFrame
            list where each element is a pd.DataFrame corresponding to an incident report.
            The data frame has columns as hazards and rows as different severity measures

        """
        #transform df
        indices = []
        new_df = reports.copy()
        dfs = []
        for i in range(len(reports)):
            report_df = pd.DataFrame(columns=new_df.columns)
            for hazard in self.hazards: #for each hazard, we set the input value at the hazard=1
                indices.append((i,hazard))
                temp_input = new_df.loc[i][:]#.copy()
                temp_input.at[hazard] = 1
               
                for other_hazard in self.hazards:
                    if other_hazard != hazard:
                        temp_input.at[other_hazard] = 0 #and set the input value at all other hazards=0 -> thus we assume only the selected hazard is occuring
                
                report_df = report_df.append(temp_input, ignore_index=True)
            dfs.append(report_df)
            
        new_reports = pd.concat(dfs, ignore_index=True)
        new_reports.index = pd.MultiIndex.from_tuples(indices, names=["index", "hazard"])
        preds = np.around(self.severity_model_0.predict(new_reports[self.severity_model_inputs_0].astype(float)))
        preds_df = pd.DataFrame(preds, index=pd.MultiIndex.from_tuples(indices, names=["index", "hazard"]))
        preds_dfs = []
        for i in range(len(reports)):
            severities = {}
            for hazard in self.hazards:
                severities[hazard] = preds_df.loc[(i,hazard)][:].values
            preds_dfs.append(pd.DataFrame(severities, 
                                index=['Diff_Injuries', 'Diff_Structures_Damages', 
                                         'Diff_Structures_Destroyed','Diff_Fatalities']))
        return preds_dfs
                                

    
    def get_likelihoods(self, reports, likelihood_df=None):
        """
        Converts raw probabilities into likelihood categories for a risk matrix.
        Saves these likelihoods as a member variable for use in rm construction.

        Parameters
        ----------
        reports : pd.DataFrame
            ICS-209-PLUS situation reports
        likelihood_df :  pandas DataFrame, optional
           a dataframe where columns are hazards and the rows contains the probability of occurence. 
           The default is None.

        Returns
        -------
        None.

        """
        self.curr_likelihoods = []
        if reports is not None:
            probs_df = self.predict_likelihoods(reports)
        elif likelihood_df is not None:
            probs_df = likelihood_df
        else:
            print("Error: must input a report or a dataframe of hazard likelihoods")
            return
        for i in range(len(probs_df)):
            curr_likelihoods = {hazard:0 for hazard in self.hazards}
            for hazard in self.hazards:
                p = probs_df.at[i, hazard]
                if p>=0.8:
                    likelihood = 'Highly Likely'
                elif p>=0.3 and p<0.8:
                    likelihood = 'Likely'
                elif p>=0.05 and p<0.3:
                    likelihood = 'Possible'
                elif p>=0.005 and p<0.05:
                    likelihood = 'Improbable'
                elif p<0.005:
                    likelihood = 'Extremely Improbable'
                curr_likelihoods[hazard] = likelihood
            self.curr_likelihoods.append(curr_likelihoods)
        
    def get_severities(self, reports, severity_df=None):
        """
        Converts raws severity predictions into severity categories for a risk matrix.
        Saves these severities as a member variable for use in rm construction.

        Parameters
        ----------
        reports : pd.DataFrame
            ICS-209-PLUS situation reports
        severity_df : pandas DataFrame, optional
            a dataframe where columns are hazards rows are the predicted severity for each measure (injuries, fatalities, etc)
            The default is None.

        Returns
        -------
        None.

        """
        self.curr_severities = []
        if reports is not None:
            preds_dfs = self.predict_severity(reports)
        elif severity_df is not None:
            preds_dfs = [severity_df]
        else:
            print("Error: must input a report or a dataframe of hazard severities")
            return
        for preds_df in preds_dfs:
            curr_severities = {}
            for hazard in self.hazards: 
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
                 curr_severities[hazard] = impact
            self.curr_severities.append(curr_severities) 
        
    def build_risk_matrix(self, input_reports, clean=False, vectorize=False, condense_text=True, target=None, model_type=None, nlp_model_type=None, nlp_model_number=None,figsize=(9,5), show=True, save=False, id_col=None):
        """
        Builds a dynamic risk matrix for each report in the input_reports.

        Parameters
        ----------
        input_reports :  pandas DataFrame
            a dataframe containing situation reports
        clean : Boolean, optional
            dictates whether text data is cleaned or not. The default is False.
        vectorize : Boolean,, optional
            dictates whether text data is vectorized or not. The default is False.
        condense_text : Boolean, optional
            dictates whether full hazard names are shown or not. If true, hazards are numbered. 
            The default is True.
        target : String, optional
            Column in input_reports where the text target is. The default is None.
        model_type : string
            Type of machine learning model this is an input for. Currently only uses "likelihood"
        nlp_model_type : string
            Vectorization method. Currently supports "tfidf", "word2vec", or "sBERT".
        nlp_model_number : int
            The number in the pipeline for the model. Most cases this is 0.
        figsize : Tuple, optional
            figsize for rm display. The default is (9,5).
        show : Boolean, optional
            dictates whether or not figure is shown. The default is True.
        save: Boolean, optional
            dictates whether or not the figure is saved as a pdf. The default is False.
        id_col: String, optional
            name of the column where unique identifiers are stored. used for created file name for saved images.
        Returns
        -------
        annotation_dfs : list
            list of pandas dataframes, where each dataframe holds the annotations (i.e. hazard placements on rm)
            for each incident report.

        """
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
        self.get_likelihoods(input_reports)
        self.get_severities(input_reports)
        annotation_dfs = []; hazard_likelihoods = {hazard:[] for hazard in self.hazards}; hazard_severities = {hazard:[] for hazard in self.hazards}
        for i in tqdm(range(len(input_reports)), desc="Building Dynamic Risk Matrices..."):
            #report = input_reports.iloc[i][:]
            
            annotation_df = pd.DataFrame([["" for i in range(5)] for j in range(5)],
                             columns=['Minimal Impact', 'Minor Impact', 'Major Impact', 'Significant Critical Impact', 'Catastrophic Impact'],
                              index=['Highly Likely', 'Likely', 'Possible','Improbable', 'Extremely Improbable'])
            #self.get_likelihoods(report)
            #self.get_severities(report)
            hazards_per_row_df = pd.DataFrame([[0 for i in range(5)] for j in range(5)],
                             columns=['Minimal Impact', 'Minor Impact', 'Major Impact', 'Significant Critical Impact', 'Catastrophic Impact'],
                              index=['Highly Likely', 'Likely', 'Possible','Improbable', 'Extremely Improbable'])
            rows = pd.DataFrame([[0 for i in range(5)] for j in range(5)],
                             columns=['Minimal Impact', 'Minor Impact', 'Major Impact', 'Significant Critical Impact', 'Catastrophic Impact'],
                              index=['Highly Likely', 'Likely', 'Possible','Improbable', 'Extremely Improbable'])
            annot_font = 16
            for hazard in self.hazards:
                hazard_likelihoods[hazard].append(self.curr_likelihoods[i][hazard])
                hazard_severities[hazard].append(self.curr_severities[i][hazard])
                if annotation_df.at[self.curr_likelihoods[i][hazard],self.curr_severities[i][hazard]]!= "":
                    annotation_df.at[self.curr_likelihoods[i][hazard],self.curr_severities[i][hazard]] += ", "
                    hazards_per_row_df.at[self.curr_likelihoods[i][hazard],self.curr_severities[i][hazard]]+=1
                    if hazards_per_row_df.at[self.curr_likelihoods[i][hazard],self.curr_severities[i][hazard]]>2:
                        annotation_df.at[self.curr_likelihoods[i][hazard],self.curr_severities[i][hazard]]+="\n"
                        rows.at[self.curr_likelihoods[i][hazard],self.curr_severities[i][hazard]] += 1
                        if rows.at[self.curr_likelihoods[i][hazard],self.curr_severities[i][hazard]] > 2: annot_font=12
                        if rows.at[self.curr_likelihoods[i][hazard],self.curr_severities[i][hazard]] > 3: annot_font=10
                        if rows.at[self.curr_likelihoods[i][hazard],self.curr_severities[i][hazard]] > 4: annot_font=8
                        hazards_per_row_df.at[self.curr_likelihoods[i][hazard],self.curr_severities[i][hazard]]=0
                if condense_text: hazard_annot = condense_dict[hazard]
                else: hazard_annot = hazard
                annotation_df.at[self.curr_likelihoods[i][hazard],self.curr_severities[i][hazard]] += (str(hazard_annot))
            annotation_dfs.append(annotation_df)
            if show:
                df = pd.DataFrame([[0, 5, 10, 10, 10], [0, 5, 5, 10, 10], [0, 0, 5, 5, 10],
                        [0, 0, 0, 5, 5], [0, 0, 0, 0, 5]],
                      columns=['Minimal Impact', 'Minor Impact', 'Major Impact', 'Significant Critical Impact', 'Catastrophic Impact'],
                      index=['Highly Likely', 'Likely', 'Possible','Improbable', 'Extremely Improbable']
                      )
                fig,ax = plt.subplots(figsize=figsize)
                #annot df has hazards in the cell they belong to #annot=annotation_df
                sn.heatmap(df, annot=annotation_df, fmt='s',annot_kws={'fontsize':annot_font},cbar=False,cmap='RdYlGn_r')
                plt.title("Dynamic Risk Matrix", fontsize=16)
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
                if save: plt.savefig(os.path.join(os.path.dirname(os.getcwd()),'smart_nlp','results','risk_matrices', str(input_reports.at[i,id_col])+".pdf"), bbox_inches="tight")
                plt.show()
        if not show: 
            return annotation_dfs, hazard_likelihoods, hazard_severities
        
    def build_static_risk_matrix(self,  severity_df, likelihood_df, condense_text=True, figsize=(9,5), rates=False, show=True, save=False):
        """
        Builds a static risk matrix from known hazard likelihoods and severities

        Parameters
        ----------
        severity_df : pandas DataFrame 
            dataframe where columns are hazards, and rows are severity values for each measure (injuries, fatalities, etc)
        likelihood_df : pandas DataFrame 
            single row dataframe where columns are hazards, row contains probabilties or rates
        condense_text : Boolean, optional
            dictates whether full hazard names are shown or not. If true, hazards are numbered. 
            The default is True.
        figsize : Tuple, optional
           figsize for rm display. The default is (9,5).
        rates : Boolean, optional
            dictates whether the likelihood is based on rates or probabilities. The default is False.
        show : Boolean, optional
            dictates whether or not figure is shown. The default is True.
        save: Boolean, optional
            dictates whether or not the figure is saved as a pdf. The default is False.
        Returns
        -------
        None.

        """
        annotation_df = pd.DataFrame([["" for i in range(5)] for j in range(5)],
                             columns=['Minimal Impact', 'Minor Impact', 'Major Impact', 'Significant Critical Impact', 'Catastrophic Impact'],
                              index=['Highly Likely', 'Likely', 'Possible','Improbable', 'Extremely Improbable'])
        if rates == True: 
            self.get_static_likelihoods_from_rates(likelihood_df)
        else: 
            self.get_likelihoods(reports=None, likelihood_df=likelihood_df)
            self.curr_likelihoods=self.curr_likelihoods[0]
        self.get_severities(reports=None, severity_df=severity_df); self.curr_severities=self.curr_severities[0]
        hazards_per_row_df = pd.DataFrame([[0 for i in range(5)] for j in range(5)],
                         columns=['Minimal Impact', 'Minor Impact', 'Major Impact', 'Significant Critical Impact', 'Catastrophic Impact'],
                          index=['Highly Likely', 'Likely', 'Possible','Improbable', 'Extremely Improbable'])
        rows = pd.DataFrame([[0 for i in range(5)] for j in range(5)],
                         columns=['Minimal Impact', 'Minor Impact', 'Major Impact', 'Significant Critical Impact', 'Catastrophic Impact'],
                          index=['Highly Likely', 'Likely', 'Possible','Improbable', 'Extremely Improbable'])
        annot_font = 16
        hazard_likelihoods = {hazard:"" for hazard in self.hazards}; hazard_severities={hazard:"" for hazard in self.hazards}
        for hazard in self.hazards:
            hazard_likelihoods[hazard] = self.curr_likelihoods[hazard]
            hazard_severities[hazard] = self.curr_severities[hazard]
            if annotation_df.at[self.curr_likelihoods[hazard],self.curr_severities[hazard]]!= "":
                annotation_df.at[self.curr_likelihoods[hazard],self.curr_severities[hazard]] += ", "
                hazards_per_row_df.at[self.curr_likelihoods[hazard],self.curr_severities[hazard]]+=1
                if hazards_per_row_df.at[self.curr_likelihoods[hazard],self.curr_severities[hazard]]>2:
                    annotation_df.at[self.curr_likelihoods[hazard],self.curr_severities[hazard]]+="\n"
                    rows.at[self.curr_likelihoods[hazard],self.curr_severities[hazard]] += 1
                    if rows.at[self.curr_likelihoods[hazard],self.curr_severities[hazard]] > 3: annot_font=12
                    hazards_per_row_df.at[self.curr_likelihoods[hazard],self.curr_severities[hazard]]=0
            if condense_text: 
                condense_dict = {self.hazards[i]:"H"+str(i+1) for i in range(len(self.hazards))}
                hazard_annot = condense_dict[hazard]
            else: hazard_annot = hazard
            annotation_df.at[self.curr_likelihoods[hazard],self.curr_severities[hazard]] += (str(hazard_annot))
        self.static_rm = annotation_df; self.static_likelihoods = hazard_likelihoods; self.static_severities = hazard_severities
        if show == False: return annotation_df, hazard_likelihoods, hazard_severities
        
        df = pd.DataFrame([[0, 5, 10, 10, 10], [0, 5, 5, 10, 10], [0, 0, 5, 5, 10],
                [0, 0, 0, 5, 5], [0, 0, 0, 0, 5]],
              columns=['Minimal Impact', 'Minor Impact', 'Major Impact', 'Significant Critical Impact', 'Catastrophic Impact'],
              index=['Highly Likely', 'Likely', 'Possible','Improbable', 'Extremely Improbable']
              )
        fig,ax = plt.subplots(figsize=figsize)
        #annot df has hazards in the cell they belong to #annot=annotation_df
        sn.heatmap(df, annot=annotation_df, fmt='s',annot_kws={'fontsize':annot_font},cbar=False,cmap='RdYlGn_r')
        plt.title("Static Risk Matrix", fontsize=16)
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
        if save: 
            file_path = os.path.join(os.path.dirname(os.getcwd()),'smart_nlp','results','risk_matrices', "static_rm")
            if rates: file_path += "_rates"
            plt.savefig(file_path+".pdf", bbox_inches="tight")
        plt.show()
        
    def calc_static_likelihoods(self, frequency_fires, total_fires):
        """
        Calculates the static probability for each hazard

        Parameters
        ----------
        frequency_fires : dict
            Nested dictionary. Keys are hazards, value is inner dictionary. 
            Inner dictionary keys are years, values are a list of incident IDs.
            This is an output generated from trend analysis functions.
        total_fires : int
            The total number of fires in the historical dataset

        Returns
        -------
        probs_df : pandas DataFrame
            a single row of a dataframe where columns are hazards and the row contains the probability


        """
        probs = {hazard:(sum([frequency_fires[hazard.replace("_", " ")][year] for year in frequency_fires[hazard.replace("_", " ")]])/total_fires) for hazard in self.hazards}
        probs_df = pd.DataFrame(probs, columns=self.hazards, index=[0])
        return probs_df
    
    def calc_static_likelihoods_rates(self, frequency_fires):
        """
        Calculates the static rate of occurrence (in years) for each hazard

        Parameters
        ----------
        frequency_fires : dict
            Nested dictionary. Keys are hazards, value is inner dictionary. 
            Inner dictionary keys are years, values are a list of incident IDs.
            This is an output generated from trend analysis functions.

        Returns
        -------
        rates_df : pandas DataFrame
            a single row of a dataframe where columns are hazards and the row contains the rate (occurrence/1 year)

        """
        rates = {hazard:round(np.sum([frequency_fires[hazard.replace("_", " ")][year] for year in frequency_fires[hazard.replace("_", " ")]])/len(frequency_fires[hazard.replace("_", " ")])) for hazard in self.hazards}
        rates_df = pd.DataFrame(rates, columns=self.hazards, index=[0])
        return rates_df
    
    def get_static_likelihoods_from_rates(self, rate_df):
        """
        Converts raw rates into likelihood categories for a risk matrix.
        Saves these likelihoods as a member variable for use in rm construction.

        Parameters
        ----------
        rate_df : pandas DataFrame
            a single row of a dataframe where columns are hazards and the row contains the rate (occurrence/1 year)

        Returns
        -------
        None.

        """
        self.curr_likelihoods = {hazard:0 for hazard in self.hazards}
        for hazard in self.hazards:
            r = rate_df.at[0, hazard]
            if r>=100:
                likelihood = 'Highly Likely'
            elif r>=10 and r<100:
                likelihood = 'Likely'
            elif r>=1 and r<10:
                likelihood = 'Possible'
            elif r>=0.1 and r<1:
                likelihood = 'Improbable'
            elif r<0.1:
                likelihood = 'Extremely Improbable'
            self.curr_likelihoods[hazard] = likelihood

    def calc_static_severity(self, severity_table):
        """
        Calculates the static or averaage severity for each hazard

        Parameters
        ----------
        severity_table : pandas DataFrame
            Dataframe with rows as hazards and columns as average severity values.
            This is an output generated from trend analysis functions.

        Returns
        -------
        severity_df : pandas DataFrame 
            dataframe where columns are hazards, and rows are severity values for each measure (injuries, fatalities, etc)

        """
        severities = {hazard:[] for hazard in self.hazards}
        #severity table -> index
        for hazard in self.hazards:
            temp_df = severity_table.loc[severity_table["Hazard"]==hazard.replace("_", " ")].reset_index(drop=True)
            severities[hazard].append(temp_df.at[0,"Average Injuries"])
            severities[hazard].append(temp_df.at[0,"Average Structures Damaged"])
            severities[hazard].append(temp_df.at[0,"Average Structures Destroyed"])
            severities[hazard].append(temp_df.at[0,"Average Fatalities"])
        severity_df = pd.DataFrame(severities, 
                                index=['Diff_Injuries', 'Diff_Structures_Damages', 
                                         'Diff_Structures_Destroyed','Diff_Fatalities'])
        return severity_df
      
    def compare_results(self, data, frequency_fires=None, total_fires=None, severity_table=None, rate=True, dynamic_kwargs={}):
        """
        Compares the static risk matrix results to the dynamic risk matrix results. 
        Calculates the % of incident reports where static riks matrix is equal to the dynamic risk matrix.

        Parameters
        ----------
        data : pandas DataFrame
            dataframe containing all incident reports used to calculate the static risk matrix
        frequency_fires : dict, optional
            Nested dictionary. Keys are hazards, value is inner dictionary. 
            Inner dictionary keys are years, values are a list of incident IDs.
            This is an output generated from trend analysis functions.
            The default is None.
        total_fires : int, optional
            The total number of fires in the historical dataset. The default is None.
        severity_table : pandas DataFrame, optional
            Dataframe with rows as hazards and columns as average severity values.
            This is an output generated from trend analysis functions. The default is None.
        rate :Boolean, optional
            dictates whether the likelihood is based on rates or probabilities. The default is True.
        dynamic_kwargs : dict, optional
            dictionary of inputs for the dynamic risk matrix. The default is {}.
            required inputs are: 
                    clean : Boolean, optional
                        dictates whether text data is cleaned or not. The default is False.
                    vectorize : Boolean,, optional
                        dictates whether text data is vectorized or not. The default is False.
                    condense_text : Boolean, optional
                        dictates whether full hazard names are shown or not. If true, hazards are numbered. 
                        The default is True.
                    target : String, optional
                        Column in input_reports where the text target is. The default is None.
                    model_type : string
                        Type of machine learning model this is an input for. Currently only uses "likelihood"
                    nlp_model_type : string
                        Vectorization method. Currently supports "tfidf", "word2vec", or "sBERT".
                    nlp_model_number : int
                        The number in the pipeline for the model. Most cases this is 0.
        Returns
        -------
        percent_same: float
            the percent of incident reports where static_rm == dynamic_rm

        """
        # get static
        if (self.static_rm is None):
            if (frequency_fires is not None) and (severity_table is not None):
                if rate == True or total_fires is None:
                    probs_df = self.calc_static_likelihoods_rates(frequency_fires)
                else:
                    probs_df = self.calc_static_likelihoods(frequency_fires, total_fires=total_fires)
                severity_df = self.calc_static_severity(severity_table)
                static_df = self.build_static_risk_matrix(severity_df, probs_df, rates=rate, show=False)
            else:
                print("Error: not enough information provide to create static risk matrix.")
                print("Please input the required data.")
                return
        else:
            static_df = self.static_rm; static_likelihoods = self.static_likelihoods; static_severities = self.static_severities
        # for each report, get dynamic
        if dynamic_kwargs == {}:
            print("Error: the arguments for the dynamic risk matrix are missing.")
            print("Please input the required data.")
            return
        else:
            dynamic_dfs, dynamic_likelihoods, dynamic_severities = self.build_risk_matrix(data, show=False, **dynamic_kwargs)
        # calc % where static=dynamic
        percent_same = (len([df for df in dynamic_dfs if df.equals(static_df)])/len(dynamic_dfs))*100
        # calc % of likelihoods/ severities where static=dynamic
        comparison = {"Likelihood % same":[], "Likelihood average distance":[],
                      "Severity % same":[], "Severity average distance":[],
                      "Likelihood and Severity % same":[]}
        dist_dict= {'Highly Likely':1, 'Likely':2, 'Possible':3,'Improbable':4, 'Extremely Improbable':5,
                'Minimal Impact':1, 'Minor Impact':2, 'Major Impact':3, 'Significant Critical Impact':4, 'Catastrophic Impact':5}
        for hazard in self.hazards:
            comparison["Likelihood % same"].append((len([i for i in dynamic_likelihoods[hazard] if i==static_likelihoods[hazard]])/len(dynamic_likelihoods[hazard]))*100)
            comparison["Likelihood average distance"].append(np.average([dist_dict[i]-dist_dict[static_likelihoods[hazard]] for i in dynamic_likelihoods[hazard]]))
            comparison["Severity % same"].append((len([i for i in dynamic_severities[hazard] if i==static_severities[hazard]])/len(dynamic_severities[hazard]))*100)
            comparison["Severity average distance"].append(np.average([dist_dict[i]-dist_dict[static_severities[hazard]] for i in dynamic_severities[hazard]]))
            num_same = len([i for i in range(len(dynamic_severities[hazard])) if (dynamic_likelihoods[hazard][i]==static_likelihoods[hazard] and dynamic_severities[hazard][i]==static_severities[hazard])])
            comparison["Likelihood and Severity % same"].append((num_same/len(dynamic_severities[hazard]))*100)
        results_df = pd.DataFrame(comparison, index=self.hazards)
        return percent_same, results_df