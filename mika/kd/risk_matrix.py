# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 12:26:04 2022

@author: srandrad
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors
import seaborn as sn

class RiskMatrix():
    def __init__(self, levels=[], severity_levels=[], likelihood_levels=[], rm_type=None):
        self.hazards = []
        self.hazard_likelihoods = {}
        self.hazard_severities = {}
        
        return
    
    def get_likelihood_FAA(rates):
        curr_likelihoods = {hazard:0 for hazard in rates}
        for hazard in rates:
            r = rates[hazard]
            if r>=100:
                likelihood = 'Frequent'
            elif r>=10 and r<100:
                likelihood = 'Probable'
            elif r>=1 and r<10:
                likelihood = 'Remote'
            elif r>=0.1 and r<1:
                likelihood = 'Extremely Remote'
            elif r<0.1:
                likelihood = 'Extremely Improbable'
            curr_likelihoods[hazard] = likelihood
        return curr_likelihoods
    
    def get_likelihood_USFS(rates):
        curr_likelihoods = {hazard:0 for hazard in rates}
        for hazard in rates:
            r = rates[hazard]
            if r>=100:
                likelihood = 'Frequent'
            elif r>=10 and r<100:
                likelihood = 'Probable'
            elif r>=1 and r<10:
                likelihood = 'Occasional'
            elif r>=0.1 and r<1:
                likelihood = 'Remote'
            elif r<0.1:
                likelihood = 'Improbable'
            curr_likelihoods[hazard] = likelihood
        return curr_likelihoods
    
    def plot_USFS_risk_matrix(likelihoods, severities, figsize=(9,5), save=False, results_path="", fontsize=12, max_chars=20, title=False):
        hazards = [h for h in likelihoods]
        curr_likelihoods = likelihoods
        curr_severities = severities
        annotation_df = pd.DataFrame([["" for i in range(4)] for j in range(5)],
                             columns=['Negligible', 'Marginal', 'Critical', 'Catastrophic'],
                              index=['Frequent', 'Probable', 'Occasional', 'Remote','Improbable'])
        annot_font = fontsize
        hazard_likelihoods = {hazard:"" for hazard in hazards}; hazard_severities={hazard:"" for hazard in hazards}
        for hazard in hazards:
            hazard_likelihoods[hazard] = curr_likelihoods[hazard]
            hazard_severities[hazard] = curr_severities[hazard]
            new_annot = annotation_df.at[curr_likelihoods[hazard], curr_severities[hazard]]
            if new_annot != "": new_annot += ", "
            hazard_annot = hazard.split(" ")
            #if line>20 then new line
            if len(hazard_annot)>1 and len(new_annot.split("\n")[-1]) + len(hazard_annot[0]) +len(hazard_annot[1]) < max_chars:
                new_annot += hazard_annot[0] + " "+ hazard_annot[1] 
                annot_ind = 2
            elif len(new_annot.split("\n")[-1]) + len(hazard_annot[0]) < max_chars:
                new_annot += hazard_annot[0]
                annot_ind = 1
            elif len(hazard_annot)>1 and len(hazard_annot[1]) + len(hazard_annot[0]) < max_chars:
                new_annot += "\n" + hazard_annot[0] + " " + hazard_annot[1]
                annot_ind = 2
            else:
                new_annot += "\n" + hazard_annot[0]
                annot_ind = 1
            if len(hazard_annot)>1 and annot_ind<len(hazard_annot):
                new_annot += "\n"+" ".join(hazard_annot[annot_ind:])
            annotation_df.at[curr_likelihoods[hazard], curr_severities[hazard]] = new_annot #+= (str(hazard_annot))
        
        df = pd.DataFrame([[2, 3, 4, 4], [2, 3, 4, 4], [1, 2, 3, 4], [1, 2, 2, 3], [1, 2, 2, 2]],
                          columns=['Negligible', 'Marginal', 'Critical', 'Catastrophic'],
                          index=['Frequent', 'Probable', 'Occasional', 'Remote','Improbable'])
        fig,ax = plt.subplots(figsize=figsize)
        myColors = (mcolors.to_rgb(mcolors.cnames['green']),
                    mcolors.to_rgb(mcolors.cnames['dodgerblue']),
                    mcolors.to_rgb(mcolors.cnames['yellow']),
                    mcolors.to_rgb(mcolors.TABLEAU_COLORS['tab:red']))
        cmap = LinearSegmentedColormap.from_list('Custom', myColors, len(myColors))
        #annot df has hazards in the cell they belong to #annot=annotation_df
        sn.heatmap(df, annot=annotation_df, fmt='s',annot_kws={'fontsize':annot_font},cbar=False,cmap=cmap)
        if title: plt.title("Risk Matrix", fontsize=fontsize)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)#, ha="right")
        plt.tick_params(labelsize=fontsize)
        plt.ylabel("Likelihood", fontsize=fontsize)
        plt.xlabel("Severity", fontsize=fontsize)
        minor_ticks = np.arange(1, 5, 1)
        ax.set_xticks(minor_ticks, minor=True)
        ax.set_yticks(minor_ticks, minor=True)
        ax.tick_params(which='minor',length=0, grid_color='black', grid_alpha=1)
        ax.grid(which='minor', alpha=1)
        if save: 
            plt.savefig(results_path+".pdf", bbox_inches="tight")
        plt.show()
    
    def plot_risk_matrix(likelihoods, severities, figsize=(9,5), save=False, results_path="", fontsize=12, max_chars=20):
        hazards = [h for h in likelihoods]
        curr_likelihoods = likelihoods
        curr_severities = severities
        annotation_df = pd.DataFrame([["" for i in range(5)] for j in range(5)],
                             columns=['Minimal Impact', 'Minor Impact', 'Major Impact', 'Hazardous Impact', 'Catastrophic Impact'],
                              index=['Frequent', 'Probable', 'Remote','Extremely Remote', 'Extremely Improbable'])
        annot_font = fontsize
        hazard_likelihoods = {hazard:"" for hazard in hazards}; hazard_severities={hazard:"" for hazard in hazards}
        for hazard in hazards:
            hazard_likelihoods[hazard] = curr_likelihoods[hazard]
            hazard_severities[hazard] = curr_severities[hazard]
            new_annot = annotation_df.at[curr_likelihoods[hazard], curr_severities[hazard]]
            if new_annot != "": new_annot += ", "
            hazard_annot = hazard.split(" ")
            #if line>20 then new line
            if len(hazard_annot)>1 and len(new_annot.split("\n")[-1]) + len(hazard_annot[0]) +len(hazard_annot[1]) < max_chars:
                new_annot += hazard_annot[0] + " "+ hazard_annot[1] 
                annot_ind = 2
            elif len(new_annot.split("\n")[-1]) + len(hazard_annot[0]) < max_chars:
                new_annot += hazard_annot[0]
                annot_ind = 1
            elif len(hazard_annot)>1 and len(hazard_annot[1]) + len(hazard_annot[0]) < max_chars:
                new_annot += "\n" + hazard_annot[0] + " " + hazard_annot[1]
                annot_ind = 2
            else:
                new_annot += "\n" + hazard_annot[0]
                annot_ind = 1
            if len(hazard_annot)>1 and annot_ind<len(hazard_annot):
                new_annot += "\n"+" ".join(hazard_annot[annot_ind:])
            annotation_df.at[curr_likelihoods[hazard], curr_severities[hazard]] = new_annot #+= (str(hazard_annot))
        
        df = pd.DataFrame([[0, 5, 10, 10, 10], [0, 5, 5, 10, 10], [0, 0, 5, 5, 10],
                [0, 0, 0, 5, 5], [0, 0, 0, 0, 5]],
              columns=['Minimal Impact', 'Minor Impact', 'Major Impact', 'Hazardous Impact', 'Catastrophic Impact'],
               index=['Frequent', 'Probable', 'Remote','Extremely Remote', 'Extremely Improbable'])
        fig,ax = plt.subplots(figsize=figsize)
        #annot df has hazards in the cell they belong to #annot=annotation_df
        sn.heatmap(df, annot=annotation_df, fmt='s',annot_kws={'fontsize':annot_font},cbar=False,cmap='RdYlGn_r')
        plt.title("Risk Matrix", fontsize=fontsize)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)#, ha="right")
        plt.tick_params(labelsize=fontsize)
        plt.ylabel("Likelihood", fontsize=fontsize)
        plt.xlabel("Severity", fontsize=fontsize)
        minor_ticks = np.arange(1, 6, 1)
        ax.set_xticks(minor_ticks, minor=True)
        ax.set_yticks(minor_ticks, minor=True)
        ax.tick_params(which='minor',length=0, grid_color='black', grid_alpha=1)
        ax.grid(which='minor', alpha=1)
        if save: 
            plt.savefig(results_path+".pdf", bbox_inches="tight")
        plt.show()