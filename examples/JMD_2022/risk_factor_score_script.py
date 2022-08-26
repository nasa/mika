# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 09:41:19 2022
Update: removed topic model plus dependency
@author: srandrad
"""

import pandas as pd
import sys
import os
sys.path.append(os.path.join("..", ".."))
from mika.utils import Data
from mika.utils.LLIS import drop_uniformitive_text

#read in lexicons
lexicon_file = os.path.join('Risk_Factor_Lexicons.csv')
lexicons = pd.read_csv(lexicon_file, index_col=0) 
#read in llis using Data
llis_file = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir)),'data', 'LLIS', 'useable_LL_combined.csv')
combine_cols = ['Abstract', 'Lesson(s) Learned', 'Recommendation(s)', 'Driving Event']
llis = Data()
llis.load(llis_file, id_col='Lesson ID', )
llis.data_df = drop_uniformitive_text(llis.data_df, combine_cols)
llis.prepare_data(combine_columns=combine_cols)
llis_df = llis.data_df
#llis_df_2 = prepare_data(llis_file, combine_cols, 'Lesson ID')

#combining columns, loading data
def calc_risk_factor_scores(df, text_col, lexicons):
    risk_factors = [c for c in lexicons.columns]
    risk_factor_words = {risk_factor: lexicons[risk_factor].dropna().tolist() for risk_factor in risk_factors}
    risk_factor_scores = {risk_factor:[] for risk_factor in risk_factors} #for storing each documents score
    
    for doc in df[text_col].tolist():
        temp_risk_factor_counts = {risk_factor:0 for risk_factor in risk_factors}
        total_risk_factor_words = 0
        for risk_factor in risk_factors:
            for word in risk_factor_words[risk_factor]:
                if word in doc:
                    total_risk_factor_words += 1
                    temp_risk_factor_counts[risk_factor] += 1
        
        for risk_factor in risk_factors:
            if total_risk_factor_words == 0:
                score = 0
            else:
                score = temp_risk_factor_counts[risk_factor]/total_risk_factor_words
            risk_factor_scores[risk_factor].append(score)
    
    df_with_scores = pd.concat([df, pd.DataFrame(risk_factor_scores)], axis=1)
    return df_with_scores

llis_with_scores = calc_risk_factor_scores(llis_df, 'Combined Text', lexicons)
print(llis_with_scores)
#llis_with_scores.to_csv(os.path.join('results', 'llis_with_risk_factor_scores.csv'))