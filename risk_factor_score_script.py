# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 09:41:19 2022
Update: removed topic model plus dependency
@author: srandrad
"""

import pandas as pd
import sys
import os
from time import sleep
from tqdm import tqdm

def prepare_data(file, combine_cols, id_col, extra_cols=[]):
    # load data
    data_df = pd.read_csv(open(file,encoding='utf8',errors='ignore'))
    #combine columns
    columns_to_drop = [cols for cols in data_df.columns if cols not in [id_col]+extra_cols]
    rows_to_drop = []
    combined_text = []
    sleep(0.5)
    for i in tqdm(range(0, len(data_df)), "Combining Columns…"):
        text = ""
        for attr in combine_cols:
            if not(str(data_df.iloc[i][attr]).strip("()").lower().startswith("see") or str(data_df.iloc[i][attr]).strip("()").lower().startswith("same") or str(data_df.iloc[i][attr])=="" or isinstance(data_df.iloc[i][attr],float) or str(data_df.iloc[i][attr]).lower().startswith("none")):
                if text != "":
                    text += ". " 
                text += str(data_df.iloc[i][attr])
        if text == "":
            rows_to_drop.append(i)
        combined_text.append(text)
    sleep(0.5)
    data_df["Combined Text"] = combined_text
    data_df = data_df.drop(columns_to_drop, axis=1)
    data_df = data_df.drop(rows_to_drop).reset_index(drop=True)
    return data_df

#read in lexicons
lexicon_file = os.path.join('results', 'Risk_Factor_Lexicons.csv')
lexicons = pd.read_csv(lexicon_file, index_col=0) 
#read in llis using topic model plus
llis_file = os.path.join('data', 'LLIS', 'useable_LL_combined.csv')
combine_cols = ['Abstract', 'Lesson(s) Learned', 'Recommendation(s)', 'Driving Event']
llis_df = prepare_data(llis_file, combine_cols, 'Lesson ID')
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