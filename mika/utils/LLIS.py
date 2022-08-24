# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 16:14:04 2022

@author: srandrad
"""

def drop_uniformitive_text(data_df, text_cols):
    rows_to_drop = []
    for i in range(0, len(data_df)):
        drop = False
        for attr in text_cols:
            if (str(data_df.iloc[i][attr]).strip("()").lower().startswith("see") or str(data_df.iloc[i][attr]).strip("()").lower().startswith("same") or str(data_df.iloc[i][attr])=="" or isinstance(data_df.iloc[i][attr],float) or str(data_df.iloc[i][attr]).lower().startswith("none")):
                drop = True
            else:
                drop = False
                break
        if drop == True:
            rows_to_drop.append(i)
    data_df = data_df.drop(rows_to_drop).reset_index(drop=True)
    return data_df