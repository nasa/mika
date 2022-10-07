# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 11:23:55 2022

@author: srandrad
"""

import pandas as pd
import os
import sys

file_path = r"C:\Users\srandrad\OneDrive - NASA\Desktop\asrs_data"
dfs = []
for root, dirs, files in os.walk(file_path):
    for file in files:
        df = pd.read_csv(os.path.join(root, file))
        dfs.append(df)

asrs = pd.concat(dfs).reset_index(drop=True)
asrs.to_csv("ASRS_1988_2022.csv")