# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 2022

@author: srandrad
"""
import os
import pandas as pd

folder_path = "SAFENET"

dfs = []
for root, dirs, files in os.walk(folder_path):
    for file in files:
        df = pd.read_csv(os.path.join(root, file))
        dfs.append(df)

safenet = pd.concat(dfs).reset_index(drop=True)

file = "SAFENET_full.csv"
safenet.to_csv(file)