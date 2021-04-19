# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 13:28:51 2020

@author: srandrad
"""


import pingouin as pg
import pandas as pd

df = pd.read_csv("C:/Users/srandrad/Desktop/smart_stereo/validation_set_HW.csv")
cols_to_drop = ["lesson number", "notes", "H notes"]
df = df.drop(cols_to_drop, axis=1)
S = []
H = []
encoding = {"use":1, "not use":0}
for i in range (0, len(df)):
    S.append(encoding[df.iloc[i]['use/not use']])
    H.append(encoding[df.iloc[i]['H use/not use']])
df = pd.DataFrame({
    "s":S,
    "h":H
    })
alpha = pg.cronbach_alpha(data = df)
print(alpha)

from sklearn.metrics import cohen_kappa_score
kappa = cohen_kappa_score(S, H)
print(kappa)