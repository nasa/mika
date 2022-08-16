# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 16:21:52 2020

@author: srandrad
"""


import pandas as pd
useable_LL = pd.read_csv(r"C:\Users\srandrad\OneDrive - NASA\Desktop\useable_LL_combined.csv")
#"Driving event", "Reccomendation(s)", "Lesson(s) Learned", "Driving Event
num_sections = 0
for i in range (len(useable_LL)):
    if str(useable_LL.iloc[i]["Recommendation(s)"]).lower().startswith("see") or str(useable_LL.iloc[i]["Recommendation(s)"])=="":
        continue
    if str(useable_LL.iloc[i]['Driving Event']).lower().startswith("see") or str(useable_LL.iloc[i]['Driving Event'])=="":
        continue
    if str(useable_LL.iloc[i]["Lesson(s) Learned"]).lower().startswith("see") or str(useable_LL.iloc[i]["Lesson(s) Learned"])=="":
        continue
    num_sections +=1

print(num_sections/len(useable_LL))

useable_LL = pd.read_csv(r"C:\Users\srandrad\OneDrive - NASA\Desktop\useable_LL.csv")
#"Driving event", "Reccomendation(s)", "Lesson(s) Learned", "Driving Event
num_sections = 0
for i in range (len(useable_LL)):
    if str(useable_LL.iloc[i]["Recommendation(s)"]).lower().startswith("see") or str(useable_LL.iloc[i]["Recommendation(s)"])=="":
        continue
    if str(useable_LL.iloc[i]['Driving Event']).lower().startswith("see") or str(useable_LL.iloc[i]['Driving Event'])=="":
        continue
    if str(useable_LL.iloc[i]["Lesson(s) Learned"]).lower().startswith("see") or str(useable_LL.iloc[i]["Lesson(s) Learned"])=="":
        continue
    num_sections +=1

print(num_sections/len(useable_LL))