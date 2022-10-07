# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 15:57:11 2022

@author: srandrad
"""

import pypyodbc
import csv
import os
import pandas as pd

file_path = r"C:\Users\srandrad\OneDrive - NASA\Desktop\ntsb"
dfs = []
for root, dirs, files in os.walk(file_path):
    for file in files:
        if ".mdb" in file:
            pypyodbc.lowercase = False
            conn = pypyodbc.connect(
                r"Driver={Microsoft Access Driver (*.mdb, *.accdb)};" +
                r"Dbq="+os.path.join(root, file)+";")
    
            # OPEN CURSOR AND EXECUTE SQL
            cur = conn.cursor()
            SQL_query = "SELECT *  \
                        FROM ((narratives \
                        INNER JOIN \
                            Events_Sequence \
                        ON \
                            narratives.ev_id = Events_Sequence.ev_id) \
                        INNER JOIN \
                            injury \
                        ON \
                           narratives.ev_id = injury.ev_id) \
                        INNER JOIN \
                            aircraft \
                        ON \
                            narratives.ev_id = aircraft.ev_id;"
            cur.execute(SQL_query);#"SELECT * FROM narratives");
    
            # OPEN CSV AND ITERATE THROUGH RESULTS
            #output_file = os.path.join(root, file).replace(".mdb", ".csv")
            output_file = os.path.join(root, file).strip(".mdb") + "_full_tables.csv"
            with open(output_file, 'w', newline='', encoding='utf8') as f:
                writer = csv.writer(f)    
                # ADD LINE BEFORE LOOP
                writer.writerow([i[0] for i in cur.description])  
    
                for row in cur.fetchall() :
                    writer.writerow(row)
    
            cur.close()
            conn.close()


file_path = r"C:\Users\srandrad\OneDrive - NASA\Desktop\ntsb"
dfs = []
for root, dirs, files in os.walk(file_path):
    for file in files:
        if "_full_tables.csv" in file:
            df = pd.read_csv(os.path.join(root, file))
            dfs.append(df)
#pre_1987 = pd.read_excel(r"C:\Users\srandrad\OneDrive - NASA\Desktop\ntsb\tblSecondHalf.xlsx")
#pre_1987 = pre_1987[['REMARKS', 'CAUSE']]

#dfs.append(pre_1987)

ntsb = pd.concat(dfs).reset_index(drop=True)
print(len(ntsb))
ntsb.to_csv("ntsb_full_tables.csv")

dfs = []
for root, dirs, files in os.walk(file_path):
    for file in files:
        if ".mdb" in file:
            pypyodbc.lowercase = False
            conn = pypyodbc.connect(
                r"Driver={Microsoft Access Driver (*.mdb, *.accdb)};" +
                r"Dbq="+os.path.join(root, file)+";")
    
            # OPEN CURSOR AND EXECUTE SQL
            cur = conn.cursor()
            
            cur.execute("SELECT * FROM narratives");
    
            # OPEN CSV AND ITERATE THROUGH RESULTS
            output_file = os.path.join(root, file).strip(".mdb") + "_full_narratives.csv"
            with open(output_file, 'w', newline='', encoding='utf8') as f:
                writer = csv.writer(f)    
                # ADD LINE BEFORE LOOP
                writer.writerow([i[0] for i in cur.description])  
    
                for row in cur.fetchall() :
                    writer.writerow(row)
    
            cur.close()
            conn.close()


file_path = r"C:\Users\srandrad\OneDrive - NASA\Desktop\ntsb"
dfs = []
for root, dirs, files in os.walk(file_path):
    for file in files:
        if "_full_narratives.csv" in file:
            df = pd.read_csv(os.path.join(root, file))
            dfs.append(df)
pre_1987 = pd.read_excel(r"C:\Users\srandrad\OneDrive - NASA\Desktop\ntsb\tblSecondHalf.xlsx")
pre_1987 = pre_1987[['REMARKS', 'CAUSE']]

dfs.append(pre_1987)

ntsb = pd.concat(dfs).reset_index(drop=True)
print(len(ntsb))
ntsb.to_csv("ntsb_full_narratives.csv")

