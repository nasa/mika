# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 15:57:11 2022

@author: srandrad
"""
import pypyodbc
import csv
import os
import pandas as pd

def get_phase_mishap_from_occurrence_code(occurrence_code, occurrence_dict):
    occurrence_code = str(occurrence_code)
    phase_code = "xxx" + occurrence_code[3:]
    mishap_code = occurrence_code[0:3] + "xxx"
    if mishap_code not in occurrence_dict or phase_code not in occurrence_dict:
        return None, None
    else:
        mishap = occurrence_dict[mishap_code]
        phase = occurrence_dict[phase_code]
        return phase, mishap

def get_phase_mishap_from_occurrence_and_phase(occurrence_code, phase_code, occurrence_dict, phase_dict):
    phase_code =  str(phase_code)
    mishap_code = str(occurrence_code)
    if mishap_code not in occurrence_dict or phase_code not in phase_dict:
        return None, None
    else:
        mishap = occurrence_dict[mishap_code]
        phase = phase_dict[phase_code]
        return phase, mishap

def add_list_of_phases_mishaps(df, occurrence_dict, from_occurrence_code, phase_dict=None): #add error handling for missing codes
    phases = []
    mishaps = []
    ind_to_drop = []
    for i in range(len(df)):
        if from_occurrence_code: 
            occurrence_code = df.at[i, 'Occurrence_Code']
            phase, mishap = get_phase_mishap_from_occurrence_code(occurrence_code, occurrence_dict)
        else:
            occurrence_code = df.at[i, 'Occurrence_Code']
            phase_code = df.at[i, 'Phase_of_Flight']
            phase, mishap = get_phase_mishap_from_occurrence_and_phase(occurrence_code, phase_code, occurrence_dict, phase_dict)
        if phase == None or mishap == None:
            ind_to_drop.append(i)
        phases.append(phase)
        mishaps.append(mishap)
    df['Phase'] = phases
    df['Mishap Category'] = mishap
    df = df.drop(ind_to_drop).reset_index(drop=True)
    return df

file_path = r"C:\Users\srandrad\OneDrive - NASA\Desktop\ntsb_test"

for root, dirs, files in os.walk(file_path):
    for file in files:
        if ".mdb" in file:
            pypyodbc.lowercase = False
            conn = pypyodbc.connect(
                r"Driver={Microsoft Access Driver (*.mdb, *.accdb)};" +
                r"Dbq="+os.path.join(root, file)+";") #connect to db
    
            cur = conn.cursor() #open cursor
            SQL_query = "SELECT events.ev_id, damage, ev_highest_injury, inj_tot_f, ev_date, ev_year, ev_state, wind_dir_ind, wind_vel_ind, light_cond, sky_cond_nonceil, narr_accp, narr_accf, narr_cause, Occurrence_Code, acft_make, acft_model, flt_plan_filed, flight_plan_activated, total_seats \
                        FROM ((events \
                        INNER JOIN \
                            Events_Sequence \
                        ON \
                            events.ev_id = Events_Sequence.ev_id) \
                        INNER JOIN \
                            aircraft \
                        ON \
                            events.ev_id = aircraft.ev_id)\
                        INNER JOIN \
                             narratives \
                        ON \
                             narratives.ev_id = events.ev_id;" #write SQL
            cur.execute(SQL_query); #execute SQL
            output_file = os.path.join(root, file).strip(".mdb") + "event_full_tables.csv" #open outputfile and write query results
            with open(output_file, 'w', newline='', encoding='utf8') as f:
                writer = csv.writer(f)    
                # ADD LINE BEFORE LOOP
                writer.writerow([i[0] for i in cur.description])  
                for row in cur.fetchall() :
                    writer.writerow(row)
            cur.close()

            #documents with occurrences
            cur = conn.cursor() #open cursor
            SQL_query = "SELECT events.ev_id, damage, ev_highest_injury, inj_tot_f, ev_date, ev_year, ev_state, wind_dir_ind, wind_vel_ind, light_cond, sky_cond_nonceil, narr_accp, narr_accf, narr_cause, Occurrence_Code, Phase_of_Flight, acft_make, acft_model, flt_plan_filed, flight_plan_activated, total_seats \
                        FROM ((events \
                        INNER JOIN \
                            Occurrences \
                        ON \
                            events.ev_id = Occurrences.ev_id) \
                        INNER JOIN \
                            aircraft \
                        ON \
                            events.ev_id = aircraft.ev_id)\
                        INNER JOIN \
                             narratives \
                        ON \
                             narratives.ev_id = events.ev_id;" #write SQL
            cur.execute(SQL_query); #execute SQL
            output_file = os.path.join(root, file).strip(".mdb") + "occurrence_full_tables.csv" #open outputfile and write query results
            with open(output_file, 'w', newline='', encoding='utf8') as f:
                writer = csv.writer(f)    
                # ADD LINE BEFORE LOOP
                writer.writerow([i[0] for i in cur.description])  
                for row in cur.fetchall() :
                    writer.writerow(row)
            cur.close()
            #pulling documentation
            cur = conn.cursor() #open cursor
            SQL_query = "SELECT *  FROM eADMSPUB_DataDictionary;" #write SQL
            cur.execute(SQL_query);
            output_file = os.path.join(root, file).strip(".mdb") + "eADMSPUB_DataDictionary.csv" #open outputfile and write query results
            with open(output_file, 'w', newline='', encoding='utf8') as f:
                writer = csv.writer(f)    
                writer.writerow([i[0] for i in cur.description])  
                for row in cur.fetchall() :
                    writer.writerow(row)
    
            cur.close()
            conn.close()

            #pulling narratives
dfs = []
documentation_dfs = []
for root, dirs, files in os.walk(file_path):
    for file in files:
        if "eADMSPUB_DataDictionary.csv" in file:
            df = pd.read_csv(os.path.join(root, file))
            os.remove(os.path.join(root, file))
            documentation_dfs.append(df)

docs = pd.concat(documentation_dfs).drop_duplicates().reset_index(drop=True)
keys = docs.loc[(docs['Column']=='Occurrence_Code') & (docs['Table']=='Events_Sequence')].reset_index(drop=True)['code_iaids'].tolist()
vals = docs.loc[(docs['Column']=='Occurrence_Code') & (docs['Table']=='Events_Sequence')].reset_index(drop=True)['meaning'].tolist()
occurrence_dict =dict(zip(keys, vals))

dfs = []
for root, dirs, files in os.walk(file_path):
    for file in files:
        if "event_full_tables.csv" in file: # need to drop rows with multiple event ids
            df = pd.read_csv(os.path.join(root, file))
            os.remove(os.path.join(root, file))
            dfs.append(df)

ntsb_from_events = pd.concat(dfs).drop_duplicates().reset_index(drop=True)
ntsb_from_events = add_list_of_phases_mishaps(ntsb_from_events, occurrence_dict, True)
ntsb_from_events = ntsb_from_events.drop(['Occurrence_Code'], axis=1)

dfs = []
for root, dirs, files in os.walk(file_path):
    for file in files:
        if "occurrence_full_tables.csv" in file: # need to drop rows with multiple event ids
            df = pd.read_csv(os.path.join(root, file))
            os.remove(os.path.join(root, file))
            dfs.append(df)
keys = docs.loc[(docs['Column']=='Occurrence_Code') & (docs['Table']=='Occurrences')].reset_index(drop=True)['code_iaids'].tolist()
vals = docs.loc[(docs['Column']=='Occurrence_Code') & (docs['Table']=='Occurrences')].reset_index(drop=True)['meaning'].tolist()
occurrence_dict =dict(zip(keys, vals))
keys = docs.loc[(docs['Column']=='Phase_of_Flight') & (docs['Table']=='Occurrences')].reset_index(drop=True)['code_iaids'].tolist()
vals = docs.loc[(docs['Column']=='Phase_of_Flight') & (docs['Table']=='Occurrences')].reset_index(drop=True)['meaning'].tolist()
phase_dict =dict(zip(keys, vals))

ntsb_from_occurrences = pd.concat(dfs).drop_duplicates().reset_index(drop=True)
ntsb_from_occurrences = add_list_of_phases_mishaps(ntsb_from_occurrences, occurrence_dict, False, phase_dict) #need to make different occurrence dicts
ntsb_from_occurrences = ntsb_from_occurrences.drop(['Occurrence_Code', 'Phase_of_Flight'], axis=1)
ntsb = pd.concat([ntsb_from_events, ntsb_from_occurrences]).reset_index(drop=True)
ntsb.to_csv("ntsb_full.csv")
print("Size of NTSB:", len(ntsb))

dfs = []
#get narratives
for root, dirs, files in os.walk(file_path):
    for file in files:
        if ".mdb" in file or file=='PRE1982.MDB':
            pypyodbc.lowercase = False
            conn = pypyodbc.connect(
                r"Driver={Microsoft Access Driver (*.mdb, *.accdb)};" +
                r"Dbq="+os.path.join(root, file)+";") #connect to db
            cur = conn.cursor() #open cursor
            if 'PRE1982' not in file:
                cur.execute("SELECT * FROM narratives"); #execute SQL
            else:
                cur.execute("SELECT REMARKS, CAUSE FROM tblSecondHalf"); #execute SQL
            output_file = os.path.join(root, file).strip(".mdb") + "_full_narratives.csv" #open outputfile and write query results
            with open(output_file, 'w', newline='', encoding='utf8') as f:
                writer = csv.writer(f)    
                writer.writerow([i[0] for i in cur.description])  
                for row in cur.fetchall() :
                    writer.writerow(row)
    
            cur.close()
            conn.close()

dfs = []
for root, dirs, files in os.walk(file_path):
    for file in files:
        if "_full_narratives.csv" in file:
            df = pd.read_csv(os.path.join(root, file))
            os.remove(os.path.join(root, file))
            dfs.append(df)

ntsb = pd.concat(dfs).drop_duplicates().reset_index(drop=True)
print("Size of just NTSB narratives:", len(ntsb))
ntsb.to_csv("ntsb_full_narratives.csv")
