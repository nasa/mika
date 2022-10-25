# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 11:23:55 2022

@author: srandrad
"""

import pandas as pd
import os
import sys
import spacy
from tqdm import tqdm
from spacy.matcher import Matcher
import re

def get_abbrevation_dict(file):
    abbr = pd.read_excel(file)
    keys = [a.strip(" ") for a in abbr['Abbreviation'].tolist()]
    values = [a.strip(" ") for a in abbr['Meaning'].tolist()]
    dictionary = dict(zip(keys, values))
    return dictionary

def replace_word(orig_text, replacement, matcher, nlp):
    tok = nlp(orig_text)
    text = ''
    buffer_start = 0
    for _, match_start, _ in matcher(tok):
        if match_start > buffer_start:  # If we've skipped over some tokens, let's add those in (with trailing whitespace if available)
            text += tok[buffer_start: match_start].text + tok[match_start - 1].whitespace_
        text += replacement + tok[match_start].whitespace_  # Replace token, with trailing whitespace if available
        buffer_start = match_start + 1
    if buffer_start < len(tok): 
        text += tok[buffer_start:].text
    
    return text

def make_matcher(abbreviation_dict, asrs_reports, text_cols, nlp):
    matcher = Matcher(nlp.vocab)
    for key in tqdm(abbreviation_dict):
        matcher = Matcher(nlp.vocab)
        if len(key.split(" ")) >1:
            matcher.add(key, [[{"LOWER": key_component} for key_component in key.split(" ")]])
        else:
            matcher.add(key, [[{"LOWER": key}]])
        for i in range(len(asrs_reports)):
            for col in text_cols:
                text = asrs_reports.at[i, col]
                if type(text) is str:
                    if key in text:
                        text = replace_word(text, abbreviation_dict[key], matcher, nlp)
                        asrs_reports.at[i,col] = text
    return asrs_reports

def find_replace_abbreviations(text, abbreviation_dict):
    for abb in abbreviation_dict:
        if " "+abb in text or text.startswith(abb+" "):
            if text.startswith(abb+" "): #handles case where sentence begins with an abbreviation
                text = " ".join([abbreviation_dict[abb]]+text.split(" "))
            substrings = re.findall("( "+abb+'[ .!?;:]'+"| '"+abb+'[ .!?;:]'+")", text)
            if len(substrings)>0:
                parts_of_string = re.split("( "+abb+'[ .!?;:]'+"| '"+abb+'[ .!?;:]'+")", text)
                fixed_str = "".join([parts_of_string[i+i]+substrings[i].replace(abb, abbreviation_dict[abb]) for i in range(len(substrings))]+[parts_of_string[-1]])
                text = fixed_str
    return text


def find_abbreviations_and_XYs(abbreviation_dict, asrs_reports, text_cols):
    for i in tqdm(range(len(asrs_reports))):
        for col in text_cols:
            text = asrs_reports.at[i, col]
            if type(text) is str:
                text = find_replace_abbreviations(text, abbreviation_dict)
                text = re.sub("( X{1,}[ .!?;:])", "", text) #removes Xs 
                text = re.sub("( Y{1,}[ .!?;:])", "", text) #removes Ys
            asrs_reports.at[i, col] = text
    return asrs_reports

def convert_abbreviations_to_full_text(asrs_reports, text_cols, file):
    nlp = spacy.load("en_core_web_trf")
    abbreviation_dict = get_abbrevation_dict(file)
    asrs_reports = make_matcher(abbreviation_dict, asrs_reports, text_cols, nlp)
    #BASE OPS
    #abbrevs with / in them
    return asrs_reports

def convert_abbreviations_with_re(asrs_reports, text_cols, file):
    abbreviation_dict = get_abbrevation_dict(file)
    asrs_reports = find_abbreviations_and_XYs(abbreviation_dict, asrs_reports, text_cols)
    return asrs_reports

file_path = r"C:\Users\srandrad\OneDrive - NASA\Desktop\asrs_data"
dfs = []
for root, dirs, files in os.walk(file_path):
    for file in files:
        df = pd.read_csv(os.path.join(root, file))
        dfs.append(df)

asrs = pd.concat(dfs).reset_index(drop=True).iloc[1:][:].reset_index(drop=True)
ASRS_text_cols = ['Report 1', 'Report 1.1', 'Report 2',	'Report 2.1', 'Report 1.2']

file = r"C:\Users\srandrad\Downloads\ASRS_acronyms.xlsx"
#asrs = convert_abbreviations_to_full_text(asrs, ASRS_text_cols, file)
asrs = convert_abbreviations_with_re(asrs, ASRS_text_cols, file)
asrs.to_csv("ASRS_1988_2022_cleaned.csv")