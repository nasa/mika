# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 13:14:21 2022

@author: srandrad
"""
import pandas as pd
import numpy as np

#safecom utils
def get_SAFECOM_severity_FAA(severities):
    """
    Assigns a severity category according to FAA risk matrix

    Parameters
    ----------
    severities : dict
        Dictionary with keys as hazards and values as average severities.

    Returns
    -------
    curr_severities : dict
        Dictionary with keys as hazards and values as severity category.

    """
    curr_severities = {hazard:0 for hazard in severities}
    for hazard in severities:
        s = severities[hazard]
        if s<=0.1: #negligible impact
            severity = 'Minimal Impact'
        elif s>0.1 and s <= 0.5:
            severity = 'Minor Impact'
        elif s>0.5 and s<=1:
            severity = 'Major Impact'
        elif s>1 and s<=2:
            severity = 'Hazardous Impact'
        elif s>2:
            severity = 'Catastrophic Impact'
        curr_severities[hazard] = severity
    return curr_severities


def get_SAFECOM_severity_USFS(severities):
    """
    Assigns a severity category according to USFS risk matrix

    Parameters
    ----------
    severities : dict
        Dictionary with keys as hazards and values as average severities.

    Returns
    -------
    curr_severities : dict
        Dictionary with keys as hazards and values as severity category.

    """
    curr_severities = {hazard:0 for hazard in severities}
    for hazard in severities:
        s = severities[hazard]
        if s<=0.1: #negligible impact
            severity = 'Negligible'
        elif s>0.1 and s <= 1:
            severity = 'Marginal'
        elif s>1 and s<=2:
            severity = 'Critical'
        elif s>2:
            severity = 'Catastrophic'
        curr_severities[hazard] = severity
    return curr_severities

def get_UAS_likelihood_FAA(frequency):
    """
    Assigns a liklihood category according to FAA risk matrix

    Parameters
    ----------
    rates : dict
        Dictionary with keys as hazards and values as rates of occurence.

    Returns
    -------
    curr_likelihoods : dict
        Dictionary with keys as hazards and values as likelihood category.

    """
    curr_likelihoods = {hazard:0 for hazard in frequency}
    for hazard in frequency:
        r = frequency[hazard]
        if r==5:
            likelihood = 'Frequent'
        elif r==4:
            likelihood = 'Probable'
        elif r==3:
            likelihood = 'Remote'
        elif r==2:
            likelihood = 'Extremely Remote'
        elif r==1:
            likelihood = 'Extremely Improbable'
        curr_likelihoods[hazard] = likelihood
    return curr_likelihoods

def get_UAS_likelihood_USFS(frequency):
    """
    Assigns a liklihood category according to USFS risk matrix

    Parameters
    ----------
    rates : dict
        Dictionary with keys as hazards and values as rates of occurence.

    Returns
    -------
    curr_likelihoods : dict
        Dictionary with keys as hazards and values as likelihood category.

    """
    curr_likelihoods = {hazard:0 for hazard in frequency}
    for hazard in frequency:
        r = frequency[hazard]
        if r==5:
            likelihood = 'Frequent'
        elif r==4:
            likelihood = 'Probable'
        elif r==3:
            likelihood = 'Occasional'
        elif r==2:
            likelihood = 'Remote'
        elif r==1:
            likelihood = 'Improbable'
        curr_likelihoods[hazard] = likelihood
    return curr_likelihoods

def correct_regions(df):
    """corrects the region in safecom reports to correspond to the correct USFS region

    Parameters
    ----------
    df :  pandas DataFrame
        pandas datframe containing documents.

    Returns
    -------
    df :  pandas DataFrame
        pandas datframe containing documents with new column for corrected region.
    """
    region_dict ={
        'Region 05 Pacific Southwest Region': ['5', 'California', 'Hawaii', 'Pacific West Regional Office','California State Office','R2-Southwest Regional Office', 'National Guard'],
        'Region 06 Pacific Northwest Region':['Oregon/Washington State Office', 'Washington', 'Oregon', 'Washington Office','R9-Washington Office',
                                             'PNW Research Station FIA','R1-Pacific Regional Office'],
        'Region 01 Northern Rockies Region':['Montana/Dakotas State Office', 'Montana'], 
        'Region 04 Intermountain Region':['Nevada State Office','DOI-OAS - Headquarters Boise', 'Intermountain Regional NPS Headquarters',
                                         'Utah','Utah State Office', 'Nevada', 'National Interagency Fire Center', 'DOI-OAS - Western Region Office',
                                         'DOI-OAS - Technical Services', 'DOI-OAS - Unmanned Aircraft System Office'],
        'Region 03 Southwest Region':['Southeast Region','Arizona State Office','Arizona','New Mexico','New Mexico State Office'],
        'Region 09 Eastern Area Region':['New Jersey', 'Pennsylvania', 'Minnesota','Pennsylvania','Midwest Regional NPS Headquarters','R3-Great Lakes - Big Rivers Regional Office',
                                        'Northeast Regional Office', 'DOI-OAS - Eastern Region Office','Northeastern Area, S&PF','Wisconsin','Eastern States Office','National Capitol Parks'],
        'Region 02 Rocky Mountain Region':['Colorado State Office', 'Colorado','R6-Mountain-Praire Regional Office', 'DMBM - Migratory Birds', 'Wyoming', 'South Dakota', 'Nebraska'], 
        'Region 08 Southern Area Region':['Gulf of Mexico Region','Texas','Oklahoma','North Carolina', 'South Carolina','Florida','R4-Southeast Regional Office', 'Tennessee', 'Georgia', 'Louisiana', 'Virginia'],
        'Region 10 Alaska Region': ['Alaska State Office','Alaska OCS Region', 'Alaska Regional Office', 'Alaska','DOI-OAS - Alaska Regional Office',
                                   'R7-Alaska Regional Office']

    }
    regions_corrected = []
    indices_to_drop = []
    for i in range(len(df)):
        current_region = df.iloc[i]['Region']
        correct_region = None
        for region in region_dict:
            if (region == current_region) or (current_region in region_dict[region]):
                correct_region = region
                break 
        if not correct_region:
            correct_region = current_region
        if correct_region in ['CAMP â€“ Campaign Against Marijuana Program', 'Commercial Aircraft Services', 'Aircraft Operations Center','National Capitol Parks', 'Department of Defense', 'National Guard']:
            indices_to_drop.append(i)
        if correct_region == 'Idaho':
            if df.iloc[i]['Location'] in ['Clear Creak Fire', 'garden valley', 'Idaho City', 'Idaho City Helibase (U98)','Ranft Fire']:
                correct_region = 'Region 04 Intermountain Region'
            else:
                correct_region = 'Region 01 Northern Rockies Region'
        elif correct_region == 'Pacific Region':
            correct_region = 'Region 05 Pacific Southwest Region'
        elif correct_region == 'Idaho State Office':
            if df.iloc[i]['Location'] in ['Granite Creek Fire', '47.512775, -116.002886','Post Falls, ID']:
                correct_region = 'Region 01 Northern Rockies Region'
            else: 
                correct_region = 'Region 04 Intermountain Region'
        elif correct_region == 'Wyoming State Office':
            if df.iloc[i]['Location'] in ['Tokewanna Fire','Rock Springs Sweetwater County', 'Evanston, WY Airport']:
                correct_region = 'Region 04 Intermountain Region'
            else:
                correct_region = 'Region 02 Rocky Mountain Region'
        regions_corrected.append(correct_region)

    df['region_corrected'] = regions_corrected
    df = df.drop(indices_to_drop, axis=0).reset_index(drop=True)
    return df

def get_categories_from_docs(docs, preprocessed_df, id_field, category_fields = ['Hazard', 'UAS', 'Accident', 'Airspace', 'Maintenance', 'Mishap Prevention']):
    """Gets the most common category and subcategory for safecom reports corresponding to specific hazards

    Parameters
    ----------
    docs :  Dict
        nested dictionary used to store documents per hazard. Keys are hazards 
        and value is an inner dict. Inner dict has keys as time variables (e.g., years) and 
        values are lists.
    preprocessed_df : pandas DataFrame
        pandas datframe containing documents.
    id_field : string
        the column in preprocessed df that contains document ids
    category_fields : list, optional
        List of columns in the preprocessed df that hold category information, by default ['Hazard', 'UAS', 'Accident', 'Airspace', 'Maintenance', 'Mishap Prevention']

    Returns
    -------
    categories : dict
        dictionary with keys as hazards and values as the corresponding higher level category
    sub_categories : dict
        dictionary with keys as hazards and values as the corresponding sub category
    """
    categories = {hazard: [] for hazard in docs}
    sub_categories = {hazard: [] for hazard in docs}
    for hazard in docs:
        ids = [id_ for year in docs[hazard] for id_ in docs[hazard][year]]
        hazard_df = preprocessed_df.loc[preprocessed_df[id_field].isin(ids)].reset_index(drop=True)
        #main category is the category with the most reports
        cat_counts = {}
        for cat in category_fields:
            cat_counts[cat] = len(hazard_df.loc[hazard_df[cat]!=""])
        main_category = max(cat_counts, key=cat_counts.get)
        categories[hazard] = main_category
        #subcategories are the most common value in that catgeory
        if len(hazard_df[main_category].value_counts()) == 0:
            sub_categories[hazard] = ""
        else:
            sub_cats = hazard_df[main_category].tolist()
            sub_cats = [cat for sub_cat in sub_cats for cat in str(sub_cat).split(", ")]
            sub_cats =[cat for cat in sub_cats if cat != '']
            sub_cat_series = pd.Series(sub_cats).dropna()
            sub_categories[hazard] = sub_cat_series.value_counts().idxmax(axis = 0)
    return categories, sub_categories

def create_table(docs, frequency, preprocessed_df, id_field, categories, subcategories, hazards, time_field):
    """_summary_

    Parameters
    ----------
    docs :  Dict
        nested dictionary used to store documents per hazard. Keys are hazards 
        and value is an inner dict. Inner dict has keys as time variables (e.g., years) and 
        values are lists.
    frequency : Dict
        Nested dictionary used to store hazard frequencies. Keys are hazards and inner dict keys are years, values are ints.
    preprocessed_df : pandas DataFrame
        pandas datframe containing documents.
    id_field : string
        the column in preprocessed df that contains document ids
    categories : dict
        dictionary with keys as hazards and values as the corresponding higher level category
    sub_categories : dict
        dictionary with keys as hazards and values as the corresponding sub category
    hazards : list
        list of hazards
    time_field : string
        the column in preprocessed df that contains document time values, such as report year

    Returns
    -------
    table : Pandas DataFrame
        dataframe containing the primary results table, including hazards sorted by categories, hazard
        rate, severity, and frequency
    severities : dict
        dictionary with keys as hazards and values as average severity
    rates : dict
        dictionary with keys as hazards and values as average rate of occurrence
    """
    table = pd.DataFrame({"Category": categories, "Subcategory": subcategories, "Hazards": hazards})
    time_period = preprocessed_df[time_field].unique()
    severities = {name:{str(time_p):[] for time_p in time_period} for name in hazards}
    rates = {name:{str(time_p):0 for time_p in time_period} for name in hazards}
    total_docs_per_year = preprocessed_df[time_field].value_counts()
    total_rates = {hazard:0 for hazard in hazards}
    total_hazard_freq = {hazard:0 for hazard in hazards}
    total_severities_hazard = {hazard:0 for hazard in hazards}
    for hazard in hazards:
        for year in docs[hazard]:
            year = str(year)
            for doc in docs[hazard][year]:
                doc_df = preprocessed_df.loc[preprocessed_df[id_field] == doc].reset_index(drop=True)
                severities[hazard][year].append(doc_df.at[0, 'severity']) #safecom_severity(doc_df.iloc[0]['Persons Onboard'], doc_df.iloc[0]['Injuries'], doc_df.iloc[0]['Damages']))
            rates[hazard][year] = frequency[hazard][year]/total_docs_per_year[year]
            total_hazard_freq[hazard] += frequency[hazard][year]
            if severities[hazard][year]==[]:
                severities[hazard][year] = [0]
        total_rates[hazard] = round(total_hazard_freq[hazard]/len(time_period), 3)
        total_severities_hazard[hazard] = round(np.average([sev for year in severities[hazard] for sev in severities[hazard][year]]),3)
    table["Frequency"] = [total_hazard_freq[hazard] for hazard in total_hazard_freq]
    table["Rate"] = [total_rates[hazard] for hazard in total_rates]
    table["Severity"] = [total_severities_hazard[hazard] for hazard in total_severities_hazard]
    return table, severities, rates