"""
hswalsh, srandrad
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),"..",".."))
import numpy as np
import pandas as pd
from mika.utils import Data
from mika.ir import search
from datetime import datetime as dt
from mika.kd.topic_model_plus import Topic_Model_plus
from mika.kd.trend_analysis import plot_frequency_time_series, plot_metric_time_series, plot_metric_averages, make_pie_chart, chi_squared_tests, plot_risk_matrix, get_likelihood_FAA, identify_docs_per_fmea_row, calc_severity_per_hazard, add_hazards_to_docs, chi_squared_tests
from mika.kd import FMEA
from sklearn.feature_extraction.text import CountVectorizer
from torch import cuda
from sentence_transformers import SentenceTransformer
from torch import cuda
import matplotlib as mpl
import matplotlib.pyplot as plt

# functions

#definted according to FAA order 8040.4B: https://www.faa.gov/documentLibrary/media/Order/FAA_Order_8040.4B.pdf
def NTSB_severity(damage, inj_level, inj_tot_f, persons_onboard): #damage, ev_highest_injury, inj_tot_f	
    if float(persons_onboard) == 0:
        persons_onboard = inj_tot_f
    if inj_tot_f != 0 :
        pct_fatal = inj_tot_f/persons_onboard
    else:
        pct_fatal = 0
    #minimal: no injuries, no damage. 
    if inj_level == 'NONE' and damage == 'UKN':
        severity = 'Minimal'
    #minor: slight (MINR) damage, physical discomfort
    elif inj_level == 'MINR' or inj_level == 'NONE':
        #major: substaintail (SUBS) damage, injuries
        if damage == 'SUBS':
            severity = 'Major' 
        elif damage == 'DEST':
            severity = 'Hazardous'
        else:
            severity = 'Minor'
    #hazardous: multiple serious injuries, fatalities<2, hull loss (DEST)
    elif inj_level == 'SERS' or (inj_level == 'FATL' and (inj_tot_f <= 2 or pct_fatal < 0.75)) or damage == 'DEST':
        severity = 'Hazardous'
    #catatrophic: fatalities > 2, or num person on board= num fatalities,  hull loss (DEST)
    elif inj_level == 'FATL' and (inj_tot_f > 2 or pct_fatal > 0.75):
        severity = 'Catastrophic'
    else: 
        severity = 'Minimal' #case with no reported fatalities, injury level, or damage level
    return severity

def severity_func(df, numeric=True):
    severity_dict = {'Catastrophic':5, 'Hazardous': 4, 'Major':3, 'Minor':2, 'Minimal':1}
    severities = []
    for i in range(len(df)):
        severities.append(NTSB_severity(df.at[i, 'damage'], df.at[i, 'ev_highest_injury'], float(df.at[i, 'inj_tot_f'].replace('','0')), float(df.at[i, 'total_seats'].replace('','0'))))
    if numeric == True:
        df['severity'] = [severity_dict[severity] for severity in severities]
    else:
        df['severity'] = severities
    return df

# main function - this is what runs

if __name__ == "__main__":

    # load in data
    ntsb_filepath = os.path.join("data/NTSB/ntsb_full.csv")
    ntsb_text_columns = ['narr_cause', 'narr_accf'] # narrative accident cause and narrative accident final
    ntsb_document_id_col = 'ev_id'
    ntsb_database_name = 'NTSB'
    ntsb_data = Data()
    ntsb_data.load(ntsb_filepath, preprocessed=False, id_col=ntsb_document_id_col, text_columns=ntsb_text_columns.copy(), name=ntsb_database_name, load_kwargs={'dtype':str})
    ntsb_data.prepare_data(create_ids=False, combine_columns=ntsb_text_columns.copy(), remove_incomplete_rows=False)

    # filter data for years of interest

    years_of_interest = ['2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021']
    ntsb_data.data_df = ntsb_data.data_df.loc[ntsb_data.data_df['ev_year'].isin(years_of_interest)].drop_duplicates(subset='ev_id', keep="last").reset_index(drop=True) # keep the last record, this one has the phase of flight and mishap right before the accident
    ntsb_data._Data__update_ids()

    # IR
    # there are options here to use pretrained or finetuned models - comment out appropriate lines as needed
    model = SentenceTransformer("NASA-AIML/MIKA_Custom_IR")
    ir_ntsb = search('narr_cause', ntsb_data, model)
    embeddings_path = os.path.join('data', 'NTSB', 'ntsb_sentence_embeddings_finetune.npy')
    ir_ntsb.get_sentence_embeddings(embeddings_path) # comment this out if the embeddings already exist
    #ir_ntsb.load_sentence_embeddings(embeddings_path) # uncomment this if you wish to load sentence embeddings that already exist

    queries = ['what components are vulnerable to fatigue crack', 'what are the consequences of a fuel leak', 'what are the risks of low visibility']
    for query in queries:
        print(query)
        print(ir_ntsb.run_search(query, return_k=10, rank_k=10))

    # bert topics
    tm = Topic_Model_plus(text_columns=ntsb_text_columns, data=ntsb_data, results_path=os.path.join(os.getcwd(),"examples/Case_Study_NTSB"))
    vectorizer_model = CountVectorizer(ngram_range=(1, 3), stop_words="english") #removes stopwords
    BERTkwargs={"top_n_words": 10, 'min_topic_size':25}
    tm.bert_topic(count_vectorizor=vectorizer_model, BERTkwargs=BERTkwargs, from_probs=False)
    tm.save_bert_model()
    tm.save_bert_results(from_probs=False) 
    tm.save_bert_taxonomy() #saves taxonomy #takes 15 min to run

    # NER for FMEA
    # - rows: narratives table
    # - severities: injuries table
    # prepare the df for FMEA - drop repeat rows
    # fmea_input_df = ntsb_data.data_df.copy()
    # fmea_input_df = fmea_input_df.loc[~(fmea_input_df['Combined Text']=='')].reset_index(drop=True) # remove docs with no text

    # model_checkpoint = "NASA-AIML/MIKA_BERT_FMEA_NER"

    # device = 'cuda' if cuda.is_available() else 'cpu'
    # cuda.empty_cache()
    # print(device)

    # fmea = FMEA() # initialize FMEA object
    # fmea.load_model(model_checkpoint) # load the NER model
    # print("loaded model")
    # input_data = fmea.load_data('Combined Text','ev_id', df=fmea_input_df, formatted=False) # load the data
    # print("loaded data")
    # preds = fmea.predict() # get failure modes, effects, causes, etc.
    # df = fmea.get_entities_per_doc() # get the entites for each document
    # fmea.group_docs_with_meta(grouping_col='Mishap Category', additional_cols=['Phase']) # group documents according to mishap category
    # fmea.grouped_df.to_csv(os.path.join(os.getcwd(),"examples/Case_Study_NTSB/ntsb_fmea_raw.csv")) # save raw FMEA results
    # fmea.calc_severity(severity_func, from_file=False) # calculate severity
    # fmea.calc_frequency(year_col="ev_year") # calculate frequency 
    # fmea.calc_risk() # calculate risk according to FAA
    # fmea.post_process_fmea(phase_name='Phase', id_name='NTSB', max_words=10) # post process results
    # fmea.fmea_df.astype(str).to_csv(os.path.join(os.getcwd(),"examples/Case_Study_NTSB/NTSB_FMEA.csv")) # save results

    # mishaps_of_interest = ['Birdstrike','Midair collision', 'Loss of control in flight','Loss of control on ground','Turbulence encounter', 'Ground collision'] # filter for specific mishaps
    # fmea.fmea_df.loc[fmea.fmea_df.index.isin(mishaps_of_interest)].astype(str).to_csv(os.path.join(os.getcwd(),"examples/Case_Study_NTSB/NTSB_FMEA_truncated.csv")) # save filtered results

    # HEAT - Trend Analysis
    # - involves generating plots, statistical tests, and a risk matrix

    frequency, docs_per_row = identify_docs_per_fmea_row(ntsb_data.data_df, 'Mishap Category', 'ev_year', 'ev_id') # get documents for each fmea row

    mishaps_of_interest = ['Birdstrike','Midair collision', 'Loss of control in flight','Loss of control on ground','Turbulence encounter', 'Ground collision']
    years_of_interest = ['2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021']
    frequency_set = {mishap:{year: frequency[mishap][year] for year in years_of_interest} for mishap in mishaps_of_interest} # filter data
    doc_set = {mishap:{year: docs_per_row[mishap][year] for year in years_of_interest} for mishap in mishaps_of_interest} # filter data

    plot_frequency_time_series(frequency_set, metric_name='Frequency', line_styles=[], 
                                markers=[], title="Frequency from 2011-2021", time_name="Year", xtick_freq=2, 
                                scale=False, save=False, yscale='log', legend=True,
                                figsize=(6,4), fontsize=24) # plot hazard frequency over time

    df_with_severities = severity_func(ntsb_data.data_df, numeric=True) # calculate severity per document
    severities, total_severities_hazard = calc_severity_per_hazard(doc_set, df_with_severities, 'ev_id', metric='max') # calculate severity per hazard
    plot_metric_time_series(metric_data=severities, metric_name="Severity", title="Average Severity from 2011-2021", time_name='Year', xtick_freq=2, show_std=False, save=False, figsize=(6,4), fontsize=24) # plot hazard severity
    plot_metric_averages(metric_data=severities, metric_name="Severity", show_std=True, title="Average Severity", save=False, legend=False, figsize=(8,4),fontsize=24) # plot average hazard severity

    sky_conditions_dict = {'CLER': 'clear', # map sky conditions to english description
                        '':'unknown', 
                        'SCAT': 'scattered', 
                        'FEW': 'few', 
                        'UNK': 'unknown', 
                        'OVCT': 'thin overcast', 
                        'OVC': 'overcast',
                        'OBSC': 'obscured',
                        'BKNT': 'thin broken', 
                        'BKN': 'broken',
                        'POBS': 'partially obscured',
                        'VV': 'indefinite',
                        'NONE': 'none'}
    df_with_severities['sky_cond_nonceil_cleaned'] = [sky_conditions_dict[cond] for cond in df_with_severities['sky_cond_nonceil'].tolist()]
    df_with_severities['flight_plan_activated_cleaned'] = [val if val!='' else 'U' for val in df_with_severities['flight_plan_activated'].tolist()] # fill missing flight plans with unknown
    make_pie_chart(doc_set, df_with_severities, 'flight_plan_activated_cleaned', mishaps_of_interest, 'ev_id', 'Activated Flight Plan', save=False, fontsize=12, pie_kwargs={'pctdistance':1.27}, figsize=(6,4), padding=10) # flight plan plot
    make_pie_chart(doc_set, df_with_severities, 'sky_cond_nonceil_cleaned', mishaps_of_interest, 'ev_id', 'Sky Condition', save=False, fontsize=12, pie_kwargs={'pctdistance':1.27}, figsize=(6,4), padding=10) # sky condition plot

    df_with_hazards = add_hazards_to_docs(df_with_severities, id_field='ev_id', docs=doc_set) # add columns labeling the hazards in each document
    stats_df, counts_dfs = chi_squared_tests(df_with_hazards, mishaps_of_interest, predictors=['flight_plan_activated_cleaned','sky_cond_nonceil_cleaned'], pred_dict={'flight_plan_activated_cleaned':'flight plan','sky_cond_nonceil_cleaned': 'sky condition'}) # conduct chi-squared tests to see if hazard occurence varies on conditions
    print(stats_df)

    # for df in counts_dfs: # uncomment to print the residuals for the chi-squared test
    #     print(counts_dfs[df])

    severity_dict = {5:'Catastrophic Impact', 4:'Hazardous Impact', 3:'Major Impact', 2:'Minor Impact', 1:'Minimal Impact'}
    rm_severities = {hazard: severity_dict[round(total_severities_hazard[hazard])] for hazard in total_severities_hazard}
    rates = {hazard: sum([frequency_set[hazard][year] for year in frequency_set[hazard]])/len(years_of_interest) for hazard in frequency_set}
    rm_likelihoods = get_likelihood_FAA(rates)
    mpl.rcParams.update(mpl.rcParamsDefault) # set plot formatting
    plt.rcParams["font.family"] = "Times New Roman" # set plot formatting

    plot_risk_matrix(rm_likelihoods, rm_severities, figsize=(8,4), fontsize=12, max_chars=20, annot_font=10)