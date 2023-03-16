import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),"..",".."))
import numpy as np
import pandas as pd
from mika.utils import Data
from mika.ir import search
from datetime import datetime as dt
from mika.kd.topic_model_plus import Topic_Model_plus
from mika.kd import FMEA
from sklearn.feature_extraction.text import CountVectorizer
from torch import cuda
from sentence_transformers import SentenceTransformer

asrs_filepath_original = os.path.join("examples","fishbone_example","ASRS_DBOnline_v2.csv")
asrs_filepath_edit = os.path.join("examples","fishbone_example","ASRS_DBOnline_v2_edit.csv")

# preprocess - merge first two rows to get column names
asrs_df = pd.read_csv(asrs_filepath_original, header=[0,1])
asrs_df.columns = asrs_df.columns.map(' '.join)
asrs_df.rename(columns={'  ACN':'ACN'}, inplace=True)
asrs_df.to_csv(asrs_filepath_edit)

# load into Data()
asrs_data = Data()
asrs_text_columns = ['Report 1 Narrative']
asrs_document_id_col = 'ACN'
asrs_database_name = 'ASRS UAS Reports'
asrs_data.load(asrs_filepath_edit, preprocessed=False, text_columns=asrs_text_columns, name=asrs_database_name, load_kwargs={'dtype':str})
asrs_data.prepare_data(create_ids=True, combine_columns=asrs_text_columns, remove_incomplete_rows=False)

# bert topics - might be good for identifying general categories to use as branches
tm = Topic_Model_plus(text_columns=asrs_text_columns, data=asrs_data)
vectorizer_model = CountVectorizer(ngram_range=(1, 3), stop_words="english") #removes stopwords
tm.bert_topic(sentence_transformer_model=None, umap=None, hdbscan=None, count_vectorizor=vectorizer_model, ngram_range=(1,3), BERTkwargs={}, from_probs=False, thresh=0.01)
tm.save_bert_model()

BERTkwargs={"top_n_words": 10, 'min_topic_size':5}
tm.bert_topic(count_vectorizor=vectorizer_model, BERTkwargs=BERTkwargs, from_probs=True)
tm.save_bert_results(from_probs=True)
tm.save_bert_taxonomy()

# IR - might be good for filling out branches once you have an idea of general categories
model = SentenceTransformer('all-distilroberta-v1')
ir_asrs = search(asrs_data, model)
embeddings_path = os.path.join('data', 'asrs_sentence_embeddings_finetune.npy')
ir_asrs.get_sentence_embeddings(embeddings_path)

queries = [
    'do bird strikes affect uas',
    'are there issues with drones landing on certain surfaces',
    'do drones ever collide with aircraft',
    'do drones ever interfere with landing',
    'do drones interfere with weather balloons',
    'is evasive action required when a drone flies near an aircraft',
    'what near misses have happened with uas',
    'when there are waivers for drones, are there additional risks to the mission',
    'are drones a risk to helicopters',
    'what happens when there is a lost link with a drone',
    'is drone battery reliable for the mission',
    'what if there is a lost link with a drone',
    'what if a pilot loses contact with a drone',
    'do drones interfere with takeoff',
    'do drones fly above 400 feet above ground level',
    'do drones fly in restricted areas',
    'do drones collide with terrain',
    'do drones fly without waivers when a waiver is required',
    'do drones have difficulties landing on certain surfaces',
    'have drones collided with buildings',
    'have drones collided with trees',
    'have drones collided with objects'
    ]
search_result = pd.DataFrame()
for query in queries:
    new_search_result = ir_asrs.run_search(query,return_k=10)
    query_row = pd.DataFrame([query], columns=['query'])
    new_search_result = pd.concat([query_row, new_search_result], axis=0)
    search_result = pd.concat([search_result, new_search_result], axis=0)
search_result.to_csv('IR_result.csv')

# NER
# NER for FMEA
# model_checkpoint = os.path.join("models", "FMEA-ner-model", "checkpoint-1424")
# device = 'cuda' if cuda.is_available() else 'cpu'
# cuda.empty_cache()

# fmea = FMEA()
# fmea.load_model(model_checkpoint)
# input_data = fmea.load_data('Report 1 Narrative', asrs_document_id_col, filepath=asrs_filepath_edit, formatted=False)

# preds = fmea.predict()
# df = fmea.get_entities_per_doc()
# fmea.group_docs_with_meta(grouping_col='Events Anomaly')
# fmea.grouped_df.to_csv(os.path.join(os.getcwd(),"asrs_fmea.csv"))