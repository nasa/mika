"""
hswalsh
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),"..",".."))
import numpy as np
import pandas as pd
from mika.utils import Data
from mika.ir import search
from datetime import datetime as dt
from mika.kd.topic_model_plus import Topic_Model_plus
from mika.kd import trend_analysis
from mika.kd import NER
from sklearn.feature_extraction.text import CountVectorizer

# before loading data, grab only recent reports and save as a new csv
ntsb_filepath = os.path.join("data/NTSB/ntsb_full.csv")
ntsb_df = pd.read_csv(ntsb_filepath)
datetimes = [dt.strptime(date_str, "%Y-%m-%d %H:%M:%S") for date_str in ntsb_df['lchg_date']]
ntsb_df['datetimes'] = datetimes
ntsb_df = ntsb_df.loc[ntsb_df['datetimes'] >= dt.strptime("09/01/2022", "%m/%d/%Y")]
ntsb_recent_filepath = os.path.join("data/NTSB/ntsb_recent_full.csv")
ntsb_df.to_csv(ntsb_recent_filepath)

# now load into Data()
ntsb_data = Data()
ntsb_text_columns = ['narr_accf'] # narrative accident final and narrative accident cause
ntsb_document_id_col = 'ev_id'
ntsb_database_name = 'NTSB'
ntsb_data.load(ntsb_recent_filepath, preprocessed=False, text_columns=ntsb_text_columns, id_col=ntsb_document_id_col, name=ntsb_database_name,preprocessed_kwargs={'dtype':str}) # way to load as str?
ntsb_data.prepare_data(create_ids=True, combine_columns=ntsb_text_columns, remove_incomplete_rows=False)
    
# IR
# there are options here to use pretrained or finetuned models - comment out appropriate lines as needed
model = os.path.join('models', 'fine_tuned_llis_model')
#model = 'all-distilroberta-v1'
query = 'fatigue crack'
ir_ntsb = search(ntsb_data, model)
embeddings_path = os.path.join('data', 'LLIS', 'llis_sentence_embeddings_finetune.npy')
#embeddings_path = os.path.join('data', 'LLIS', 'llis_sentence_embeddings.npy')
#ir_ntsb.get_sentence_embeddings(embeddings_path) # comment this out if the embeddings already exist
ir_ntsb.load_sentence_embeddings(embeddings_path)
print(ir_ntsb.run_search(query,return_k=5))

# taxonomy
# - narr_cause and narr_accf/narr_accp - these are in ntsb_full_narratives.csv
tm = Topic_Model_plus(text_columns=ntsb_text_columns, data=ntsb_data)
vectorizer_model = CountVectorizer(ngram_range=(1, 3), stop_words="english") #removes stopwords
#tm.bert_topic(sentence_transformer_model=None, umap=None, hdbscan=None, count_vectorizor=vectorizer_model, ngram_range=(1,3), BERTkwargs={}, from_probs=False, thresh=0.01)
#tm.save_bert_model()

BERTkwargs={"calculate_probabilities":True, "top_n_words": 20, 'min_topic_size':150}
tm.bert_topic(count_vectorizor=vectorizer_model, BERTkwargs=BERTkwargs, from_probs=True)
tm.save_bert_results(from_probs=True)

# NER for FMEA
# - rows: event table
# - severities: injuries table
#
# HEAT
# - aircraft info table
