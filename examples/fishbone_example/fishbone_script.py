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

asrs_filepath_original = os.path.join("examples","fishbone_example","ASRS_DBOnline.csv")
asrs_filepath_edit = os.path.join("examples","fishbone_example","ASRS_DBOnline_edit.csv")

# preprocess - merge first two rows to get column names
asrs_df = pd.read_csv(asrs_filepath_original, header=[0,1])
asrs_df.columns = asrs_df.columns.map(' '.join)
asrs_df.rename(columns={'  ACN':'ACN'}, inplace=True)
asrs_df.to_csv(asrs_filepath_edit)

# load into Data()
asrs_data = Data()
asrs_text_columns = ['Report 1 Narrative', 'Report 2 Narrative']
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

queries = ['what happens when uas goes out of bounds', 'what are the risks of battery operated uas', 'what if uas flies over people on ground']
for query in queries:
    print(ir_asrs.run_search(query,return_k=5))

# NER
# NER for FMEA
model_checkpoint = os.path.join("models", "FMEA-ner-model", "checkpoint-1424")
device = 'cuda' if cuda.is_available() else 'cpu'
cuda.empty_cache()

fmea = FMEA()
fmea.load_model(model_checkpoint)
input_data = fmea.load_data('Report 1 Narrative', asrs_document_id_col, filepath=asrs_filepath_edit, formatted=False)

preds = fmea.predict()
df = fmea.get_entities_per_doc()
fmea.group_docs_with_meta(grouping_col='Events Anomaly')
fmea.grouped_df.to_csv(os.path.join(os.getcwd(),"asrs_fmea.csv"))