"""
hswalsh
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),"..",".."))
from sentence_transformers import SentenceTransformer
from mika.ir.custom_ir_model import custom_ir_model
from mika.ir.search import search
from mika.utils import Data
from datetime import datetime as dt
import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# before loading data, grab only recent reports and save as a new csv
#ntsb_filepath = os.path.join("data/NTSB/ntsb_full.csv")
ntsb_filepath = os.path.join("data/NTSB/ntsb_recent_full.csv")
ntsb_df = pd.read_csv(ntsb_filepath)
datetimes = [dt.strptime(date_str, "%Y-%m-%d %H:%M:%S") for date_str in ntsb_df['lchg_date']]
ntsb_df['datetimes'] = datetimes
ntsb_df = ntsb_df.loc[ntsb_df['datetimes'] >= dt.strptime("09/01/2022", "%m/%d/%Y")]
ntsb_df.to_csv(ntsb_filepath)

# now load into Data()
ntsb_data = Data()
ntsb_text_columns = ['narr_cause', 'narr_accf'] # narrative accident cause and narrative accident final
ntsb_document_id_col = 'id'
ntsb_database_name = 'NTSB'
ntsb_data.load(ntsb_filepath, preprocessed=False, text_columns=ntsb_text_columns, name=ntsb_database_name,preprocessed_kwargs={'dtype':str}) # way to load as str?
ntsb_data.prepare_data(create_ids=True, combine_columns=ntsb_text_columns, remove_incomplete_rows=False)

# using custom_ir_model to setup the training data and fine-tune the model
model = SentenceTransformer('sentence-transformers/msmarco-roberta-base-v3')
ntsb_custom_ir_model = custom_ir_model(base_model=model, training_data=ntsb_data)

embeddings_path = os.path.join('data', 'NTSB', 'ntsb_sentence_embeddings_finetune.npy')
ntsb_custom_ir_model.load_sentence_embeddings(embeddings_path)
tokenizer = T5Tokenizer.from_pretrained('BeIR/query-gen-msmarco-t5-large-v1')

t5_model = T5ForConditionalGeneration.from_pretrained('BeIR/query-gen-msmarco-t5-large-v1')

training_data_filepath = os.path.join('data','NTSB','ir_model_training_data.csv')
ntsb_custom_ir_model.prepare_training_data(tokenizer, t5_model, training_data_filepath)
ntsb_custom_ir_model.fine_tune_model(data_filepath=training_data_filepath, train_batch_size=16, model=model, num_epochs=3, model_name='custom_model')

# quick check of semantic search using the fine tuned model
custom_model = SentenceTransformer(os.path.join('results', 'custom_model'))
query = 'what are common UAS mishaps'

ir_ntsb = search(ntsb_data, custom_model)
ir_ntsb.get_sentence_embeddings(os.path.join('results', 'ntsb_embeddings_custom_model.npy'))
print(ir_ntsb.run_search(query,return_k=5))
