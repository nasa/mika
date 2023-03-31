"""
hswalsh
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),"..",".."))
from sentence_transformers import SentenceTransformer
from mika.ir.search import search
from mika.utils import Data
from datetime import datetime as dt
import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

filepath = os.path.join("data/LLIS/lessons_learned_2021-12-10.xlsx")
df = pd.read_excel(filepath)

# now load into Data()
data = Data()
text_columns = ['Abstract', 'Lesson(s) Learned', 'Recommendation(s)', 'Driving Event', 'Evidence']
document_id_col = 'Lesson ID'
database_name = 'LLIS'
data.load(filepath, preprocessed=False, text_columns=text_columns, name=database_name,preprocessed_kwargs={'dtype':str}, id_col = document_id_col)
data.prepare_data(create_ids=False, combine_columns=text_columns, remove_incomplete_rows=False)

model = SentenceTransformer('sentence-transformers/msmarco-roberta-base-v3')

query = 'Mars rover'

ir_llis = search('Combined Text', data, model)
ir_llis.get_sentence_embeddings(os.path.join('results'))
search_result = ir_llis.run_search(query,return_k=10)
print(search_result)