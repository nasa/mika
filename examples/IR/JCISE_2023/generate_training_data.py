"""
@author: hswalsh
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),"..","..",".."))
import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# load llis data and extract just a list of the text portions for training
llis_docs_df = pd.read_excel(os.path.join('data', 'LLIS','lessons_learned_2021-12-10.xlsx'))
llis_docs = llis_docs_df['Abstract'].to_list() + llis_docs_df['Recommendation(s)'].to_list() + llis_docs_df['Driving Event'].to_list() + llis_docs_df['Lesson(s) Learned'].to_list()

# preprocess docs
llis_docs = [doc for doc in llis_docs if str(doc) != 'nan' and len(str(doc))>64]

# setup tokenizer and model to generate queries
tokenizer = T5Tokenizer.from_pretrained('BeIR/query-gen-msmarco-t5-large-v1')
model = T5ForConditionalGeneration.from_pretrained('BeIR/query-gen-msmarco-t5-large-v1')
model.eval()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# generate queries
query_list = []
doc_list = []
for llis_doc in llis_docs:
    input = tokenizer.encode(llis_doc, max_length=300, truncation=True, return_tensors='pt').to(device)
    outputs = model.generate(
        inputs=input,
        max_length=64,
        do_sample=True,
        top_p=0.95,
        num_return_sequences=3)

    for i in range(3):
        query_list.append(tokenizer.decode(outputs[i], skip_special_tokens=True))
        doc_list.append(llis_doc)

# save queries to file
training_data = pd.DataFrame(data={'Query': query_list, 'Doc': doc_list})
training_data.to_csv('examples/IR/JCISE_2023/training_data.csv')