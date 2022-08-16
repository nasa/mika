"""
@author: hswalsh
"""

import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
import csv
import time

# pick your model - comment out one
#model = SentenceTransformer('msmarco-roberta-base-v3') # this is the model we are starting with and would like to fine tune further; possibly check input sequence length as well
model = SentenceTransformer(os.path.join('results', 'fine_tuned_llis_model')) # fine tuned model

# load LLIS data for fine-tuning
llis_filename = os.path.join('data','LLIS','lessons_learned_2021-12-10.xlsx')
llis_df = pd.read_excel(llis_filename)

# remove nans and combine sections to create documents
def remove_nans(docs):
    for i in range(0,len(docs)):
        if pd.isnull(docs[i]):
            docs[i] = ''
    return docs
llis_abstract = remove_nans(llis_df['Abstract'].to_list())
llis_lessons_learned = remove_nans(llis_df['Lesson(s) Learned'].to_list())
llis_recommendations = remove_nans(llis_df['Recommendation(s)'].to_list())
llis_driving_event = remove_nans(llis_df['Driving Event'].to_list())
llis_evidence = remove_nans(llis_df['Evidence'].to_list())
llis_corpus = []
for i in range(0,len(llis_abstract)):
    llis_corpus.append(llis_abstract[i] + llis_lessons_learned[i] + llis_recommendations[i] + llis_driving_event[i] + llis_evidence[i])

# breakdown documents into sentences
sentence_corpus = []
for doc in llis_corpus:
    sentences = sent_tokenize(doc)
    for sentence in sentences:
        sentence_corpus.append(sentence)

start_time = time.time()

# get embeddings
sentence_corpus_embeddings = model.encode(sentence_corpus, convert_to_tensor=True)
end_time = time.time()
total_run_time = end_time - start_time

def print_runtime(run_time):
    if run_time < 60:
        print("--- %s seconds ---" % (run_time))
    else:
        print("--- %s minutes ---" % (run_time/60))

print('Total run time: ')
print_runtime(total_run_time)

# save - recommend giving a different name for finetuned embeddings
embeddings_as_numpy = sentence_corpus_embeddings.numpy()
np.save(os.path.join('data', 'LLIS', 'llis_sentence_embeddings_finetune.npy'), embeddings_as_numpy)
