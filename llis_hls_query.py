"""
@author: hswalsh
"""

import os
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import time
import csv

# load data
llis_filename = os.path.join('data','lessons_learned_2021-12-10.xlsx')
llis_df = pd.read_excel(llis_filename)
requirements_filename = os.path.join('data','functional_performance_requirements_docs.csv')
requirements_df = pd.read_csv(requirements_filename)

# remove nans
def remove_nans(docs):
    for i in range(0,len(docs)):
        if pd.isnull(docs[i]):
            docs[i] = ''
    return docs

# combine sections to create documents
llis_abstract = remove_nans(llis_df['Abstract'].to_list())
llis_lessons_learned = remove_nans(llis_df['Lesson(s) Learned'].to_list())
llis_recommendations = remove_nans(llis_df['Recommendation(s)'].to_list())
llis_driving_event = remove_nans(llis_df['Driving Event'].to_list())
llis_evidence = remove_nans(llis_df['Evidence'].to_list())
llis_corpus = []
for i in range(0,len(llis_abstract)):
    llis_corpus.append(llis_abstract[i] + llis_lessons_learned[i] + llis_recommendations[i] + llis_driving_event[i] + llis_evidence[i])

# grab relevant metadata
llis_title = llis_df['Title'].to_list()
llis_phase = llis_df['From what phase of the program or project was this lesson learned captured?'].to_list()
llis_topics = llis_df['Topics'].to_list()

# get list of lesson id's aligned with corpus
llis_numbers = llis_df['Lesson ID'].to_list()
llis_numbers_corpus = llis_numbers

# select query
print('Relevant Requirement: ', requirements_df['Requirement'][0])
query = 'cyber security measures for data and systems' # do not search full requirement - use short phrase
print('Query: ', query)

start_time = time.time() # start timer

# sbert model
#sbert_model = SentenceTransformer('msmarco-roberta-base-v3') # this model is for asymmetric search and is tuned for cosine similarity (models tuned for cosine similarity will prefer retrieval of shorter documents - models trained for dot product will prefer retrieval of longer documents); this is a roberta model as opposed to pure bert; can be replaced with fine tuned model
sbert_model = SentenceTransformer(os.path.join('results', 'fine_tuned_llis_model')) # fine tuned model

# get query embedding
query_embedding = sbert_model.encode(query, convert_to_tensor=True)
query_embedding_time = time.time()
query_embedding_time_delta = query_embedding_time - start_time

# load sentence embeddings - these are obtained in the file get_corpus_embeddings.py - the runtime for this is significant so we save them after obtaining
embeddings_as_numpy = np.load(os.path.join('data', 'llis_sentence_embeddings.npy'))
corpus_embeddings_original = torch.from_numpy(embeddings_as_numpy)
corpus_embeddings_time = time.time()
corpus_embeddings_time_delta = corpus_embeddings_time - start_time - query_embedding_time_delta

# semantic search
hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=5)
hits = hits[0] # Get the hits for the first query
search_time = time.time()
search_time_delta = search_time - start_time - query_embedding_time_delta - corpus_embeddings_time_delta
total_run_time = search_time - start_time # end timer

# save outputs
writer = csv.writer(open(os.path.join('results','llis_hls_query_results_0.csv'), "w"))
writer.writerow(['Query: ',query])
writer.writerow(['--------------------------------------'])
for hit in hits:
    writer.writerow(['Lesson ID: ', str(llis_numbers_corpus[hit['corpus_id']])])
    writer.writerow([' Title: ', str(llis_title[hit['corpus_id']])])
    writer.writerow([' LLIS-Designated Topics: ', str(llis_topics[hit['corpus_id']])])
    writer.writerow([' Phase: ', str(llis_phase[hit['corpus_id']])])
    writer.writerow(["Score: {:.4f}".format(hit['score'])])
    writer.writerow([llis_corpus[hit['corpus_id']]])

def print_runtime(run_time):
    if run_time < 60:
        print("--- %s seconds ---" % (run_time))
    else:
        print("--- %s minutes ---" % (run_time/60))

print('Query embedding run time: ')
print_runtime(query_embedding_time_delta)
print('Corpus embedding run time: ')
print_runtime(corpus_embeddings_time_delta)
print('Semantic search run time: ')
print_runtime(search_time_delta)
print('Total run time: ')
print_runtime(total_run_time)
