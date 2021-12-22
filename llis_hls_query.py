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

# treat paragraphs in llis as separate documents - paragraphs are more "requirement-sized"
llis_abstract = llis_df['Abstract'].to_list()
llis_lessons_learned = llis_df['Lesson(s) Learned'].to_list()
llis_recommendations = llis_df['Recommendation(s)'].to_list()
llis_driving_event = llis_df['Driving Event'].to_list()
llis_evidence = llis_df['Evidence'].to_list()
llis_corpus = llis_abstract + llis_lessons_learned + llis_recommendations + llis_driving_event + llis_evidence

# grab relevant metadata
llis_title = llis_df['Title'].to_list()
llis_title = llis_title + llis_title + llis_title + llis_title + llis_title
llis_phase = llis_df['From what phase of the program or project was this lesson learned captured?'].to_list()
llis_phase = llis_phase + llis_phase + llis_phase + llis_phase + llis_phase
llis_topics = llis_df['Topics'].to_list()
llis_topics = llis_topics + llis_topics + llis_topics + llis_topics + llis_topics

# get list of lesson id's aligned with corpus
llis_numbers = llis_df['Lesson ID'].to_list()
llis_numbers_corpus = llis_numbers + llis_numbers + llis_numbers + llis_numbers + llis_numbers

# get section names aligned with corpus
section_names = ['Abstract', 'Lesson(s) Learned', 'Recommendation(s)', 'Driving Event', 'Evidence']
sections = [llis_abstract, llis_lessons_learned, llis_recommendations, llis_driving_event, llis_evidence]
llis_sections = []
for i in range(0,len(sections)):
    section = sections[i]
    for j in range(0,len(section)):
        llis_sections.append(section_names[i])
        
# remove nans
def remove_nans(docs):
    for i in range(0,len(docs)):
        if pd.isnull(docs[i]):
            docs[i] = ''
    return docs
llis_corpus = remove_nans(llis_corpus)

# select query
query = requirements_df['Requirement'][1]

start_time = time.time() # start timer

# get query embedding
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
query_embedding = sbert_model.encode(query, convert_to_tensor=True)
query_embedding_time = time.time()
query_embedding_time_delta = query_embedding_time - start_time

# get corpus embeddings - eventually these should be pre-run and loaded for use in each query
corpus_embeddings = sbert_model.encode(llis_corpus, convert_to_tensor=True)
corpus_embeddings_time = time.time()
corpus_embeddings_time_delta = corpus_embeddings_time - start_time - query_embedding_time_delta

# semantic search
hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=5)
hits = hits[0] # Get the hits for the first query
search_time = time.time()
search_time_delta = search_time - start_time - query_embedding_time_delta - corpus_embeddings_time_delta
total_run_time = search_time - start_time # end timer

# save outputs
writer = csv.writer(open(os.path.join('results','llis_hls_query_results_1.csv'), "w"))
writer.writerow(['Query: ',query])
writer.writerow(['--------------------------------------'])
for hit in hits:
    writer.writerow(['Lesson ID: ', str(llis_numbers_corpus[hit['corpus_id']])])
    writer.writerow([' Title: ', str(llis_title[hit['corpus_id']])])
    writer.writerow([' LLIS-Designated Topics: ', str(llis_topics[hit['corpus_id']])])
    writer.writerow([' Phase: ', str(llis_phase[hit['corpus_id']])])
    writer.writerow([' Section: ', str(llis_sections[hit['corpus_id']])])
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
