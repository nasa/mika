import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, losses, InputExample, util
import torch
from nltk.tokenize import sent_tokenize
import time
import csv

model = SentenceTransformer('msmarco-roberta-base-v3') # this is the model we are starting with and would like to fine tune further; possibly check input sequence length as well

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

# load sentence embeddings - these are obtained in the file get_corpus_embeddings.py - the runtime for this is significant so we save them after obtaining
embeddings_as_numpy = np.load(os.path.join('data', 'LLIS', 'llis_sentence_embeddings.npy'))
sentence_corpus_embeddings = torch.from_numpy(embeddings_as_numpy)

start_time = time.time() # start timer

# format needed for training set: train_examples = [InputExample(texts=['My first sentence', 'My second sentence'], label=0.8), ...]
# use semantic search sampling to find pairs of sentences to use in training set
get_top_n = 3
top_k = min(get_top_n+1, len(sentence_corpus))
high_matching_pairs = []
for sentence in sentence_corpus:
    sentence_embedding = model.encode(sentence, convert_to_tensor=True)
    cos_scores = util.cos_sim(sentence_embedding, sentence_corpus_embeddings)[0]
    top_results = torch.topk(cos_scores, k=top_k)
    for score, idx in zip(top_results[0], top_results[1]):
        if sentence == sentence_corpus[idx]:
            pass
        else:
            high_matching_pairs.append([sentence, sentence_corpus[idx], score.numpy()])

find_high_matching_pairs_time = time.time()
total_run_time = find_high_matching_pairs_time - start_time

writer = csv.writer(open(os.path.join('data','LLIS','llis_bert_training_set.csv'), "w"))
for pair in high_matching_pairs:
    writer.writerow(pair)

def print_runtime(run_time):
    if run_time < 60:
        print("--- %s seconds ---" % (run_time))
    else:
        print("--- %s minutes ---" % (run_time/60))

print('Total run time: ')
print_runtime(total_run_time)
