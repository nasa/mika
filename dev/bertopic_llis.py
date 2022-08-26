"""
@author: hswalsh
"""

import os
import pandas as pd
import sys
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from module.calc_coherence import calc_coherence
from sklearn.feature_extraction.text import CountVectorizer
import time

# load data - no preprocessing required for bertopic!
# import LLIS docs
llis_filename = os.path.join('data','LLIS','lessons_learned_2021-12-10.xlsx')
llis_df = pd.read_excel(llis_filename)

# treat paragraphs in llis as separate documents - paragraphs are more "requirement-sized"
llis_abstract = llis_df['Abstract'].to_list()
llis_lessons_learned = llis_df['Lesson(s) Learned'].to_list()
llis_recommendations = llis_df['Recommendation(s)'].to_list()
llis_driving_event = llis_df['Driving Event'].to_list()
llis_evidence = llis_df['Evidence'].to_list()
llis_corpus = llis_abstract + llis_lessons_learned + llis_recommendations + llis_driving_event + llis_evidence

# remove nans
def remove_nans(docs):
    for i in range(0,len(docs)):
        if pd.isnull(docs[i]):
            docs[i] = ''
    return docs
llis_corpus = remove_nans(llis_corpus)
llis_corpus = list(filter(('').__ne__, llis_corpus))

start_time = time.time() # start timer

# setup embeddings
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = sentence_model.encode(llis_corpus, show_progress_bar=False)

# create and train topic model
umap_model = UMAP(random_state=1) # setting the random state sacrifices performance so that results are consistent between runs
vectorizer_model = CountVectorizer(ngram_range=(1,3), stop_words="english") # remove stopwords, otherwise topics seem to contain low-information words
topic_model = BERTopic(umap_model=umap_model, vectorizer_model = vectorizer_model, verbose=True)
topics, probs = topic_model.fit_transform(llis_corpus, embeddings)
coherence = calc_coherence(llis_corpus, topics, topic_model, 'c_v')

run_time = time.time() - start_time # end timer

topic_info = topic_model.get_topic_info()
topic_info['coherence'] = ['coherence'] + coherence
topic_info.to_csv(os.path.join('results','bertopics_llis.csv'))

if run_time < 60:
    print("--- %s seconds ---" % (run_time))
else:
    print("--- %s minutes ---" % (run_time/60))
