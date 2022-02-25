# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 10:47:30 2022

@author: srandrad
"""

# clustering
import pandas as pd
import os
from sklearn.cluster import DBSCAN, AgglomerativeClustering    
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

data = pd.read_csv(os.path.join('data', 'SAFECOM_UAS_fire_data.csv'), index_col=0)
cols_to_clean = [col for col in data.columns if 'Abstractive' in col]
for col in cols_to_clean:
    data[col] = [str(text).strip('<pad>').strip('</s') for text in data[col]]

embedder = SentenceTransformer('all-MiniLM-L6-v2')

summary_cols = ['Abstractive Summarized Narrative','Extractive Summarized Narrative']

def cluster_embeddings(corpus, db_params={}):
    corpus_embeddings = embedder.encode(corpus)
    corpus_embeddings = corpus_embeddings /  np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)
    #perform clustering: Agglomerative
    clustering_model = AgglomerativeClustering(n_clusters=None, distance_threshold=1.5) #, affinity='cosine', linkage='average', distance_threshold=0.4)
    clustering_model.fit(corpus_embeddings)
    cluster_assignment = clustering_model.labels_
    
    clustered_sentences = {}
    for sentence_id, cluster_id in enumerate(cluster_assignment):
        if cluster_id not in clustered_sentences:
            clustered_sentences[cluster_id] = []
    
        clustered_sentences[cluster_id].append(corpus[sentence_id])
    
    for i, cluster in clustered_sentences.items():
        print("Cluster ", i+1)
        print(cluster)
        print("")
    #perform clustering: DBSCAN
    db = DBSCAN(**db_params).fit(corpus_embeddings)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    print("Estimated number of clusters: %d" % n_clusters_)
    print("Estimated number of noise points: %d" % n_noise_)
    print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(corpus_embeddings, labels))
    
    # #############################################################################
    # Plot result
    
    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]
    
        class_member_mask = labels == k
    
        xy = corpus_embeddings[class_member_mask & core_samples_mask]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=14,
        )
    
        xy = corpus_embeddings[class_member_mask & ~core_samples_mask]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=6,
        )

    plt.title("Estimated number of clusters: %d" % n_clusters_)
    plt.show()
    
for col in summary_cols:
    db_params = {'eps':0.8, 'min_samples':4}
    print(col, "\n")
    cluster_embeddings(corpus=data[col].tolist(), db_params=db_params)