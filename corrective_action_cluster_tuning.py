# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 09:14:20 2022

@author: srandrad
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
plt.style.use('seaborn')
import sys
import os
sys.path.append(os.path.join(".."))

from tqdm import tqdm
from sklearn.cluster import KMeans, SpectralClustering, MeanShift, AffinityPropagation, DBSCAN, AgglomerativeClustering  
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

data = pd.read_csv(os.path.join('data', 'SAFECOM_UAS_fire_data.csv'), index_col=0)
kmeans_param_grid = {'n_clusters':[i for i in range(4, 25)],
                    'init':['k-means++', 'random'],
                    'n_init':[i for i in range(5, 30, 5)],
                    'tol':[1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
                    'random_state':[1],
                    'algorithm':['full', 'elkan']}
spectral_param_grid = {'n_clusters':[i for i in range(4, 25)],
                       'eigen_solver':['arpack', 'lobpcg', 'amg'],
                       'n_init':[i for i in range(5, 30, 5)],
                       'gamma': [1e-4,1e-3,1e-2,1e-1,1,1e1],
                       'affinity':['rbf', 'nearest_neighbors', 'precomputed', 'precomputed_nearest_neighbors'],
                       'n_neighbors': [i for i in range(5, 40, 5)],
                       'eigen_tol': [0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
                       'assign_labels':['kmeans', 'discretize'],
                       'degree': [2,3,4,5],
                       'random_state':[1],
                       'coef0':[1,2,3,-1,-2,0]}
mean_shift_param_grid = {'bandwidth':[0, 1e-1, 1e-2, 1e-3, 1e-4,1,5,10],
                        'min_bin_freq':[1,2,3,4,5,6,7,8,9,10],
                        'cluster_all':[True, False]}
affintiy_param_grid = {'damping': [0.5,0.6,0.7,0.8,9,0.99],
                       'random_state':[1],
                      'affinity':['euclidean']}#, 'precomputed']}
DBSCAN_param_grid = {'eps':[1e-2, 1e-3, 0.1, 0.2, 0.3, 0.4, 0.5,0.6,0.7,0.8,0.9,1,2,3],
                    'min_samples':[1,2,3,4,5,6,7,8],
                    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                     'leaf_size': [i for i in range(5, 45, 5)]}
aglommorative_param_grid = {'n_clusters':[None]+[i for i in range(4, 25)],
                           'linkage':['ward', 'complete', 'average', 'single'],
                           'distance_threshold':[1e-2, 1e-3, 0.1, 0.2, 0.3, 0.4, 0.5,0.6,0.7,0.8,0.9,1,2,3]}

cluster_dict = {#'kmeans':KMeans, 'spectral':SpectralClustering, 
                #'mean_shift': MeanShift,
               'affinity': AffinityPropagation, 'DBSCAN': DBSCAN, 'aglommorative': AgglomerativeClustering}
cluster_params = {'kmeans':kmeans_param_grid, 'spectral':spectral_param_grid, 'mean_shift': mean_shift_param_grid,
               'affinity': affintiy_param_grid, 'DBSCAN':DBSCAN_param_grid, 'aglommorative': aglommorative_param_grid}
best_models = {}
best_params = {}
embedder = SentenceTransformer('all-mpnet-base-v2')

def silhouette_score_cust(estimator, X):
    labels = estimator.fit_predict(X)
    try:
        score = silhouette_score(X, labels)
        # score = sklearn.metrics.calinski_harabasz_score(X, labels)
    except ValueError:
        score = -1
    return score

for cluster in tqdm(cluster_dict):
    x = embedder.encode(data['Corrective Action'].tolist())
    cluster_model = cluster_dict[cluster]
    GS = GridSearchCV(estimator=cluster_model(), param_grid=cluster_params[cluster],scoring=silhouette_score_cust)
    GS.fit(x)
    best_models[cluster] = GS.best_estimator_
    best_params[cluster] = GS.best_params_
    print(GS.best_params_)

print(best_params)
print(best_models)

file = os.path.join('models', 'SAFECOM_Corrective_Action_cluster_models.xlsx')
with pd.ExcelWriter(file) as writer2:
    for results in best_params:
        params_dict_formatted = { x:[y] for x,y in best_params[results].iteritems() }
        params_dict_formatted.to_excel(writer2, sheet_name = results, index = False)