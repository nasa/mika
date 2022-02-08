# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 10:28:14 2022

@author: srandrad
"""
import pandas as pd
import numpy as np
import os

from tqdm import tqdm

from xgboost import XGBClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier, ClassifierChain
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB, ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, hamming_loss
from sklearn.preprocessing import MinMaxScaler
#import tensorflow_hub as hub


#import data
test_data = pd.read_csv(os.path.join('data','ICS_data',"ICS_test_sitreps_preprocessed.csv")).drop(["Unnamed: 0"], axis=1)
train_data = pd.read_csv(os.path.join('data','ICS_data',"ICS_train_sitreps_preprocessed.csv")).drop(["Unnamed: 0"], axis=1)
val_data = pd.read_csv(os.path.join('data','ICS_data',"ICS_val_sitreps_preprocessed.csv")).drop(["Unnamed: 0"], axis=1)

meta_predictors = ["TOTAL_PERSONNEL", "TOTAL_AERIAL", "PCT_CONTAINED_COMPLETED",
              "ACRES",  "WF_FSR", "INJURIES", "FATALITIES", "EST_IM_COST_TO_DATE", "STR_DAMAGED",
              "STR_DESTROYED", "NEW_ACRES", "EVACUATION_IN_PROGRESS", 
              "NUM_REPORTS", "DAYS_BURING", #'Combined_Text', 
              'Incident_region_AICC', 
              'Incident_region_CA', 'Incident_region_EACC','Incident_region_GBCC', 'Incident_region_HICC', 
              'Incident_region_NRCC','Incident_region_NWCC', 'Incident_region_RMCC', 'Incident_region_SACC',
              'Incident_region_SWCC', 'INC_MGMT_ORG_ABBREV_1', 'INC_MGMT_ORG_ABBREV_2','INC_MGMT_ORG_ABBREV_3', 
              'INC_MGMT_ORG_ABBREV_4','INC_MGMT_ORG_ABBREV_5', 'INC_MGMT_ORG_ABBREV_B','INC_MGMT_ORG_ABBREV_C', 
              'INC_MGMT_ORG_ABBREV_D','INC_MGMT_ORG_ABBREV_E', 'INC_MGMT_ORG_ABBREV_F']
targets = ["Traffic","Command_Transitions","Evacuations", "Inaccurate_Mapping", "Aerial_Grounding", 
           "Resource_Issues", "Injuries", "Cultural_Resources","Livestock", "Law_Violations", "Military_Base", 
           "Infrastructure", "Extreme_Weather", "Ecological", "Hazardous_Terrain", "Floods", "Dry_Weather"]

#prepare data
#Xtrain = train_data['Raw_Combined_Text']; 
ytrain = train_data[targets]
#Xval = val_data['Raw_Combined_Text']; 
yval = val_data[targets]
#Xtest = test_data['Raw_Combined_Text']; 
ytest = test_data[targets]


#embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
vec_cols = [col for col in train_data.columns if col not in targets+meta_predictors+['INCIDENT_ID']]
#print(vec_cols)
Xtrain_vec = train_data[vec_cols]#embed(Xtrain)
Xval_vec = val_data[vec_cols]#embed(Xval)
Xtest_vec = test_data[vec_cols]#embed(Xtest)
#scaler = MinMaxScaler()
#Xtrain_vec = pd.DataFrame(scaler.fit_transform(Xtrain_vec))
#Xval_vec = pd.DataFrame(scaler.fit_transform(Xval_vec))
#Xtest_vec = pd.DataFrame(scaler.fit_transform(Xtest_vec))
predictors = meta_predictors + [str(c) for c in Xtrain_vec.columns]
train = pd.concat([train_data, Xtrain_vec], axis=1)
train.columns = train.columns.astype(str)
val = pd.concat([val_data, Xval_vec], axis=1)
val.columns = val.columns.astype(str)
test = pd.concat([test_data, Xtest_vec], axis=1)
test.columns = test.columns.astype(str)

#hyper parameter optimization
def hyper_parameter_optimization(params, Xtrain, ytrain, Xtest, ytest, multilabel_func=None, classifier=None):
    best_params = {}
    for param in tqdm(params):
        if len(params[param]) == 1:
            best_params[param] = params[param][0]
            continue
        test_hamming_loss = []; train_hamming_loss = []
        test_acc = []; train_acc = []
        test_f1 = []; train_f1 = []
        test_precision = []; train_precision = []
        test_recall = []; train_recall = []
        for val in params[param]:
            test_params = {param:val}
            test_params.update(best_params)
            if multilabel_func is not None:
                clf = multilabel_func(classifier(**test_params))#, n_jobs=-1)
            else:
                clf = classifier(**test_params)
            clf.fit(Xtrain, ytrain)
            predictions = clf.predict(Xtest)
            train_preds = clf.predict(Xtrain)
            test_acc.append(round(accuracy_score(ytest,predictions),3)); train_acc.append(round(accuracy_score(ytrain,train_preds),3))
            test_f1.append(round(f1_score(ytest,predictions, average='macro',zero_division=0),3))
            train_f1.append(round(f1_score(ytrain,train_preds, average='macro',zero_division=0),3))
            test_precision.append(round(precision_score(ytest,predictions, average='macro', zero_division=0),3))
            train_precision.append(round(precision_score(ytrain,train_preds, average='macro', zero_division=0),3))
            test_recall.append(round(recall_score(ytest,predictions, average='macro', zero_division=0),3))
            train_recall.append(round(recall_score(ytrain,train_preds, average='macro', zero_division=0),3))
            test_hamming_loss.append(round(hamming_loss(ytest,predictions),3))
            train_hamming_loss.append(round(hamming_loss(ytrain,train_preds),3))
        best_params[param] = params[param][np.argmin(test_hamming_loss)]
    return best_params, classifier(**best_params)

#inputs
input_types = ['meta', 'text', 'meta+text']
X_train_inputs = [train[meta_predictors],
                  Xtrain_vec, 
                  train[predictors]]
y_train_inputs = [ytrain[targets], ytrain[targets], ytrain[targets]]
X_test_inputs = [test[meta_predictors], Xtest_vec, test[predictors]]
y_test_inputs = [ytest[targets], ytest[targets], ytest[targets]]
#models
models = {#'knn':KNeighborsClassifier, "svm":SVC, 
          #"decision tree":DecisionTreeClassifier, 
          #"random forest":RandomForestClassifier, 
        #"logisitc regression":LogisticRegression, "mlp":MLPClassifier, 
        #'ridge':RidgeClassifier,
        'xgboost':XGBClassifier, 'adaboost':AdaBoostClassifier}
#OneVsRest
one_v_rest_mdls = {model_name:[] for model_name in models}; best_ovr_params = {model_name:[] for model_name in models}
one_v_rest_mdls['decision tree'] = [{'criterion':'gini', 'max_features':'auto', 'class_weight':'balanced', 'splitter':'best'}, {'criterion':'entropy', 'max_features':'auto', 'class_weight':'balanced', 'splitter':'best'}, {'criterion':'entropy', 'max_features':'sqrt', 'class_weight':'balanced', 'splitter':'best'}]
ovr_model_params = {'knn':{'n_neighbors': [10, 50]+[i for i in range(55,1055,100)],
                       'weights':['uniform', 'distance'],'p':[1,2]}, 
                "svm":{'C': [1*(10**i) for i in range(-3,2,1)],'class_weight':[None, "balanced"],
                       'gamma': ['scale', 'auto',2],'break_ties':[True,False]}, 
                "decision tree": {'criterion':['gini', 'entropy'], 'max_features':['auto', 'sqrt', 'log2'],
                                  'class_weight':[None, "balanced"], 'splitter':['best', 'random']}, 
                "random forest":{'criterion':['gini', 'entropy'], 'max_features':['auto', 'sqrt', 'log2'],
                                 'class_weight':[None, "balanced"], 'n_estimators':[i for i in range(50, 550, 50)]}, 
                "logisitc regression":{'max_iter':[10000],'multi_class':['ovr'],
                                       'solver':['newton-cg', 'lbfgs', 'sag', 'saga'],
                                       'C': [10**i for i in range(-3,0)]+[i for i in range(1,11,1)],
                                       'class_weight':[None, "balanced"]}, 
                "mlp":{'max_iter':[10000],'alpha': [1*((10)**(i)) for i in range(-4,3,1)],
                       'learning_rate':['constant', 'invscaling', 'adaptive'],
                       'hidden_layer_sizes':[(50,), (50,50), (50,50,50), (100,), (100,100), (100,100,100)]}, 
                'ridge':{'alpha': [1*((10)**(i)) for i in range(-4,3,1)],
                         'class_weight':[None, "balanced"]},
                'xgboost':{'eval_metric':['logloss'], 'use_label_encoder':[False], 'max_depth':[None]+[i for i in range(3,25)],
                           'booster': ['gbtree', 'gblinear', 'dart'], 'n_estimators': [i for i in range(100, 500, 25)]}, 
                'adaboost': {'base_estimator':[None, one_v_rest_mdls['decision tree']],
                             'n_estimators':[i for i in range(50, 550, 50)],
                             'learning_rate':[1*((10)**(i)) for i in range(-4,2,1)]}}

for model_name in tqdm(models):
    if model_name == "random forest": start=1
    elif model_name == 'xgboost': start=1
    else: start = 0
    for i in range(start, len(input_types)):
    
        if model_name == 'adaboost':
            ovr_model_params['adaboost']['base_estimator'] = [None, one_v_rest_mdls['decision tree'][i]]
    
        best_params, best_mdl = hyper_parameter_optimization(ovr_model_params[model_name], X_train_inputs[i], y_train_inputs[i], X_test_inputs[i], y_test_inputs[i], 
                                                             multilabel_func=OneVsRestClassifier, classifier=models[model_name])
        one_v_rest_mdls[model_name].append(best_mdl)
        best_ovr_params[model_name].append(best_params)
        print(best_params)

ovr_params_df = pd.DataFrame(best_ovr_params, index=input_types)
file = os.path.join('results','hazard_classification_model_params.xlsx')
ovr_params_df.to_excel(file, sheet_name="One Vs Rest (BR)")  
ovr_params_df.to_csv(os.path.join('results','hazard_classification_model_BR_params.csv'))

#Classifier Chain
"""
cc_mdls = {model_name:[] for model_name in models}; best_cc_params = {model_name:[] for model_name in models}
cc_model_params = ovr_model_params.copy()

for model_name in tqdm(models):
    for i in range(len(input_types)):
        
        if model_name == 'adaboost':
            cc_model_params['adaboost']['base_estimator'] = [None, cc_mdls['decision tree'][i]]
    
        best_params, best_mdl = hyper_parameter_optimization(cc_model_params[model_name], X_train_inputs[i], y_train_inputs[i], X_test_inputs[i], y_test_inputs[i], 
                                                             multilabel_func=ClassifierChain, classifier=models[model_name])
        cc_mdls[model_name].append(best_mdl)
        best_cc_params[model_name].append(best_params)
        
cc_params_df = pd.DataFrame(best_cc_params, index=input_types)
with pd.ExcelWriter(file, engine='openpyxl', mode='a') as writer:  
    cc_params_df.to_excel(writer, sheet_name="Classifier Chains (CC)")

cc_params_df.to_csv(os.path.join('results','hazard_classification_model_CC_params.csv'))
"""
#Label Powerset method
"""

ytrain['powerlabel'] = ytrain.apply(lambda x : sum([(2**i)*x[targets[i]] for i in range(len(targets))]),axis=1)
yval['powerlabel'] = yval.apply(lambda x : sum([(2**i)*x[targets[i]] for i in range(len(targets))]),axis=1)
ytest['powerlabel'] = ytest.apply(lambda x : sum([(2**i)*x[targets[i]] for i in range(len(targets))]),axis=1)

lp_mdls = {model_name:[] for model_name in models}; best_lp_params = {model_name:[] for model_name in models}
lp_model_params = ovr_model_params.copy()

for model_name in tqdm(models):
    for i in range(len(input_types)):
        
        if model_name == 'adaboost':
            cc_model_params['adaboost']['base_estimator'] = [None, cc_mdls['decision tree'][i]]
    
        best_params, best_mdl = hyper_parameter_optimization(lp_model_params[model_name], X_train_inputs[i], ytrain['powerlabel'],  X_test_inputs[i], ytest['powerlabel'],
                                                             multilabel_func=None, classifier=models[model_name])
        lp_mdls[model_name].append(best_mdl)
        best_lp_params[model_name].append(best_params)
        
lp_params_df = pd.DataFrame(best_lp_params, index=input_types)
with pd.ExcelWriter(file, engine='openpyxl', mode='a') as writer:  
    lp_params_df.to_excel(writer, sheet_name="Label Powerset (LP)")
lp_params_df.to_csv(os.path.join('results','hazard_classification_model_LP_params.csv'))
"""