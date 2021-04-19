# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 10:11:37 2020
Useful Functions for ML
@author: srandrad
"""

def feature_importance_graph(model,features, title = "Feature Importance"):
    """Plots a bar graph of the relative importance of each feature in classification"""
    import matplotlib as plt
    import numpy as np
    plt.style.use('seaborn')
    y = model.feature_importances_
    #plot
    fig, ax = plt.subplots() 
    width = 0.4 # the width of the bars 
    ind = np.arange(len(y)) # the x locations for the groups
    ax.barh(ind, y, width, color="purple")
    ax.set_yticks(ind+width/10)
    ax.set_yticklabels(features, minor=False)
    plt.title(title)
    plt.xlabel('Relative importance')
    plt.ylabel('Feature') 
    plt.figure(figsize=(5,5))
    fig.set_size_inches(6.5, 4.5, forward=True)
    
def feature_contribution(model, X_test, features, classes):
    """Plots the contribution of each feature in categorizing a given sample
    Decision tree based algorithms only"""
    import matplotlib as plt
    from matplotlib.pyplot import cm
    plt.style.use('seaborn')
    import numpy as np
    from treeinterpreter import treeinterpreter as ti
    prediction, bias, contributions = ti.predict(model, X_test[8:9])
    N = len(features)+1 # no of entries in plot , 4 ---> features & 1 ---- class label
    data ={}
    for key in classes: data[key]=[]
    for j in range(len(classes)):
        list_ =  [data[key] for key in classes]
        for i in range(len(features)):
            val = contributions[0,i,j]
            list_[j].append(val)
        list_[j].append(prediction[0,j]/N)
    fig, ax = plt.subplots()
    ind = np.arange(N)   
    width = 0.15
    i=0
    ps = []
    color=iter(cm.rainbow(np.linspace(0,1,len(classes))))
    for key in classes:
        c=next(color)
        p=ax.bar(ind+(i*width), data[key], width, color=c, bottom=0)
        i+=1
        ps.append(p)
    ax.set_title('Importance of All Features for \n a Single Classification')
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(features+['classification'], rotation = 90)
    ax.legend((p[0] for p in ps), classes , bbox_to_anchor=(1.04,1), loc="upper left")
    ax.autoscale_view()
    plt.show()
    
def learning_curves(estimator, data, features, target, train_sizes, cv):
    """Plots the learning curves. We expect to see convergence if there is no over fitting,
    gaps indicated more training is needed"""
    import matplotlib as plt
    from sklearn.model_selection import learning_curve
    plt.style.use('seaborn')
    train_sizes, train_scores, validation_scores = learning_curve(estimator, data[features].astype(float), data[target].astype(int), train_sizes =train_sizes,
                                                                  cv = cv)#, scoring = 'neg_mean_squared_error')
    train_scores_mean = train_scores.mean(axis = 1)
    validation_scores_mean = validation_scores.mean(axis = 1)

    plt.plot(train_sizes, train_scores_mean, label = 'Training error')
    plt.plot(train_sizes, validation_scores_mean, label = 'Validation error')
    #plt.ylim([0.75, 1])
    plt.ylabel('Score', fontsize = 14)
    plt.xlabel('Training set size', fontsize = 14)
    title = 'Learning curves for a ' + str(estimator).split('(')[0] + ' model'
    plt.title(title, fontsize = 18, y = 1.03)
    plt.legend()

from sklearn.metrics import accuracy_score
def t_test_for_train_test_scores(model, X, y, score = accuracy_score()):
    """Used to examine the difference between the test score and train score.
    The goal is a small, non-significant difference. Differences indicate overfitting"""
    import matplotlib as plt
    import matplotlib.patches as patches
    plt.style.use('seaborn')
    from sklearn.model_selection import train_test_split
    import pingouin as pg
    from sklearn.metrics import score
    train_acc = []
    test_acc = []
    for i in range (0,100):
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=i)
        model.fit(X_train, y_train) #Training the model
        #conf_mat.append(confusion_matrix(y_test.astype(int),model.predict(X_test.astype(float))))
        train_acc.append(score(y_train.astype(int), model.predict(X_train.astype(float))))
        test_acc.append(score(y_test.astype(int), model.predict(X_test.astype(float))))
    
    fig = plt.figure(figsize= (20, 10))
    ax = fig.add_subplot(111)
    
    p_d1= plt.hist(train_acc, label= "train",density= True, alpha=0.75, color = 'red')
    p_d2 = plt.hist(test_acc, label= "test",density= True, alpha=0.75, color = 'blue')
    title = 'Distribution of'+str(score)+' scores between training and testing data' 
    plt.suptitle(title, fontsize= 20)
    plt.xlabel("accuracy", fontsize= 16)
    plt.ylabel("Probability density", fontsize= 16)
    label1 = patches.Patch(color = 'red', label = "train")
    label2 = patches.Patch(color = 'blue', label = "test")
    plt.legend(handles = [label1, label2], bbox_to_anchor=(1.05, 1.0), loc='upper left')
    
    plt.show()
    print(pg.ttest(train_acc, test_acc, paired = True))

"""for ROC with cross validation see https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html#sphx-glr-auto-examples-model-selection-plot-roc-crossval-py"""

def tree_viz1(model, features, target_names, save = False):
    from sklearn import tree
    import matplotlib as plt
    import graphviz
    i = 0
    for tree_in_forest in model.estimators_:
        if i <2:
            fig = plt.figure(figsize=(25,20))
            _ = tree.plot_tree(tree_in_forest, feature_names=features, class_names=target_names,filled=True)
            i+=1
            fig.show()
            if save == True: fig.savefig("decistion_tree.png")
            
def tree_viz2(model, X, y, features, target_names):
    from dtreeviz.trees import dtreeviz
    from graphviz import dot# remember to load the package
    import os
    viz = dtreeviz(model.estimators_[4], X, y,
                    target_name="target",
                    feature_names=features,
                    class_names=target_names)
    
    print(viz)
    viz.save("decision_tree.svg")

def oob_error_plot(models,X,y):
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from collections import OrderedDict
    from sklearn.ensemble import RandomForestClassifier
    
    RANDOM_STATE = 123
    
    # NOTE: Setting the `warm_start` construction parameter to `True` disables
    # support for parallelized ensembles but is necessary for tracking the OOB
    # error trajectory during training.
    #EXAMPLE OF HOW models SHOULD LOOK
    ensemble_clfs = [
        ("RandomForestClassifier, max_features='log2'",
            RandomForestClassifier(warm_start=True, max_features='log2',
                                   oob_score=True,
                                   random_state=RANDOM_STATE)),
        ("RandomForestClassifier, max_features=None",
            RandomForestClassifier(warm_start=True, max_features=None,
                                   oob_score=True,
                                   random_state=RANDOM_STATE))
    ]
    
    # Map a classifier name to a list of (<n_estimators>, <error rate>) pairs.
    error_rate = OrderedDict((label, []) for label, _ in models)
    test_error_rate = OrderedDict((label, []) for label, _ in models)
    # Range of `n_estimators` values to explore.
    min_estimators = 15
    max_estimators = 175
    
    for label, clf in models:
        for i in range(min_estimators, max_estimators + 1):
            clf.set_params(n_estimators=i)
            X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=i)
            clf.fit(X_train, y_train)
    
            # Record the OOB error for each `n_estimators=i` setting.
            oob_error = 1 - clf.oob_score_
            error_rate[label].append((i, oob_error))
            clf.predict(X_test)
            oob_error = 1 - clf.oob_score_
            test_error_rate[label].append((i, oob_error))
    # Generate the "OOB error rate" vs. "n_estimators" plot.
    for label, clf_err in error_rate.items():
        xs, ys = zip(*clf_err)
        plt.plot(xs, ys, label=label)
    for label, clf_err in test_error_rate.items():
        xs, ys = zip(*clf_err)
        plt.plot(xs, ys, label=label)
    
    plt.xlim(min_estimators, max_estimators)
    plt.xlabel("n_estimators")
    plt.ylabel("OOB error rate")
    plt.legend(loc="upper right")
    plt.show()