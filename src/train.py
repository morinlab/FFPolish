#!/usr/bin/env python

import os
import pandas as pd 
import logging
import vaex

from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score, balanced_accuracy_score
from joblib import dump, load

def print_metrics_cv(outdir, model, cv, folds):
    """
    Prints and writes all scores we care about in a file
    """
    train_acc = cv['train_accuracy'].mean()
    test_acc = cv['test_accuracy'].mean()
    train_bal_acc = cv['train_balanced_accuracy'].mean()
    test_bal_acc = cv['test_balanced_accuracy'].mean()
    train_auc = cv['train_roc_auc'].mean()
    test_auc = cv['test_roc_auc'].mean()
    train_prec = cv['train_precision'].mean()
    test_prec = cv['test_precision'].mean()
    train_rec = cv['train_recall'].mean()
    test_rec = cv['test_recall'].mean()
    train_f1 = cv['train_f1'].mean()
    test_f1 = cv['test_f1'].mean()
    
    text = """

    Model: {0}
    Average scores across {1} folds

    Train accuracy: {2}
    Train balanced accuracy: {3}
    Train AUC: {4}
    Train precision: {5}
    Train recall: {6}
    Train F1-score: {7}

    Test accuracy: {8}
    Test balanced accuracy: {9}
    Test AUC: {10}
    Test precision: {11}
    Test recall: {12}
    Test F1-score: {13}

    """.format(model, folds, train_acc, train_bal_acc, train_auc, train_prec, train_rec, train_f1, 
               test_acc, test_bal_acc, test_auc, test_prec, test_rec, test_f1)

    print(text)
    with open(os.path.join(outdir, '{0}.score.txt'.format(model)), 'w') as out:
        out.write(text)

def train(model, datasets, outdir, cores, grid_search, folds, seed, loglevel):
    logging.basicConfig(level=loglevel,
                        format='%(asctime)s (%(relativeCreated)d ms) -> %(levelname)s: %(message)s',
                        datefmt='%I:%M:%S %p')

    if not os.path.exists(outdir):
        os.mkdir(outdir)    

    logging.info('Loading dataframes')
    train_df = vaex.open_many(datasets)

    lr_l1 = LogisticRegression(penalty='l1', random_state=seed, solver='saga', max_iter=10000)
    lr_l2 = LogisticRegression(penalty='l2', random_state=seed, solver='saga', max_iter=100000, class_weight='balanced')
    eNet = LogisticRegression(penalty='elasticnet', random_state=seed, solver='saga', max_iter=10000, 
                              l1_ratio=0.5, class_weight='balanced')
    rf = RandomForestClassifier(random_state=seed)
    gbm = GradientBoostingClassifier(random_state=seed)

    models = {'enet': eNet, 'gbm': gbm, 'rf': rf, 'lr_l2': lr_l2, 'lr_l1': lr_l1}

    lr_l1_param = {'C': [1e-3, 1e-2, 1e-1, 1]}
    lr_l2_param = {'C': [1e-3, 1e-2, 1e-1, 1]}
    eNet_param = {'C': [1e-3, 1e-2, 1e-1, 1], 'l1_ratio': [0.15, 0.05, 0.25, 0.5, 0.75]}
    rf_param = {'n_estimators': [10, 100, 200, 500, 1000, 1500, 2000], 'max_depth': [15, 30, 45, 60, 80, 100]}
    gbm_param = {'learning_rate': [1e-3, 1e-2, 1e-1, 1, 10], 'n_estimators': [50, 100, 200, 500],
                 'max_depth': [5, 15, 30, 50, 80], 'subsample': [0.6, 0.7, 0.8]}
    params = {'lr_l1': lr_l1_param, 'lr_l2': lr_l2_param, 'enet': eNet_param, 'rf': rf_param, 'gbm': gbm_param}

    metrics = ['accuracy', 'balanced_accuracy', 'precision', 'recall', 'roc_auc', 'f1']

    train_features = train_df.get_column_names(regex='tumor')
    train_df = train_df.to_pandas_df(train_features + ['real'])
    y = train_df['real']
    X = train_df.drop(['real'], axis=1)
    
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    dump(scaler, os.path.join(outdir, '{0}_scaler.pkl'.format(model)))

    if grid_search:
        logging.info('Performing grid search: {0}'.format(model))
        clf = GridSearchCV(models[model], params[model], n_jobs=cores, scoring='f1', cv=10, 
                           return_train_score=True, verbose=2, refit=False)
        clf.fit(X, y)

        print("Best parameters: {0}".format(clf.best_params_))
        models[model].set_params(**clf.best_params_)

    logging.info('Performing cross validation: {0}'.format(model))
    scores = cross_validate(models[model], X, y, scoring=metrics, cv=folds, n_jobs=cores, 
                            return_train_score=True, verbose=2)
    print_metrics_cv(outdir, model, scores, folds)

    models[model].fit(X, y)
    dump(models[model], os.path.join(outdir, '{0}.pkl'.format(model)))
