#!/usr/bin/env python

import os
import pandas as pd 
import numpy as np
import argparse as ap
import logging
import deepsvr_utils_formatted as dp
import joblib
import gzip
import pdb
import vaex

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV

BASE = os.path.dirname(os.getcwd())
logger = logging.getLogger()

if __name__ == '__main__':
    parser = ap.ArgumentParser(description='FFPE-\n' +
                                           'Copyright (C) 2020 Matthew Nguyen',
                               formatter_class=ap.RawTextHelpFormatter)

    parser.add_argument('-ll', '--loglevel', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Set the logging level')
    parser.add_argument('ref', metavar='REFERENCE', help='Reference genome FASTA')
    parser.add_argument('vcf', metavar='VCF', help='VCF to filter')
    parser.add_argument('bam', metavar='BAM', help='Tumor BAM file')
    parser.add_argument('-o', '--outdir', metavar='DIR', default=os.path.join(os.getcwd(), 'results'), 
                        help='Output directory')
    parser.add_argument('-p', '--prefix', default=None, help='Output prefix (default: basename of BAM)')
    # parser.add_argument('-rt', '--retrain', metavar='HDF5', default=None, 
    #                     help='Retrain model with new data (in hdf5 format)')
    parser.add_argument('-rt', '--retrain', action='store_true', 
                        help='Retrain model with new data (in hdf5 format)')
    parser.add_argument('-gs', '--grid_search', action='store_true', default=False,
                        help='Perform grid search when retraining model')
    parser.add_argument('-c', '--cores', metavar='INT', default=1, 
                        help='Number of cores to use for grid search (default: 1)')
    parser.add_argument('-s', '--seed', metavar='INT', default=667, 
                        help='Seed for retraining (default: 667)')

    args = parser.parse_args()

    logging.basicConfig(level=args.loglevel,
                        format='%(asctime)s (%(relativeCreated)d ms) -> %(levelname)s: %(message)s',
                        datefmt='%I:%M:%S %p')
    
    if not args.prefix:
        args.prefix = os.path.basename(os.path.join(args.outdir, args.bam.split('.')[0]))
    
    if args.retrain:
        orig_train_df = vaex.open(os.path.join(BASE, 'orig_train.hdf5'))
        train_features = orig_train_df.get_column_names(regex='tumor')
        # new_train_df = args.retrain

        orig_train_df = orig_train_df.to_pandas_df(train_features + ['real'])
        # new_train_df = new_train_df.to_pandas_df(train_features + ['real'])

        clf = LogisticRegression(penalty='l2', random_state=args.seed, solver='saga', max_iter=10000, 
                                 class_weight='balanced', C=0.001)
        scaler = MinMaxScaler()

        logger.info('Concatenate old and new training feature matrices')
        train_df = orig_train_df
        # train_df = pd.concat([orig_train_df, new_train_df], ignore_index=True)
        X = train_df[train_features]
        y = train_df['real']

        logger.info('Scaling training feature matrix')
        X = scaler.fit_transform(X)

        logger.info('Training model')
        if args.grid_search:
            logger.info('Training using grid search')
            param = {'C': [1e-3, 1e-2, 1e-1, 1]}
            metrics = ['accuracy', 'precision', 'recall', 'roc_auc', 'f1']
            gs = GridSearchCV(clf, param, n_jobs=args.cores, scoring=metrics, cv=10, refit='f1')
            gs.fit(X, y)
            clf = gs.best_estimator_
        else:
            logger.info('Training using previous optimized parameters')
            clf.fit(X, y)

    else:
        clf = joblib.load(os.path.join(BASE, 'trained.clf'))
        scaler = joblib.load(os.path.join(BASE, 'trained.scaler'))

    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)

    logger.info('Converting VCF to bed file')
    bed_file_path = os.path.join(args.outdir, args.prefix) + '.bed'
    os.system('zcat {} | grep PASS | vcf2bed | cut -f 1-3,6-7 > {}'.format(args.vcf, bed_file_path))
    
    logger.info('Preparing data')
    prep_data = dp.PrepareData(args.prefix, args.bam, bed_file_path, args.ref, args.outdir)

    df = prep_data.training_data

    logger.info('Hard-filter variants')
    df = df[df.tumor_VAF > 0.05]
    df = df[df.tumor_depth > 10]
    df = df[df.tumor_var_num_minus_strand + df.tumor_var_num_plus_strand > 4]

    df = df.drop(['ref', 'var'], axis=1)

    logger.info('Scaling features')
    df_scaled = scaler.transform(df)

    logger.info('Obtaining predictions')
    preds = clf.predict(df_scaled)
    df['preds'] = preds
    df = df[df.preds == 1]

    logger.info('Filtering VCF')
    kept_vars = set(df.index.str.replace(args.prefix + '~', ''))

    vcf_file_path = os.path.join(args.outdir, args.prefix + '_filtered.vcf.gz')
    with gzip.open(vcf_file_path, 'wt') as f_out:
        with gzip.open(args.vcf, 'rt') as f_in:
            for line in f_in:
                if not line.startswith('#'):
                    split = line.split('\t')
                    var = '{}:{}-{}{}>{}'.format(split[0], split[1], split[1], split[3], split[4])
                    if var in kept_vars:
                        f_out.write(line)
                else:
                    f_out.write(line)