#!/usr/bin/env python

import os
import pandas as pd
import numpy as np
import argparse as ap
import logging
import pdb
import glob
import sys
import deepsvr_utils as dp
import vaex
from tqdm import tqdm

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

PREFIX = '/projects/rmorin/projects/ffpe_ml'
DATA = os.path.join(PREFIX, 'data')

logger = logging.getLogger()

tqdm.pandas(ascii = True)

"""
Progress bar
"""
def progress(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.1 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s %s\r' % (bar, percents, '%', status))
    sys.stdout.flush()

def convert_one_based(row):
    """
    Converts 0-based coordinates to 1-based
    """
    row['start'] += 1
    return row

def extract(ref, gt, vcf, bam, outdir, prefix, skip_bam_readcount, loglevel):
    logging.basicConfig(level=loglevel,
                        format='%(asctime)s (%(relativeCreated)d ms) -> %(levelname)s: %(message)s',
                        datefmt='%I:%M:%S %p')

    if not prefix:
        prefix = os.path.basename(bam.split('.')[0])

    logger.info('Preparing data')
    prep_data = dp.PrepareData(prefix, bam, bed_file_path, ref, outdir)
    
    df = prep_data.training_data

    try:
        true_vars = pd.read_pickle(os.path.join(DATA, 'train_data', 'true_vars.pkl'))
    except:
        logger.info('Converting ground truth variants to 1-based coordinates')
        true_vars = pd.read_csv(label,
                                sep='\t', index_col=None, header=None, dtype={0: str})
        true_vars.columns = ['chr', 'start', 'end', 'ref', 'alt']
        true_vars = true_vars.progress_apply(convert_one_based, axis=1)
        true_vars.to_pickle(os.path.join(outdir, 'true_vars.pkl'))

    df['real'] = 0

    sample = df.index[0].split('~')[0]
    true_vars_set = set(df.index.str.replace(sample + '~', ''))

    for index,row in true_vars.iterrows():
        progress(index, true_vars.shape[0])
        var = "{0}:{1}-{2}{3}>{4}".format(row.chr, row.start, row.end, row.ref, row.alt)
        if var in true_vars_set:
            df.loc[sample + '~' + var, 'real'] = 1

    vaex_df = vaex.from_pandas(df)
    vaex_df.export(os.path.join(outdir, 'train.hdf5'))  
