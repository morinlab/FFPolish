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

ProgressBar().register()
logger = logging.getLogger()

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


if __name__ == '__main__':
    tqdm.pandas(ascii = True)

    parser = ap.ArgumentParser(description='FFPolish-extract\n' +
                                           'Copyright (C) 2020 Matthew Nguyen',
                               formatter_class=ap.RawTextHelpFormatter)

    parser.add_argument('-ll', '--loglevel', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Set the logging level')
    parser.add_argument('ref', metavar='REFERENCE', help='Reference genome FASTA')
    parser.add_argument('vcf', metavar='VCF', help='VCF to filter')
    parser.add_argument('bam', metavar='BAM', help='Tumor BAM file')
    parser.add_argument('outdir', metavar='DIR', help='Output directory')
    parser.add_argument('-p', '--prefix', default=None, help='Output prefix (default: basename of BAM)')
    parser.add_argument('--skip_bam_readcount', action='store_true', default=False,
                        help='Skip bam_readcount on sample')
    parser.add_argument('--labels', metavar='BED', default=None,
                        help='BED file of true variants (to create pickle file of true variants)')
    parser.add_argument('--pkl', metavar='PKL', default=None, 
                        help='Pickle file of true variants')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s (%(relativeCreated)d ms) -> %(levelname)s: %(message)s',
                        datefmt='%I:%M:%S %p')

    if not args.prefix:
        args.prefix = os.path.basename(args.bam.split('.')[0])

    # Generate matrix of true variants
    args.labels or not os.path.exists(os.path.join(args.outdir, 'true_vars.pkl')):
        logger.info('Converting ground truth variants to 1-based coordinates')
        true_vars = pd.read_csv(args.label),
                                sep='\t', index_col=None, header=None, dtype={0: str})
        true_vars.columns = ['chr', 'start', 'end', 'ref', 'alt']
        true_vars = true_vars.progress_apply(convert_one_based, axis=1)
        true_vars.to_pickle(os.path.join(args.outdir, 'true_vars.pkl'))

    logger.info('Preparing data')
    prep_data = dp.PrepareData(args.prefix, args.bam, bed_file_path, args.ref, args.outdir)
    
    df = prep_data.training_data

    true_vars = pd.read_pickle(os.path.join(DATA, 'train_data', 'true_vars.pkl'))
    df['real'] = 0

    sample = df.index[0].split('~')[0]
    true_vars_set = set(df.index.str.replace(sample + '~', ''))

    for index,row in true_vars.iterrows():
        progress(index, true_vars.shape[0])
        var = "{0}:{1}-{2}{3}>{4}".format(row.chr, row.start, row.end, row.ref, row.alt)
        if var in true_vars_set:
            df.loc[sample + '~' + var, 'real'] = 1

    vaex_df = vaex.from_pandas(df)
    vaex_df.export(os.path.join(args.outdir, 'train.hdf5'))
