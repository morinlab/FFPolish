#!/usr/bin/env python

import os
import argparse as ap
from filter import filter

VERSION = 0.1

if __name__ == '__main__':
    parser = ap.ArgumentParser(description='FFPolish\n'
                                           'Version %f\n'
                                           'Copyright (C) 2020 Matthew Nguyen' % VERSION,
                               formatter_class=ap.RawTextHelpFormatter)

    parser.add_argument('-ll', '--loglevel', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Set the logging level')
    subparsers = parser.add_subparsers(help='Choose an FFPolish tool')

    predict_parser = subparsers.add_parser('filter', help='Filter variants')
    predict_parser.add_argument('ref', metavar='REFERENCE', help='Reference genome FASTA')
    predict_parser.add_argument('vcf', metavar='VCF', help='VCF to filter')
    predict_parser.add_argument('bam', metavar='BAM', help='Tumor BAM file')
    predict_parser.add_argument('-o', '--outdir', metavar='DIR', default=os.path.join(os.getcwd(), 'results'), 
                        help='Output directory')
    predict_parser.add_argument('-p', '--prefix', default=None, help='Output prefix (default: basename of BAM)')
    predict_parser.add_argument('-rt', '--retrain', metavar='HDF5', default=None, 
                        help='Retrain model with new data (in hdf5 format)')
    predict_parser.add_argument('-gs', '--grid_search', action='store_true', default=False,
                        help='Perform grid search when retraining model')
    predict_parser.add_argument('-c', '--cores', metavar='INT', default=1, 
                        help='Number of cores to use for grid search (default: 1)')
    predict_parser.add_argument('-s', '--seed', metavar='INT', default=667, 
                        help='Seed for retraining (default: 667)')

    args = parser.parse_args()

    filter(args.ref, args.vcf, args.bam, args.outdir, args.prefix, args.retrain, args.grid_search, 
            args.cores, args.seed, args.loglevel)