#!/usr/bin/env python

import os
import argparse as ap
from predict_cli import predict

if __name__ == '__main__':
    parser = ap.ArgumentParser(description='FFPolish\n' +
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
    parser.add_argument('-rt', '--retrain', metavar='HDF5', default=None, 
                        help='Retrain model with new data (in hdf5 format)')
    parser.add_argument('-gs', '--grid_search', action='store_true', default=False,
                        help='Perform grid search when retraining model')
    parser.add_argument('-c', '--cores', metavar='INT', default=1, 
                        help='Number of cores to use for grid search (default: 1)')
    parser.add_argument('-s', '--seed', metavar='INT', default=667, 
                        help='Seed for retraining (default: 667)')

    args = parser.parse_args()

    predict(args.ref, args.vcf, args.bam, args.outdir, args.prefix, args.retrain, args.grid_search, 
            args.cores, args.seed, args.loglevel)