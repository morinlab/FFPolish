#!/usr/bin/env python

import os
import argparse as ap
from filter import filter
from extract import extract
import plot_features

VERSION = 0.1

if __name__ == '__main__':
    parser = ap.ArgumentParser(description='FFPolish - Filter Artifacts From FFPE Variant Calls\n'
                                           'Version {0}\n'
                                           'Copyright (C) 2020 Matthew Nguyen'.format(VERSION),
                               formatter_class=ap.RawTextHelpFormatter)

    parser.add_argument('-ll', '--loglevel', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Set the logging level')
    subparsers = parser.add_subparsers(dest='subcommand', required=True, 
                                       help='Choose an FFPolish tool')

    predict_parser = subparsers.add_parser('filter', help='Filter variants')
    predict_parser.add_argument('ref', metavar='REFERENCE', help='Reference genome FASTA')
    predict_parser.add_argument('vcf', metavar='VCF', help='VCF to filter')
    predict_parser.add_argument('bam', metavar='BAM', help='Tumor BAM file')
    predict_parser.add_argument('-o', '--outdir', metavar='DIR', default=os.path.join(os.getcwd(), 'results'), 
                        help='Output directory')
    predict_parser.add_argument('-c', '--cores', metavar='INT', default=1, type=int,
                        help='Number of cores to use for grid search and filtering (default: 1)')
    predict_parser.add_argument('-p', '--prefix', default=None, help='Output prefix (default: basename of BAM)')
    predict_parser.add_argument("-f", "--features", action="store_true", help="Generate an additional TSV file summarizing characteristics for each variant")
    predict_parser.add_argument("--plot_features", action="store_true", help="Generate figures summarizing variant characteristics")

    training_parser = predict_parser.add_argument_group("Model re-training")
    training_parser.add_argument('-rt', '--retrain', metavar='HDF5', default=None,
                        help='Retrain model with new data (in hdf5 format)')
    training_parser.add_argument('-gs', '--grid_search', action='store_true', default=False,
                        help='Perform grid search when retraining model')
    training_parser.add_argument('-s', '--seed', metavar='INT', default=667,
                        help='Seed for retraining (default: 667)')

    extract_parser = subparsers.add_parser('extract', help='Extract features for re-training')
    extract_parser.add_argument('ref', metavar='REFERENCE', help='Reference genome FASTA')
    extract_parser.add_argument('vcf', metavar='VCF', help='VCF to filter')
    extract_parser.add_argument('bam', metavar='BAM', help='Tumor BAM file')
    extract_parser.add_argument('outdir', metavar='DIR', help='Output directory')
    extract_parser.add_argument('-p', '--prefix', default=None, help='Output prefix (default: basename of BAM)')
    extract_parser.add_argument('--skip_bam_readcount', action='store_true', default=False,
                        help='Skip bam_readcount on sample')
    extract_parser.add_argument('--labels', metavar='BED', default=None,
                        help='BED file of true variants (to create pickle file of true variants)')
    extract_parser.add_argument('--pkl', metavar='PKL', default=None, 
                        help='Pickle file of true variants')

    plot_parser = subparsers.add_parser("plot", help="Visualize DeepSVR features")
    plot_parser.add_argument("input", metavar="TSV", help="Input TSV file listing DeepSVR features. Generated using \'filter -f \'")
    plot_parser.add_argument('outdir', metavar='DIR', help='Output directory')

    args = parser.parse_args()

    if args.subcommand == 'filter':
        filter(args.ref, args.vcf, args.bam, args.outdir, args.prefix, args.retrain, args.grid_search, 
                args.cores, args.features, args.plot_features, args.seed, args.loglevel)

    elif args.subcommand == 'extract':
        extract(args.ref, args.vcf, args.bam, args.outdir, args.prefix, args.skip_bam_readcount,
                args.labels, args.pkl)

    elif args.subcommand == 'plot':
        plot_features.main(args)

    else:
        raise parser.error("Unrecognized command: %s" % args.subcommand)
