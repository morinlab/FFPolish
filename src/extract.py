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
from utils import *

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

PREFIX = '/projects/rmorin/projects/ffpe_ml'
DATA = os.path.join(PREFIX, 'data')

logger = logging.getLogger()

tqdm.pandas(ascii = True)

def extract(ref, gt, vcf, bam, outdir, prefix, skip_bam_readcount, cores, cleanup, loglevel):
    logging.basicConfig(level=loglevel,
                        format='%(asctime)s (%(relativeCreated)d ms) -> %(levelname)s: %(message)s',
                        datefmt='%I:%M:%S %p')

    # Check for required external commands
    logging.info('Checking for tools in $PATH')
    check_commands()

    if not prefix:
        prefix = os.path.basename(bam.split('.')[0])

    # Create directories
    tmpdir = outdir + os.sep + "tmp_bam_readcounts"
    tmpdir_obj = None
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    if cleanup:
        tmpdir_obj = tempfile.TemporaryDirectory()
        tmpdir = tmpdir_obj.name
    elif not os.path.exists(tmpdir):
        os.mkdir(tmpdir)

    try:
        true_vars = pd.read_pickle(gt)
    except:
        logger.info('Converting ground truth VCF to BED file')
        bed_file_path = os.path.join(tmpdir, prefix) + '.bed'
        var_keys = vcf_to_bed(gt, bed_file_path)

        logger.info('Converting ground truth variants to 1-based coordinates')
        true_vars = pd.read_csv(bed_file_path, sep='\t', index_col=None, header=None, dtype={0: str})
        true_vars.columns = ['chr', 'start', 'end', 'ref', 'alt']
        true_vars = true_vars.progress_apply(convert_one_based, axis=1)
        true_vars.to_pickle(os.path.join(outdir, 'true_vars.pkl'))

    logger.info('Converting VCF to BED file')
    bed_file_path = os.path.join(tmpdir, prefix) + '.bed'
    var_keys = vcf_to_bed(vcf, bed_file_path)

    logger.info('Preparing data')
    if cores <= 1:
        # Single-threaded
        logging.info("Running in single-threaded mode")
        df = get_deepsvr_attr(prefix, bam, bed_file_path, ref, tmpdir)

    else:
        # Multi-threading
        # Ok, this isn't going to be very fun. Since deep-svr requires files and external utilities, we need to split the input
        # bed file of variant coordinates into smaller bed files and run those separately
        # We also need to set a unique output path

        # First, lets split the BED file of variants
        # Open files
        path_prefix = os.path.join(tmpdir, prefix)
        open_files = list(open(path_prefix + ".j" + str(i) + ".bed", "w") for i in range(0, cores))

        with open(bed_file_path) as f:
            i = 0
            for line in f:
                x = i % cores  # Decide which output file to use
                open_files[x].write(line)
                i += 1

        # Close all output files
        for file in open_files:
            file.close()

        # SANITY CHECK: If we have less lines than jobs, remove the empty job files and adjust jobs accordinging
        if i < cores:
            for j in range(i, jobs):
                os.remove(path_prefix + ".j" + str(j) + ".bed")
                cores = i

        # Now lets actually obtain variant stats
        # Prepare arguments. The only arguments that will change are the bed file and prefix
        multiproc_args = []
        tmp_dirs_mp = []
        for i in range(0, cores):
            job_prefix = prefix + ".j" + str(i)
            job_bed_path = path_prefix +".j" + str(i) + ".bed"
            job_tmp_dir = tmpdir + ".j" + str(i)
            # Make job directory
            if cleanup:
                job_tmp_dir = tempfile.TemporaryDirectory()
                tmp_dirs_mp.append(job_tmp_dir)
            elif not os.path.exists(job_tmp_dir):
                os.mkdir(job_tmp_dir)

            multiproc_args.append([job_prefix,
                                   bam,
                                   job_bed_path,
                                   ref,
                                   job_tmp_dir.name])

        logging.info("Running using %s threads" % cores)

        # To prevent deadlocks due to some shared elements, spawn processes instead of forking them
        with multiprocessing.get_context("spawn").Pool(processes=cores) as proc_pool:
            # Moment of truth. Run the jobs
            var_attr_multi = proc_pool.starmap(get_deepsvr_attr, multiproc_args)

        # Merge the output. They *should* be in the same order as the jobs that were run
        df = pd.concat(var_attr_multi)

        # Cleanup tmp directories (if specified)
        for x in tmp_dirs_mp:
            x.cleanup()

    logger.info('Hard-filter variants')
    df = df[df.tumor_VAF > 0.05]
    df = df[df.tumor_depth > 10]
    df = df[df.tumor_var_num_minus_strand + df.tumor_var_num_plus_strand > 4]

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

    # Cleanup temp directory
    if cleanup and tmpdir_obj is not None:
        tmpdir_obj.cleanup()
