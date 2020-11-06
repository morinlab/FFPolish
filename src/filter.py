#!/usr/bin/env python

import os
import pandas as pd 
import numpy as np
import logging
import deepsvr_utils as dp
import joblib
import gzip
import pdb
import vaex
import re
import subprocess
import multiprocessing

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV

BASE = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


def check_commands():
    """
    Validate that all external utilities are present and accessible via $PATH
    :return: None
    """

    vcf2bed_rc = subprocess.call(["vcf2bed"], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    bam_readcount_rc = subprocess.call(["bam-readcount"], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # If the exit code is 127 for either process, it means the accompanying process could not be found
    if vcf2bed_rc == 127:
        raise FileNotFoundError("Unable to locate \'vcf2bed\', Ensure the \'bedops\' package is installed")
    elif bam_readcount_rc == 127:
        raise FileNotFoundError("Unable to locate \'bam-readcounts\'")


def get_deepsvr_attr(prefix, bam, bed_file_path, ref, tmpdir):

    prep_data = dp.PrepareData(prefix, bam, bed_file_path, ref, tmpdir)
    return prep_data.training_data


def vcf_to_bed(vcf_path:str, bed_path:str):
    """
    Quickly coverts a VCF file (1-based) to a simple BED file with alleles included
    The following columns are output:
    - chromosome
    - start
    - end
    - ref
    - alt
    :param vcf_path: A string containing a filepath to the input VCF file. May be gzipped
    :param bed_path: A string specifying a filepath to the output BED file
    :return: A dictionary listing {orig_var_key: new_var_key}
    """

    var_keys = {}

    # VCF attributes used to symbolize a missing base (for indels)
    miss_base = ['-', '.', '0']
    # Check to see if the input VCF is gzipped
    with open(vcf_path, "rb") as test_f:
        if test_f.read(2) == b'\x1f\x8b':
            is_gzipped = True
        else:
            is_gzipped = False

    with (gzip.open(vcf_path, "rt") if is_gzipped else open(vcf_path, "r")) as f, open(bed_path, "w") as o:

        i = 0
        for line in f:
            i += 1
            if line.startswith("#"):  # Skip comment and header lines
                continue

            line = line.rstrip("\n").rstrip("\r")
            cols = line.split("\t")

            # Parse out the columns we care about
            try:
                chrom = cols[0]
                pos = cols[1]
                ref = cols[3]
                alt = cols[4]
                # Calculate the variant key based upon the origin VCF (which we will compare against at the end when filtering
                old_var_key = '{}:{}-{}{}>{}'.format(chrom, pos, pos, ref, alt)
            except IndexError as e:
                raise AttributeError("Unable to parse line %s of the input VCF file \'%s\' due to missing columns " % (i, vcf_path)) from e

            try:
                pos = int(pos)
            except TypeError as e:
                raise AttributeError("While processing line %s of the input VCF file, the position \'%s\' could not be converted" % (i, pos)) from e

            # Is this an indel??
            l_ref = len(ref)
            l_alt = len(alt)

            # Start coordinate
            start = pos - 1

            # These will be 1-based
            key_start_pos = pos
            key_end_pos = pos

            # DNPs
            if l_ref > 1 and l_alt > 1:
                raise NotImplementedError("DNPs are not currently supported")
            # Deletions
            elif l_ref > 1:
                # Has the alt allele already been adjusted?
                if alt in miss_base:
                    # In this case, just adjust the end coordinate
                    end = pos + l_ref
                else:
                    # This deletion is duplicating the reference allele in the alt and ref entry. Adjust the coordinates and alleles
                    alt = "-"
                    ref = ref[1:]
                    start = pos
                    end = start + l_ref - 1
                    key_start_pos += 1

                key_end_pos += len(ref)

            # Insertion
            elif l_alt > 1:
                # Has the reference base already been adjusted?
                if ref in miss_base:
                    # The end coordinate is very simple since its the next base in all cases
                    end = start + 1
                else:
                    # The reference allele is specified. Remove it, and adjust the coordinates
                    ref = "-"
                    alt = alt[1:]
                    start = pos
                    end = start + 1

                key_end_pos += 1

            # SNVs
            else:
                end = start + 1

            # Sanity check
            if start < 0:
                raise AttributeError("Unable to process line %s of the input VCF file: Position is zero or negative" % i)

            # Write out BED entry
            outline = [chrom, str(start), str(end), ref, alt]
            o.write("\t".join(outline))
            o.write(os.linesep)

            # Calculate new key (which is generated by DeepSVR)
            new_var_key = '{}:{}-{}{}>{}'.format(chrom, key_start_pos, key_end_pos, ref, alt)
            var_keys[old_var_key] = new_var_key

    return var_keys


def filter(ref, vcf, bam, outdir, prefix, retrain, grid_search, cores, seed, loglevel):
    logging.basicConfig(level=loglevel,
                        format='%(asctime)s (%(relativeCreated)d ms) -> %(levelname)s: %(message)s',
                        datefmt='%I:%M:%S %p')

    logger = logging.getLogger()

    # Check for required external commands
    logging.info('Checking for tools in $PATH')
    check_commands()

    logger.info('Running FFPolish prediction')

    if not prefix:
        prefix = os.path.basename(os.path.join(outdir, bam.split('.')[0]))
    
    if retrain:
        orig_train_df = vaex.open(os.path.join(BASE, 'orig_train.hdf5'))
        train_features = orig_train_df.get_column_names(regex='tumor')
        new_train_df = vaex.open(retrain)

        orig_train_df = orig_train_df.to_pandas_df(train_features + ['real'])
        new_train_df = new_train_df.to_pandas_df(train_features + ['real'])

        clf = LogisticRegression(penalty='l2', random_state=seed, solver='saga', max_iter=10000, 
                                 class_weight='balanced', C=0.001)
        scaler = MinMaxScaler()

        logger.info('Concatenate old and new training feature matrices')
        train_df = pd.concat([orig_train_df, new_train_df], ignore_index=True)
        X = train_df[train_features]
        y = train_df['real']

        logger.info('Scaling training feature matrix')
        X = scaler.fit_transform(X)

        logger.info('Training model')
        if grid_search:
            logger.info('Training using grid search')
            param = {'C': [1e-3, 1e-2, 1e-1, 1]}
            metrics = ['accuracy', 'precision', 'recall', 'roc_auc', 'f1']
            gs = GridSearchCV(clf, param, n_jobs=cores, scoring=metrics, cv=10, refit='f1')
            gs.fit(X, y)
            clf = gs.best_estimator_
        else:
            logger.info('Training using previous optimized parameters')
            clf.fit(X, y)

    else:
        clf = joblib.load(os.path.join(BASE, 'models', 'trained.clf'))
        scaler = joblib.load(os.path.join(BASE, 'models', 'trained.scaler'))

    tmpdir = outdir + os.sep + "tmp_bam_readcounts"
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    if not os.path.exists(tmpdir):
        os.mkdir(tmpdir)

    logger.info('Converting VCF to bed file')
    bed_file_path = os.path.join(tmpdir, prefix) + '.bed'
    var_keys = vcf_to_bed(vcf, bed_file_path)

    logger.info('Preparing data')
    if cores == 1:
        # Single-threaded
        logging.info("Running in single-threaded mode")
        df = get_deepsvr_attr(prefix, bam, bed_file_path, ref, tmpdir)

    else:
        # Multi-threading
        # Ok, this isn't going to be very fun. Since deep-svr requires files and external utilities, we need to split the input
        # bed file of variant coordinates into smaller bed files and run those separately

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
            for j in range (i, jobs):
                os.remove(path_prefix +".j" + str(j) + ".bed")
                cores = i


        # Now lets run the variant stats
        # Prepare arguments. The only arguments that will change are the bed file and prefix
        multiproc_args = []
        for i in range(0, cores):
            job_prefix = prefix + ".j" + str(i)
            job_bed_path = path_prefix +".j" + str(i) + ".bed"
            multiproc_args.append([job_prefix,
                                   bam,
                                   job_bed_path,
                                   ref,
                                   tmpdir])
        logging.info("Running using %s threads" % cores)

        proc_pool = multiprocessing.Pool(processes=cores)

        # Moment of truth. Run the jobs
        var_attr_multi = proc_pool.starmap(get_deepsvr_attr, multiproc_args)

        # Merge the output. They *should* be in the same order as the jobs that were run
        df = pd.concat(var_attr_multi)

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
    if cores == 1:
        kept_vars = list(df.index.str.replace(prefix + '~', ''))
    else:
        kept_vars = list(df.index.str.replace(prefix + '.j[0-9]*~', ''))

    # Is the input VCF gzipped?
    with open(vcf, "rb") as test_f:
        if test_f.read(2) == b'\x1f\x8b':
            is_gzipped = True
        else:
            is_gzipped = False

    vcf_file_path = os.path.join(outdir, prefix + '_filtered.vcf')
    with open(vcf_file_path, 'w') as f_out:
        with (gzip.open(vcf, 'rt') if is_gzipped else open(vcf, "r")) as f_in:
            for line in f_in:
                if not line.startswith('#'):
                    split = line.split('\t')
                    var = '{}:{}-{}{}>{}'.format(split[0], split[1], split[1], split[3], split[4])
                    if var_keys[var] in kept_vars:
                        f_out.write(line)
                else:
                    f_out.write(line)
