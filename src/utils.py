#!/usr/bin/env python

import os
import sys
import tempfile
import pandas as pd 
import numpy as np
import logging
import deepsvr_utils as dp
import joblib
import gzip
import pdb
import vaex
import re
import pyfaidx
import subprocess
import multiprocessing
import plot_features

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

def check_commands():
    """
    Validate that all external utilities are present and accessible via $PATH
    :return: None
    """

    bam_readcount_rc = subprocess.call(["bam-readcount"], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # If the exit code is 127 for either process, it means the accompanying process could not be found
    if bam_readcount_rc == 127:
        raise FileNotFoundError("Unable to locate \'bam-readcounts\'")


def get_deepsvr_attr(prefix, bam, bed_file_path, ref, tmpdir):

    prep_data = dp.PrepareData(prefix, bam, bed_file_path, ref, tmpdir)
    return prep_data.training_data


def vcf_to_bed(vcf_path: str, bed_path: str):
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
                o_ref = cols[3]
                raw_alt = cols[4]
                passed_filt = cols[6]

            except IndexError as e:
                raise AttributeError("Unable to parse line %s of the input VCF file \'%s\' due to missing columns " % (i, vcf_path)) from e

            # Only keep PASS/unflagged variants
            if passed_filt != "PASS" and passed_filt != ".":
                continue

            try:
                pos = int(pos)
            except TypeError as e:
                raise AttributeError("While processing line %s of the input VCF file, the position \'%s\' could not be converted" % (i, pos)) from e

            # Is this VCF entry a multiallelic variant?
            alt_alleles = raw_alt.split(",")
            for alt in alt_alleles:

                # Calculate the variant key based upon the original VCF entry (which we will compare against at the end when filtering)
                old_var_key = '{}:{}-{}{}>{}'.format(chrom, pos, pos, o_ref, alt)

                if alt == o_ref:  # Sanity check
                    raise AttributeError("While processing line %s of the input VCF file, the reference and alternate alleles are the same" % i)

                # Now, normalize alleles, since indels can be represented in different ways
                is_snv = False
                if o_ref.startswith(alt):  # Deletion with allele duplicated
                    l_alt = len(alt)
                    ref = o_ref[l_alt:]
                    alt = "-"

                    # Set start and end coordinates
                    start = pos - 1 + l_alt
                    end = start + len(ref) - 1

                elif alt in miss_base:  # Deletion which has already been adjusted
                    start = pos - 1
                    end = start + len(ref) - 1
                    ref = o_ref
                    alt = "-"

                elif alt.startswith(o_ref):  # Insertion with allele duplicated
                    l_ref = len(o_ref)
                    alt = alt[l_ref:]
                    ref = "-"

                    start = pos + l_ref - 1
                    end = start + 1

                elif o_ref in miss_base:  # Insertion which has already been adjusted
                    start = pos - 1
                    end = start + 1
                    ref = "-"

                else:  # SNVs or 1bp insertions/deletions
                    if len(o_ref) > 1 or len(alt) > 1:
                        raise AttributeError("Unable to process line %s of the input VCF file. Malformed VCF entry or DNP" % i)
                    start = pos - 1
                    end = pos
                    ref = o_ref
                    is_snv = True

                # Sanity check
                if start < 0:
                    raise AttributeError("Unable to process line %s of the input VCF file: Position is zero or negative" % i)

                # Write out BED entry
                outline = [chrom, str(start), str(end), ref, alt]
                o.write("\t".join(outline))
                o.write(os.linesep)

                # Calculate new key (which is generated by DeepSVR)
                new_key_start = start
                new_key_end = end

                if is_snv:
                    new_key_start = end
                # new_var_key = '{}:{}-{}{}>{}'.format(chrom, key_start_pos, key_end_pos, ref, alt)
                new_var_key = '{}:{}-{}{}>{}'.format(chrom, new_key_start, new_key_end, ref, alt)
                var_keys[old_var_key] = new_var_key

    return var_keys


def check_for_repeat(var_key: str, ref_genome: pyfaidx.Fasta, indel_repeat_threshold: int = 4):

    # Break up this indel key into its individual components
    try:
        chrom, remainder = var_key.split(":")
        chrom = chrom.split("~")[1]
        start, end = re.findall("[0-9][0-9]*", remainder)
        ref, alt = re.sub(".*[0-9]", "", remainder).split(">")
    except ValueError as e:
        raise ValueError("Unable to process variant string %s" % var_key) from e

    start_pos = int(start)

    logging.debug("%s: Examining variant for repetitive reference sequence" % str)
    if alt == "-":  # Deletion
        indel_seq = ref
        is_snv = False
    else:
        indel_seq = alt
        if ref == "-":
            is_snv = False
        else:
            is_snv = True

    # Check to  see if this even is an extension/contraction of an existing repeat
    # First, lets try to reduce this indel down to a unique core sequence
    # In a repetitive region, some events may delete multiple instances of the repeat
    # Ex. For reference AAAACTAAAACTAAAACTAAAACT, an insertion of AAAACTAAAACT may occur
    unique_seq = indel_seq
    if len(set(indel_seq)) != 1:  # Don't process homopolymers
        for i in range(2, int(len(indel_seq) / 2) + 1):
            substring_match = True
            test_seq = indel_seq[0:i]
            start = 0
            end = 0
            while True:
                start = end
                end = start + i
                sub_seq = indel_seq[start:end]
                if sub_seq == "":
                    break
                if indel_seq[start:end] != test_seq:
                    substring_match = False
                    break

            # Is this string a repetative substring of the full indel sequence?
            if substring_match:
                unique_seq = test_seq
                logging.debug("%s: Reduced indel sequence to %s" % (var_key, unique_seq))
                break

    seq_length = len(unique_seq)
    # If this is an SNV, lets get the bases adjacent to this mutation, instead of the current position itself
    if is_snv:
        start_pos += 1
    end_pos = start_pos + seq_length * indel_repeat_threshold

    # Obtain the reference sequence to be examined for repeats
    try:
        logging.debug(
            "%s: Parsing reference genome for sequence %s: %s-%s" % (var_key, chrom, start_pos, end_pos))
        ref_seq = ref_genome[chrom][start_pos:end_pos].seq
    except IndexError:  # i.e. this indel occurs near the end of the genome. This filter does not apply
        ref_seq = ""

    repeat_ext = False
    logging.debug("%s: Obtained reference sequence %s. Comparing to %s" % (var_key, ref_seq, unique_seq))
    if ref_seq != "":
        # Now, check each slice for the repeats
        repeat_ext = True
        for j in range(indel_repeat_threshold):
            c_pos = j * seq_length
            c_end = c_pos + seq_length
            if ref_seq[
               c_pos:c_end] != unique_seq:  # The reference sequence does not match. This indel sequence does not occur the specified number of times
                repeat_ext = False
                break

    if repeat_ext:  # This indel is an expansion/contraction of an existing repeat
        logging.debug("%s: Failed repetitive sequence filter" % var_key)
        return True
    else:
        return False