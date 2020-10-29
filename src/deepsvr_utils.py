import os
import re
import numpy as np
import pandas as pd
import pysam
import itertools
from sklearn import preprocessing
from zero_one_based_conversion import convert

# Title: String to numeric
# Author: javier http://stackoverflow.com/users/11649/javier
# Source: http://stackoverflow.com/a/379966
# License: MIT
def to_numeric(s):
    """Convert string to int or float

    Args:
        s (str): string to be converted to numeric

    Returns:
        int or float
    """
    try:
        return int(s)
    except ValueError:
        return float(s)

BASE_METRICS = ['count', 'avg_mapping_quality', 'avg_basequality',
                'avg_se_mapping_quality', 'num_plus_strand',
                'num_minus_strand', 'avg_pos_as_fraction',
                'avg_num_mismaches_as_fraction', 'avg_sum_mismatch_qualities',
                'num_q2_containing_reads',
                'avg_distance_to_q2_start_in_q2_reads',
                'avg_clipped_length', 'avg_distance_to_effective_3p_end', 
                'avg_distance_to_effective_5p_end']


# Title: PrepareData and ReadCount from DeepSVR
# Author: Griffith Lab
# Source: https://github.com/griffithlab/DeepSVR
# License: MIT
class ReadCount:
    """Parse bam-readcount out into dict or pandas.DataFrame"""

    def __init__(self, file_path):
        """Initialize dict of bam-readcount file.

        Args:
            file_path (str): File path of bam-readcount file
        """
        self.read_count_dict = self._parse(file_path)
        self.read_count_df = pd.DataFrame()

    def _parse(self, file_path):
        """Read and parse the bam_readcount file into a dict

        Args:
            file_path (str): File path of bam-readcount file

        Returns:
            dict of bam-readcount
        """
        file = open(file_path)
        counts = {}
        for line in file:
            match = re.match(r'(^[\w|.]+\t\d+\t\w\t\d+)', line)
            if match is not None:
                count = re.split('\t|:', line.strip())
                position = '{0}:{1}'.format(count[0], count[1])
                metrics = {}
                metrics['chromosome'] = count[0]
                metrics['position'] = int(count[1])
                metrics['ref'] = count[2]
                metrics['depth'] = int(count[3])
                bases = {}
                for i in range(4, len(count), 15):
                    b = list(map(to_numeric, count[i + 1: i + 15]))
                    # if not all(x == 0 for x in b):
                    if count[i] != '=':
                        bases[count[i]] = dict(zip(BASE_METRICS, b))
                metrics['bases'] = bases
                counts[position] = metrics
        return counts

    def compute_variant_metrics(self, var_bed_file_path,
                                sample_prepend_string):
        """

        Args:
            var_bed_file_path (str): filepath to bed file indicating variants
            sample_prepend_string (str): Column name to prepend sample e.g.
                                         tumor

        Returns:
             pandas.Dataframe of variant centric read count data
        """
        self.bam_readcount_keys = list(self.read_count_dict.keys())
        if len(self.read_count_df) > 0:
            return self.read_count_df
        with open(var_bed_file_path, 'r') as f:
            bed_file_header = False
            for line in f:
                line = line.strip().split('\t')
                # if review file has header, skip to the next line.
                if line[0].lower() == 'chromosome':
                    bed_file_header = True
                    continue
                if len(line) == 1:
                    print(line)
                else:
                    line = line[:6]
                    chromosome, start, stop, reference, variant= line
                start = int(start)
                stop = int(stop)
                bam_readcount_site = '{0}:{1}'.format(chromosome, start)
                variant_site = '{0}:{1}{2}>{3}'.format(chromosome, start,
                                                       reference,
                                                       variant)
                # If variant is duplicated in bed file move to next
                if variant_site in self.read_count_dict.keys():
                    continue
                # insertions
                if reference == '-':
                    try:
                        self.flatten_base_metrics(
                            self.read_count_dict[bam_readcount_site]['ref'],
                            bam_readcount_site, variant_site,
                            sample_prepend_string + '_ref')
                    except KeyError:
                        # bamreadcout does not output
                        # metrics for zero depth sites
                        self._add_zero_depth_readcount_to_dict(
                            reference, bam_readcount_site, variant_site,
                            sample_prepend_string + '_ref')
                    self.flatten_base_metrics('+{0}'.format(variant),
                                              bam_readcount_site, variant_site,
                                              sample_prepend_string + '_var')
                # deletions
                elif variant == '-':
                    try:
                        self.flatten_base_metrics(
                            self.read_count_dict[bam_readcount_site]['ref'],
                            bam_readcount_site, variant_site,
                            sample_prepend_string + '_ref')
                    except KeyError:
                        self._add_zero_depth_readcount_to_dict(
                            reference, bam_readcount_site, variant_site,
                            sample_prepend_string + '_ref')
                    self.flatten_base_metrics('-{0}'.format(reference),
                                              bam_readcount_site, variant_site,
                                              sample_prepend_string + '_var')
                # snvs
                else:
                    self.flatten_base_metrics(reference, bam_readcount_site,
                                              variant_site,
                                              sample_prepend_string + '_ref')
                    self.flatten_base_metrics(variant, bam_readcount_site,
                                              variant_site,
                                              sample_prepend_string + '_var')
                other_bases_count = 0
                try:
                    bases = self.read_count_dict[bam_readcount_site]['bases']
                    for base in bases:
                        # TODO check if other base metrics are
                        # nearly identical due to proximity
                        if base not in \
                                self.read_count_dict[bam_readcount_site][
                                    'seen_base']:
                            other_bases_count += \
                                self.read_count_dict[bam_readcount_site][
                                    'bases'][base]['count']
                    self.read_count_dict[bam_readcount_site].pop('seen_base')
                except KeyError:
                    print(chromosome, start, stop, reference, variant)
                    print(self.read_count_dict[bam_readcount_site])
                    raise KeyError
                self.read_count_dict[variant_site][sample_prepend_string +
                                                   '_other_bases_count']\
                    = other_bases_count
                # self.read_count_dict[bam_readcount_site].pop('bases')
                self.read_count_dict[variant_site]['chromosome'] = chromosome
                self.read_count_dict[variant_site]['ref'] = reference
                self.read_count_dict[variant_site]['var'] = variant
                self.read_count_dict[variant_site]['stop'] = stop
                self.read_count_dict[variant_site]['start'] = int(
                    self.read_count_dict[
                        bam_readcount_site]['position'])
                self.read_count_dict[variant_site][
                    sample_prepend_string + '_depth'] = \
                    self.read_count_dict[bam_readcount_site]['depth']

        # This will delete all the bam-readcount keys including count of
        # other positions from indels. This dropping the counts at other
        # positions could eliminate some real signal
        for key in self.bam_readcount_keys:
            self.read_count_dict.pop(key)
        self.read_count_df = pd.DataFrame.from_dict(self.read_count_dict,
                                                    orient='index')

        self.read_count_df[sample_prepend_string + '_VAF'] = \
            self.read_count_df[sample_prepend_string + '_var_count'] / \
            self.read_count_df[sample_prepend_string + '_depth']
        # set VAF to zero when denomitator is zero
        self.read_count_df.loc[self.read_count_df[
                                   sample_prepend_string + '_depth'] == 0,
                               sample_prepend_string + '_VAF'] = 0

        self.validate_bam_readcount_output(bed_file_header, var_bed_file_path)
        return self.read_count_df

    def validate_bam_readcount_output(self, bed_file_header,
                                      var_bed_file_path):
        """Determine if provided bed file and bam-readcount output return the
        same number of variants

            Args:
                bed_file_header (bool): Specify if bed file has header
                var_bed_file_path (str): File path of bed file
        """
        # Remove duplicated calls from bed files an check that the correct
        # number of variants were counted by bam-readcount
        if bed_file_header:
            bed_df = pd.read_csv(var_bed_file_path, sep='\t')
            bed_df.columns = map(str.lower, bed_df.columns)
            bed_df.rename(columns={'reference': 'ref', 'variant': 'var'},
                          inplace=True)
        else:
            bed_df = pd.read_csv(var_bed_file_path, sep='\t',
                                 names=['chromosome', 'start', 'stop', 'ref',
                                        'var'])
        bed_df = bed_df[['chromosome', 'start', 'stop', 'ref', 'var']]
        bed_df = bed_df.drop_duplicates()
        bed_df = self._remove_duplicated_calls(bed_df)
        if len(self.read_count_df) != len(bed_df):
            print('Bed file path:', var_bed_file_path)
            print('Bed file count: ', len(bed_df))
            print('Counts file count: ', len(self.read_count_df))
            raise ValueError(
                'Count and bed files return different number of variants')

    def remove_extra_indel_counts(self, chromosome, start, stop):
        for i in range(start + 1, stop + 1):
            remove_site = '{0}:{1}'.format(chromosome, i)
            self.read_count_dict.pop(remove_site)

    def flatten_base_metrics(self, base_key, bam_readcount_site, variant_site,
                             prepend_string):
        """ Renames the base metrics and flatten the data into the site level

        :param base_key (str): Key for the reference or variant
        :param site (str): Site string
        :param prepend_string (str): string to prepend the new keys
        """

        try:
            # Remove ref or var metrics so counts for other bases can be summed
            self.read_count_dict[bam_readcount_site].setdefault('seen_base',
                                                                []).append(
                base_key)
            for metric_key in \
                    self.read_count_dict[bam_readcount_site]['bases'][
                        base_key]:
                new_key = '{0}_{1}'.format(prepend_string, metric_key)
                self.read_count_dict.setdefault(variant_site, {})
                self.read_count_dict[variant_site][new_key] = \
                    self.read_count_dict[bam_readcount_site]['bases'][
                        base_key][metric_key]

        except KeyError:
            # bam-readcount does not output zero depth sites, this adds them to
            # the dictionary assuming that the keyError was raised on
            # the ref call
            if bam_readcount_site not in self.read_count_dict:
                self._add_zero_depth_readcount_to_dict(base_key,
                                                       bam_readcount_site,
                                                       variant_site,
                                                       prepend_string)
            else:
                # add all zero metrics for indels absent from count file
                keys = ['{0}_{1}'.format(prepend_string, i) for i in
                        BASE_METRICS]
                print(keys)
                self.read_count_dict.setdefault(variant_site, {}).update(
                    dict.fromkeys(keys, 0))

    def _remove_duplicated_calls(self, bed_df):
        """Removes calls with same coordinates but disagreeing calls

        Parameters:
            bed_df (pandas.DataFrame): DataFrame of manual review calls
        Returns:
            bed_df (pandas.DataFrame): DataFram of manual review calls with
                                       duplicates removed
        """

        duplicated_calls = bed_df[
            bed_df[['chromosome', 'start', 'stop', 'ref', 'var']].duplicated()]
        duplicated_call_keys = (duplicated_calls.chromosome.map(str) + ':' +
                                duplicated_calls.start.map(str) +
                                duplicated_calls.ref + '>' +
                                duplicated_calls['var'])
        duplicated_call_keys = duplicated_call_keys.values
        for key in duplicated_call_keys:
            self.read_count_dict.pop(key, None)
        return bed_df.loc[bed_df[['chromosome', 'start', 'stop',
                                  'ref', 'var']].drop_duplicates().index]

    def _add_zero_depth_readcount_to_dict(self, base_key, bam_readcount_site,
                                          variant_site, prepend_string):
        variant_site_search = re.search('(\w+):(\d+)[\w|-]+>[\w|-]+',
                                        variant_site)
        chromosome = variant_site_search.group(1)
        position = variant_site_search.group(2)
        self.bam_readcount_keys.append(bam_readcount_site)
        self.read_count_dict[bam_readcount_site] = {
            'bases': {'A': {'avg_basequality': 0.0,
                            'avg_clipped_length': 0.0,
                            'avg_distance_to_effective_3p_end': 0.0,
                            'avg_distance_to_effective_5p_end': 0.0,
                            'avg_distance_to_q2_start_in_q2_reads': 0.0,
                            'avg_mapping_quality': 0.0,
                            'avg_num_mismaches_as_fraction': 0.0,
                            'avg_pos_as_fraction': 0.0,
                            'avg_se_mapping_quality': 0.0,
                            'avg_sum_mismatch_qualities': 0.0,
                            'count': 0,
                            'num_minus_strand': 0,
                            'num_plus_strand': 0,
                            'num_q2_containing_reads': 0},
                      'C': {'avg_basequality': 0.0,
                            'avg_clipped_length': 0.0,
                            'avg_distance_to_effective_3p_end': 0.0,
                            'avg_distance_to_effective_5p_end': 0.0,
                            'avg_distance_to_q2_start_in_q2_reads': 0.0,
                            'avg_mapping_quality': 0.0,
                            'avg_num_mismaches_as_fraction': 0.0,
                            'avg_pos_as_fraction': 0.0,
                            'avg_se_mapping_quality': 0.0,
                            'avg_sum_mismatch_qualities': 0.0,
                            'count': 0,
                            'num_minus_strand': 0,
                            'num_plus_strand': 0,
                            'num_q2_containing_reads': 0},
                      'G': {'avg_basequality': 0.0,
                            'avg_clipped_length': 0.0,
                            'avg_distance_to_effective_3p_end': 0.0,
                            'avg_distance_to_effective_5p_end': 0.0,
                            'avg_distance_to_q2_start_in_q2_reads': 0.0,
                            'avg_mapping_quality': 0.0,
                            'avg_num_mismaches_as_fraction': 0.0,
                            'avg_pos_as_fraction': 0.0,
                            'avg_se_mapping_quality': 0.0,
                            'avg_sum_mismatch_qualities': 0.0,
                            'count': 0,
                            'num_minus_strand': 0,
                            'num_plus_strand': 0,
                            'num_q2_containing_reads': 0},
                      'N': {'avg_basequality': 0.0,
                            'avg_clipped_length': 0.0,
                            'avg_distance_to_effective_3p_end': 0.0,
                            'avg_distance_to_effective_5p_end': 0.0,
                            'avg_distance_to_q2_start_in_q2_reads': 0.0,
                            'avg_mapping_quality': 0.0,
                            'avg_num_mismaches_as_fraction': 0.0,
                            'avg_pos_as_fraction': 0.0,
                            'avg_se_mapping_quality': 0.0,
                            'avg_sum_mismatch_qualities': 0.0,
                            'count': 0,
                            'num_minus_strand': 0,
                            'num_plus_strand': 0,
                            'num_q2_containing_reads': 0},
                      'T': {'avg_basequality': 0.0,
                            'avg_clipped_length': 0.0,
                            'avg_distance_to_effective_3p_end': 0.0,
                            'avg_distance_to_effective_5p_end': 0.0,
                            'avg_distance_to_q2_start_in_q2_reads': 0.0,
                            'avg_mapping_quality': 0.0,
                            'avg_num_mismaches_as_fraction': 0.0,
                            'avg_pos_as_fraction': 0.0,
                            'avg_se_mapping_quality': 0.0,
                            'avg_sum_mismatch_qualities': 0.0,
                            'count': 0,
                            'num_minus_strand': 0,
                            'num_plus_strand': 0,
                            'num_q2_containing_reads': 0}},
            'chromosome': chromosome,
            'depth': 0,
            'position': position,
            'ref': base_key}
        self.flatten_base_metrics(base_key, bam_readcount_site, variant_site,
                                  prepend_string)

class PrepareData:
    """Prepare data for classification or training from bam and manual review
        files


    """

    def __init__(self, sample, bam, bed, ref, out_dir_path,
                 skip_readcount=False):
        """Assemble pandas.Dataframe of data

            Args:
                samples_file_path (str): File path of tab-separated
                                         file outlining the tumor bam path,
                                         normal bam path, and manual review
                                         sites file path (this should be a
                                         one-based tsv file containing
                                         chromosome, start, and stop),
                                         boolean indicating a solid tumor or
                                         liquid tumor, and reference fasta file
                                          path
                header (bool): True if header False otherwise.
                out_dir_path (str): path for output directory
                skip_readcount (bool): skip the read counting step by reading
                                       in the read count files from a prior run
                                       in the output directory.
        """
        self.sample = sample
        self.bam = bam
        self.bed = bed
        self.ref_file = ref
        self.out_dir_path = out_dir_path
        self.training_data = pd.DataFrame()
        self.categorical_columns = list()
        self.reference = pysam.FastaFile(self.ref_file)
        self._run_bam_readcount(skip_readcount)

    def _run_bam_readcount(self, skip_readcount):
        """Run bam-readcount on created sites file. Concatenate review calls.

            Args:
                skip_readcount (bool): Skip the read counting step by reading
                                       in the read count files from a prior run
                                       in the output directory.
        """
        out_dir_path = os.path.join(self.out_dir_path, 'readcounts')
        if not os.path.exists(out_dir_path):
            os.makedirs(out_dir_path)

        print('-----------------------------------------------------'
                '\nStarting preprocessing\n')

        sites_file_path = os.path.join(out_dir_path, self.sample + '.sites')
        if not os.path.exists(sites_file_path):
            print('Creating sites tsv')
            review = self._parse_bed_file(self.bed, sites_file_path, self.sample)
        else:
            print('Reading sites tsv')
            review = pd.read_csv(sites_file_path, sep='\t', index_col=None, header=None)

        print('Processing tumor bam file:\n\t{0}'.format(self.bam))
        tumor_readcount_file_path = '{0}/{1}_tumor' \
                                    '.readcounts'.format(out_dir_path,
                                                            self.sample)
        if not skip_readcount:
            os.system('bam-readcount -i -w 0 -l {0} -f {1} '
                        '{2} > {3}'.format(sites_file_path, self.ref_file,
                                           self.bam,
                                           tumor_readcount_file_path))

        tumor_rc = ReadCount(tumor_readcount_file_path)

        tumor_data = tumor_rc.compute_variant_metrics(self.bed+'.one_based', 'tumor')
        individual_df = tumor_data

        individual_df.index = (self.sample + '~' + individual_df.chromosome +
                                ':' + individual_df.start.map(str) + '-' +
                                individual_df.stop.map(str) +
                                individual_df.ref + '>' +
                                individual_df['var'])
        self.training_data = pd.concat([self.training_data, individual_df])
        self.training_data.drop(['chromosome', 'start', 'stop'],
                                axis=1, inplace=True)

    def _parse_bed_file(self, bed_file_path, sites_file_path, sample_name):
        manual_review = pd.read_csv(bed_file_path, sep='\t', header=None, index_col=None)
        manual_review.columns = ['chromosome', 'start', 'stop', 'ref', 'var']

        header = manual_review.columns

        manual_review['chromosome'] = manual_review['chromosome'].astype(str)

        if 'chr' in manual_review['chromosome'][0]:
            chrom = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chr10', 
                    'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19', 
                    'chr20', 'chr21', 'chr22', 'chrX', 'chrY']
        else:
            chrom = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15',
                     '16', '17', '18', '19', '20', '21', '22', 'X', 'Y']

        manual_review = manual_review[manual_review['chromosome'].isin(chrom)]

        manual_review = manual_review.apply(self._convert_one_based, axis=1)
        manual_review = manual_review.replace('', np.nan).dropna(how='all')
        manual_review.columns = header
        manual_review[['chromosome', 'start', 'stop']].to_csv(sites_file_path,
                                                              sep='\t',
                                                              index=False,
                                                              header=False)
        manual_review.to_csv(bed_file_path+'.one_based',
                             sep='\t', index=False, header=True)
        return manual_review

    def _convert_one_based(self, row):
        return pd.Series(convert.coordinate_system('\t'.join(map(str, row.values)),
                                                  'to_one_based').strip().split('\t'))
