#!/usr/bin/env python

import os
import argparse
import seaborn as sns
import matplotlib
import pandas as pd

def get_args():
    """
    Processes command line arguments

    :return: A namespace containing command line parameters and options
    """

    parser = argparse.ArgumentParser("Visualizes variant filtering features used by DeepSVR")
    parser.add_argument("-i", "--input", metavar="TSV", required=True, type=str, help="Input tab-delineated file containing DeepSVR features (see cli.py filter)")
    parser.add_argument("-o", "--outdir", metavar="/path/to/output/", required=True, type=str, help="Output directory for plots")

    return parser.parse_args()


def generate_plots(plot_data:pd.DataFrame, outdir:str):
    """
    Generate numerous plots summarizing variant characteristics

    :param plot_data: A pandas.Dataframe containing variant features (1 row per variant)
    :param outdir: A strong specifying an output folder, in which to place plots
    :return:
    """

    # Parse sample name
    sample_name = plot_data["Unnamed: 0"][0].split("~")[0]  # This is ugly, but its not my fault

    # Pair associated features (usually ref and alt) for multiplots
    plot_groups = {
        "allele_counts": ["tumor_depth", "tumor_var_count", "tumor_VAF"],
        "mapping_qual": ["tumor_ref_avg_mapping_quality", "tumor_var_avg_mapping_quality"],
        "base_qual": ["tumor_ref_avg_basequality", "tumor_var_avg_basequality"],
        "se_mapping_qual": ["tumor_ref_avg_se_mapping_quality", "tumor_var_avg_se_mapping_quality"],
        "strand_bias": ["tumor_ref_avg_pos_as_fraction", "tumor_var_avg_pos_as_fraction"],
        "mismatch": ["tumor_ref_avg_num_mismaches_as_fraction", "tumor_var_avg_num_mismaches_as_fraction"],
        "mismatch_qual": ["tumor_ref_avg_sum_mismatch_qualities", "tumor_var_avg_sum_mismatch_qualities"],
        "read2_num": ["tumor_ref_num_q2_containing_reads", "tumor_var_num_q2_containing_reads"],
        "clipping_length": ["tumor_ref_avg_clipped_length", "tumor_var_avg_clipped_length"],
        "dist_to_end": ['tumor_ref_avg_distance_to_effective_3p_end', 'tumor_ref_avg_distance_to_effective_5p_end',
                        'tumor_var_avg_distance_to_effective_3p_end',  'tumor_var_avg_distance_to_effective_5p_end']
    }

    extra_params = {
        "base_qual": {"xlim": [0, 40], "ylim": [0, 40]},
        "mapping_qual": {"xlim": [0, 60], "ylim": [0, 60]},
        "se_mapping_qual": {"xlim": [0, 60], "ylim": [0, 60]}
    }

    # For convinience and plot clarity
    plot_data["passed"] = list(x == 1 for x in plot_data["preds"])

    for g_name, group in plot_groups.items():

        plot_grid = sns.PairGrid(plot_data, vars=group, hue="passed", palette= ["red", "dodgerblue"])
        plot_grid.map_offdiag(sns.scatterplot)
        plot_grid.map_diag(sns.kdeplot)
        plot_grid.add_legend()

        if g_name in extra_params:
            plot_grid.set(**extra_params[g_name])

        plot_out_name = outdir + os.sep + sample_name + "." + g_name + ".png"
        plot_grid.savefig(plot_out_name)
    #matplotlib.pyplot.show()


def main(args=None):

    if args is None:
        args = get_args()

    # Load features
    plot_data = pd.read_csv(args.input, sep="\t")
    generate_plots(plot_data, args.outdir)

if __name__ == "__main__":
    main()