#!/usr/bin/env python

import os
import argparse
import numpy
from matplotlib import pyplot
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
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


def generate_plots(plot_data:pd.DataFrame, outdir:str, sample_name:str = None):
    """
    Generate numerous plots summarizing variant characteristics

    :param plot_data: A pandas.Dataframe containing variant features (1 row per variant)
    :param outdir: A strong specifying an output folder, in which to place plots
    :param sample_name: A string specifying a sample name to be used for output plots
    :return: None
    """

    # Parse sample name
    if sample_name is None:
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

    # Since there can be a wide range of allele counts with extreme outliers, normalize
    allele_alt_low = int(numpy.quantile(plot_data["tumor_var_count"], 0.05))
    allele_alt_high = int(numpy.quantile(plot_data["tumor_var_count"], 0.95))
    allele_depth_low = int(numpy.quantile(plot_data["tumor_depth"], 0.05))
    allele_depth_high = int(numpy.quantile(plot_data["tumor_depth"], 0.95))
    restrict = lambda x, min_n, max_n: min_n if x < min_n else max_n if x > max_n else x
    plot_data["tumor_var_count"] = list(restrict(x, allele_alt_low, allele_alt_high) for x in plot_data.tumor_var_count)
    plot_data["tumor_depth"] = list(restrict(x, allele_depth_low, allele_depth_high) for x in plot_data.tumor_depth)

    extra_params = {
        "base_qual": {"xlim": [0, 40], "ylim": [0, 40]},
        "mapping_qual": {"xlim": [0, 60], "ylim": [0, 60]},
        "se_mapping_qual": {"xlim": [0, 60], "ylim": [0, 60]}
    }

    # For convinience and plot clarity
    plot_data["passed"] = list(x == 1 for x in plot_data["preds"])

    for g_name, group in plot_groups.items():

        plot_grid = sns.PairGrid(plot_data, vars=group, hue="passed", palette=["red", "dodgerblue"])
        plot_grid.map_offdiag(sns.scatterplot, s=3)
        plot_grid.map_diag(sns.kdeplot)
        plot_grid.add_legend()
        plot_grid.fig.suptitle(sample_name)

        if g_name in extra_params:
            plot_grid.set(**extra_params[g_name])

        plot_out_name = outdir + os.sep + sample_name + "." + g_name + ".png"
        plot_grid.savefig(plot_out_name)

    pyplot.close("all")


def generate_pca_plot(plot_data:pd.DataFrame, outdir:str, sample_name:str = None):
    """
    Runs a principle component analysis on variants filtered using DeepSVR

    Runs two PCAs. One with as many components are there are dimensions to determine the optimal number of components to
    use, and a second using only two components for plotting purposes

    :param plot_data: A pandas.Dataframe containing variant features (1 row per variant)
    :param outdir: A strong specifying an output folder, in which to place plots
    :param sample_name: A string specifying a sample name to be used for output plots
    :return: None
    """


    # Parse sample name
    if sample_name is None:
        sample_name = plot_data["Unnamed: 0"][0].split("~")[0]  # This is ugly, but its still not my fault

    # Features to be scaled
    indep_var = ['tumor_ref_count', 'tumor_ref_avg_mapping_quality', 'tumor_ref_avg_basequality', 'tumor_ref_avg_se_mapping_quality',
                 'tumor_ref_num_plus_strand', 'tumor_ref_num_minus_strand', 'tumor_ref_avg_pos_as_fraction', 'tumor_ref_avg_num_mismaches_as_fraction',
                 'tumor_ref_avg_sum_mismatch_qualities', 'tumor_ref_num_q2_containing_reads', 'tumor_ref_avg_distance_to_q2_start_in_q2_reads',
                 'tumor_ref_avg_clipped_length', 'tumor_ref_avg_distance_to_effective_3p_end', 'tumor_ref_avg_distance_to_effective_5p_end',
                 'tumor_var_count', 'tumor_var_avg_mapping_quality', 'tumor_var_avg_basequality', 'tumor_var_avg_se_mapping_quality',
                 'tumor_var_num_plus_strand', 'tumor_var_num_minus_strand', 'tumor_var_avg_pos_as_fraction', 'tumor_var_avg_num_mismaches_as_fraction',
                 'tumor_var_avg_sum_mismatch_qualities', 'tumor_var_num_q2_containing_reads', 'tumor_var_avg_distance_to_q2_start_in_q2_reads',
                 'tumor_var_avg_clipped_length', 'tumor_var_avg_distance_to_effective_3p_end', 'tumor_var_avg_distance_to_effective_5p_end',
                 'tumor_other_bases_count', 'tumor_depth', 'tumor_VAF']

    plot_indep_features = plot_data[indep_var]
    # Standardize
    plot_indep_features = StandardScaler().fit_transform(plot_indep_features)

    # Run initial PCA
    pca = PCA()
    pca.fit(plot_indep_features)
    # Plot the PCA to determine the optimal number of features
    var_ratio = pd.DataFrame({"Cumulative Explained Variance": numpy.cumsum(pca.explained_variance_ratio_),
                              "Number of Components":list(x + 1 for x in range(0, len(pca.explained_variance_ratio_)))})  # I apologize

    comp_plot = sns.barplot(data=var_ratio, color="dodgerblue", x="Number of Components", y="Cumulative Explained Variance")
    comp_plot.set(title=sample_name)
    comp_plot = comp_plot.get_figure()

    # Now generate a PCA with two dimension and adjust
    pca = PCA(n_components=2)
    two_comp_features = pca.fit_transform(plot_indep_features)
    two_comp_features = pd.DataFrame(two_comp_features, columns = ["PC1", "PC2"])
    two_comp_features["passed"] = list(x == 1 for x in plot_data["preds"])

    two_comp_plot = sns.jointplot(data=two_comp_features, x="PC1", y="PC2", hue="passed", palette=["red", "dodgerblue"], joint_kws={"s":3})
    two_comp_plot.fig.suptitle(sample_name)

    # Save these plots
    comp_plot_path = outdir + os.sep + sample_name + ".pca_components.png"
    comp_plot.savefig(comp_plot_path)

    two_comp_path = outdir + os.sep + sample_name + ".pca_two_components.png"
    two_comp_plot.savefig(two_comp_path)
    pyplot.close("all")


def main(args=None):

    if args is None:
        args = get_args()

    # Load features
    plot_data = pd.read_csv(args.input, sep="\t")
    generate_plots(plot_data, args.outdir)

    generate_pca_plot(plot_data, args.outdir)

if __name__ == "__main__":
    main()
