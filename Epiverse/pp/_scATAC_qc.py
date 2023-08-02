import scanpy as sc
import anndata
import pandas as pd
import numpy as np
from muon import atac as ac
import muon as mu
from typing import Union
import seaborn as sns
import matplotlib.pyplot as plt
import re
import os 

def cal_qc(adata:anndata.AnnData,
           fragments_file:str = '',
           fragments_tbi_file:str = '',
           Number_of_fragments_to_count:Union[int, float] = None,
           RefSeq_file:str = '',
           ):
    
    """
    This function is used to calculate QC metrics for scATAC-seq data.

    Parameters
    ----------
    adata: AnnData
        An AnnData object matrix of shape `n_obs` × `n_vars`. Rows correspond to cells and columns to peaks.
    fragments_file: the save path of fragments file
    fragments_tbi_file: the save path of fragments index file
    Number_of_fragments_to_count: int, float, optional (default: None)
        Number of fragments to count. If None, 1e4* adata.n_obs is used.
    RefSeq_file: the save path of gene annotation file

    Returns
    -------
    adata: AnnData
        Annotated data matrix with calculated QC metrics.
    """
    adata.uns['files']={'fragments': fragments_file}

    print('......Calculate QC metrics')
    # Calculate general qc metrics using scanpy
    sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)

    # Rename columns
    if 'n_features_per_cell' not in adata.obs.columns:
        adata.obs.rename(
        columns={"n_genes_by_counts": "n_features_per_cell",},inplace=True,)

    if 'total_fragment_counts' not in adata.obs.columns:    
        adata.obs.rename(
        columns={"total_counts": "total_fragment_counts",},inplace=True,)

    print('......Log-transform total counts')
    # log-transform total counts and add as column
    adata.obs["log_total_fragment_counts"] = np.log10(adata.obs["total_fragment_counts"])
    
    print('......Calculate the nucleosome signal across cells')
    # Calculate the nucleosome signal across cells
    if Number_of_fragments_to_count is None:
        Number_of_fragments_to_count = int(adata.n_obs * 1e4)
    else:
        Number_of_fragments_to_count = int(Number_of_fragments_to_count) 
    ac.tl.nucleosome_signal(adata, n=Number_of_fragments_to_count)

    print('......Process the gene annotation file')
    features=pd.read_csv(RefSeq_file, sep="\t", header=None, comment="#")
    features.columns=pd.Index([
        "seqname", "source", "feature", "start", "end",
        "score", "strand", "frame", "attribute"])
    pattern = re.compile(r'([^\s]+) "([^"]+)";')
    splitted = pd.DataFrame.from_records(np.vectorize(lambda x: {
        key: val for key, val in pattern.findall(x)
    })(features["attribute"]), index=features.index)
    if set(features.columns).intersection(splitted.columns):
        self.logger.warning(
            "Splitted attribute names overlap standard GTF fields! "
            "The standard fields are overwritten!"
        )
    features=features.assign(**splitted)
    features=features.loc[features['feature']=='gene']
    new_features = features[['seqname', 'start', 'end', 'gene_id', 'gene_name']].copy()
    new_features.index = new_features['gene_name']
    new_features.rename(columns={'seqname': 'Chromosome', 'start': 'Start', 'end': 'End'}, inplace=True)

    print('......Calculate the TSS enrichment score')
    tss = ac.tl.tss_enrichment(adata, n_tss=3000, features=new_features,random_state=666)

    print('......Calculate QC metrics successfully')
    return adata



def plot_qc(adata:anndata.AnnData,
            save:bool = False,
            save_dir:str = '',):

    """
    This function is used to plot QC metrics for scATAC-seq data.

    Parameters
    ----------
    adata: AnnData
        An AnnData object of shape `n_obs` × `n_vars`. Rows correspond to cells and columns to peaks.
        This is the output of `cal_qc` function.
    save: bool, optional (default: False)
        Whether to save the image as a PNG format.
    save_dir: str, optional (default: '')
        The path to save the image.
    """

    print('......Plot QC metrics')
    # Figure_1
    # Plot the distribution of the TSS score
    print('......Plot the distribution of the TSS score')
    fig, axs = plt.subplots(1, 2, figsize=(6, 3))

    p1 = sns.histplot(adata.obs, x="tss_score", ax=axs[0])
    p1.set_title("Full range")

    p2 = sns.histplot(
        adata.obs,
        x="tss_score",
        binrange=(0, adata.obs["tss_score"].quantile(0.995)),
        ax=axs[1],
    )
    p2.set_title("Up to 99.5% percentile")
    plt.suptitle("Distribution of the TSS score")
    plt.tight_layout()
    # Save the image as a PNG format.
    if save==True:
        plt.savefig(os.path.join(save_dir,"tss_score_distribution.png"))
    plt.show()
    

    # Figure_2
    # Violin plots of nucleosome signal and TSS score.
    print('......Plot Violin plots of nucleosome signal and TSS score.')
    # These were identified by looking at the plots in this code cell before.
    fig, axs = plt.subplots(1, 2, figsize=(7, 3))

    # tss.score
    p3 = sc.pl.violin(adata, "tss_score", show=False,ax=axs[0])
    
    # nucleosome signal
    p4 = sc.pl.violin(adata, "nucleosome_signal", show=False,ax=axs[1])
    plt.suptitle("Violin plots of nucleosome signal and TSS score")
    plt.tight_layout()
    # Save the image as a PNG format.
    if save==True:
        plt.savefig(os.path.join(save_dir,"Violin_plots_of_nucleosome_signal_and_TSS_score.png"))
    plt.show()


    # Figure_3
    # Density plots of log_total_fragment_counts and tss_score
    print('......Plot Density plots of log_total_fragment_counts and tss_score.')

    # Scatter plot & histograms
    plot_tss_max = adata.obs['tss_score'].quantile(0.995)
    p5 = sns.jointplot(
    data=adata[(adata.obs["tss_score"] < plot_tss_max)].obs,
    x="log_total_fragment_counts",
    y="tss_score",
    color="black",
    marker=".",
    )
    # Density plot including lines
    p5.plot_joint(sns.kdeplot, fill=True, cmap="Blues", zorder=1, alpha=0.75)
    p5.plot_joint(sns.kdeplot, color="black", zorder=2, alpha=0.75)
    # Save the image as a PNG format.
    if save==True:
        plt.savefig(os.path.join(save_dir,"Density_plots.png"))
    plt.show()


    # Figure_4
    # Scatter plot total fragment count by number of features
    print('......Plot Scatter plot total fragment count by number of features in low counts.')
    p6 = sc.pl.scatter(
        adata[adata.obs.total_fragment_counts < 5000],
        x="total_fragment_counts",
        y="n_features_per_cell",
        size=100,
        color="tss_score",
        show=False,
    )
    # Save the image as a PNG format.
    if save==True:
        plt.savefig(os.path.join(save_dir,"Scatter_plots_low_counts.png"))
    plt.show()


    # Figure_5
    # Plot total counts of fragments & features colored by TSS score
    # Set thresholds for upper boundaries.
    # These were identified by looking at the plots in this code cell before.
    fig, axs = plt.subplots(1, 1, figsize=(3, 3))
    total_count_upper = 100000
    tss_upper = 50
    nucleosome_signal_upper = 2


    # Plot total counts of fragments & features colored by TSS score
    print('......Plot total counts of fragments & features colored by TSS score.')
    p7 = sc.pl.scatter(
        adata,
        x="total_fragment_counts",
        y="n_features_per_cell",
        size=40,
        color="tss_score",
        show=False,  # so that funstion output axis object where threshold line can be drawn.
        ax=axs,
    )
    # Save the image as a PNG format.
    if save==True:
        plt.savefig(os.path.join(save_dir,"Scatter_plots_of_features_and_fragments.png"))
    plt.show()


def filter_qc(adata:anndata.AnnData,
              tresh=None,):
    """
    Filter cells based on QC metrics.

    Parameters
    ----------
    adata: An AnnData object of shape `n_obs` × `n_vars`. Rows correspond to cells and columns to peaks.
    tresh: A dictionary of QC thresholds. The keys should be 'fragment_counts_min', 'fragment_counts_max',
         'features_per_cell_min', 'TSS_score_min', 'TSS_score_max', 'Nucleosome_singal_max', 'cells_by_counts_min'.
            Only used if mode is 'seurat'. Default is None.

    Returns
    -------
    adata：An AnnData object containing cells that passed QC filters.

    """

    if tresh is None :
        tresh={'fragment_counts_min': 2000,
                'fragment_counts_max': 100000,
                'features_per_cell_min': 1000, # There is a nearly linear relationship of the total fragment count with the number of features per cell.
                'TSS_score_min': 0.1,
                'TSS_score_max': 50,
                'Nucleosome_singal_max': 4,
                'cells_by_counts_min': 15
                }
    
    # Filter cells based on QC metrics
    print(f"Total number of cells: {adata.n_obs}")
    print(f"Total number of peaks: {adata.n_vars}")
    mu.pp.filter_obs(
        adata,
        "total_fragment_counts",
        lambda x: (x >= tresh['fragment_counts_min']) & (x <= tresh['fragment_counts_max']),
    )
    print(f"......Number of cells after filtering on total_fragment_counts: {adata.n_obs}")

    mu.pp.filter_obs(adata, "n_features_per_cell", lambda x: x >= tresh['features_per_cell_min'])
    print(f"......Number of cells after filtering on n_features_per_cell: {adata.n_obs}")

    mu.pp.filter_obs(
        adata,
        "tss_score",
        lambda x: (x >= tresh['TSS_score_min']) & (x <= tresh['TSS_score_max']),
    )
    print(f"......Number of cells after filtering on tss_score: {adata.n_obs}")
    
    mu.pp.filter_obs(adata, "nucleosome_signal", lambda x: x <= tresh['Nucleosome_singal_max'])
    print(f"......Number of cells after filtering on nucleosome_signal: {adata.n_obs}")

    # filtered out peaks that are detected in less than {cells_by_counts_min} cells
    mu.pp.filter_var(adata, "n_cells_by_counts", lambda x: x >= tresh['cells_by_counts_min'])
    print(f"......filtered out peaks that are detected in less than {tresh['cells_by_counts_min']} cells")
    print(f"......Number of peaks after filtering on cells_by_counts_min: {adata.n_vars}")

    return adata