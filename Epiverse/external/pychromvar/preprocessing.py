from typing import Union
import re
import numpy as np
import scipy as sp
from anndata import AnnData
from mudata import MuData

from tqdm import tqdm



def get_bg_peaks(data: Union[AnnData, MuData], niterations=50, n_jobs=-1):
    """Find background peaks based on GC bias and number of reads per peak

    Parameters
    ----------
    data : Union[AnnData, MuData]
        AnnData object with peak counts or MuData object with 'atac' modality
    niterations : int, optional
        Number of background peaks to sample,, by default 50
    n_jobs : int, optional
        Number of cpus for compute. If set to -1, all cpus will be used, by default -1

    Returns
    -------

    updates `data`.
    """
    from pynndescent import NNDescent

    if isinstance(data, AnnData):
        adata = data
    elif isinstance(data, MuData) and "atac" in data.mod:
        adata = data.mod["atac"]
    else:
        raise TypeError(
            "Expected AnnData or MuData object with 'atac' modality")

    # check if the object contains bias in Anndata.varm
    assert "gc_bias" in adata.var.columns, "Cannot find gc bias in the input object, please first run add_gc_bias!"

    reads_per_peak = np.log1p(adata.X.sum(axis=0)) / np.log(10)

    # here if reads_per_peak is a numpy matrix, convert it to array
    if isinstance(reads_per_peak, np.matrix):
        reads_per_peak = np.squeeze(np.asarray(reads_per_peak))

    mat = np.array([reads_per_peak, adata.var['gc_bias'].values])
    chol_cov_mat = np.linalg.cholesky(np.cov(mat))
    trans_norm_mat = sp.linalg.solve_triangular(
        a=chol_cov_mat, b=mat, lower=True).transpose()

    index = NNDescent(trans_norm_mat, metric="euclidean",
                      n_neighbors=niterations, n_jobs=n_jobs)
    knn_idx, _ = index.query(trans_norm_mat, niterations)

    adata.varm['bg_peaks'] = knn_idx

    return None


def add_peak_seq(data: Union[AnnData, MuData], genome_file: str, delimiter="-"):
    """Add the DNA sequence of each peak to data object.

    Parameters
    ----------
    data : Union[AnnData, MuData]
        AnnData object with peak counts or MuData object with 'atac' modality.
    genome_file : str
        Filename of genome reference
    delimiter : str, optional
        Delimiter that separates peaks, by default "-"

    Returns
    -------
    Update `data`
    """
    from pysam import Fastafile

    if isinstance(data, AnnData):
        adata = data
    elif isinstance(data, MuData) and "atac" in data.mod:
        adata = data.mod["atac"]
    else:
        raise TypeError(
            "Expected AnnData or MuData object with 'atac' modality")

    fasta = Fastafile(genome_file)
    adata.uns['peak_seq'] = [None] * adata.n_vars

    for i in tqdm(range(adata.n_vars)):
        peak = re.split(delimiter, adata.var_names[i])
        chrom, start, end = peak[0], int(peak[1]), int(peak[2])
        adata.uns['peak_seq'][i] = fasta.fetch(chrom, start, end).upper()

    return None

def add_peak_seq_scvi(
    data: Union[AnnData, MuData],
    genome_name: str = "GRCh38",
    genome_dir: str = "data",
    chr_var_key: str = "chr",
    start_var_key: str = "start",
    end_var_key: str = "end",
):
    if isinstance(data, AnnData):
        adata = data
    elif isinstance(data, MuData) and "atac" in data.mod:
        adata = data.mod["atac"]
    else:
        raise TypeError(
            "Expected AnnData or MuData object with 'atac' modality")
    
    import scvi
    scvi.data.add_dna_sequence(
        adata,
        genome_name=genome_name,
        genome_dir=genome_dir,
        chr_var_key=chr_var_key,
        start_var_key=start_var_key,
        end_var_key=end_var_key,
    )

def add_gc_bias(data: Union[AnnData, MuData]):
    """Compute GC bias for each peak.

    Parameters
    ----------
    data : Union[AnnData, MuData]
        AnnData object with peak counts or MuData object with 'atac' modality.

    Returns
    -------
    Update data
    """

    if isinstance(data, AnnData):
        adata = data
    elif isinstance(data, MuData) and "atac" in data.mod:
        adata = data.mod["atac"]
    else:
        raise TypeError(
            "Expected AnnData or MuData object with 'atac' modality")

    assert "peak_seq" in adata.uns_keys(
    ), "Cannot find sequences, please first run add_peak_seq!"

    bias = np.zeros(adata.n_vars)

    for i in tqdm(range(adata.n_vars)):
        if 'peak_seq' in adata.uns_keys():
            seq = adata.uns['peak_seq'][i]

            freq_a = seq.count("A")
            freq_c = seq.count("C")
            freq_g = seq.count("G")
            freq_t = seq.count("T")

            if freq_a + freq_c + freq_g + freq_t == 0:
                bias[i] = 0.5
            else:
                bias[i] = (freq_g + freq_c) / (freq_a + freq_c + freq_g + freq_t)
        elif 'dna_sequence' in adata.varm.keys():
            seq = adata.varm['dna_sequence'].iloc[i]
            #seq是pandas的一行，那么直接count肯定会报错，我需要把它变成一个字符串
            seq = seq.to_string()

            freq_a = seq.count("A")
            freq_c = seq.count("C")
            freq_g = seq.count("G")
            freq_t = seq.count("T")

    adata.var['gc_bias'] = bias

    return None
