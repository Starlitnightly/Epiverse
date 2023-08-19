from scipy.sparse import dok_matrix
from scipy.sparse import csc_matrix
import scanpy as sc
import pandas as pd
import numpy as np
import anndata
import tables
import os
import re


# Define a function named ExtractGeneInfo to extract gene information from a gene bed file.
def ExtractGeneInfo(gene_bed: str):
    """
    Extract gene information from gene bed file.
    
    Parameters
    ----------
    gene_bed: str

    Returns
    -------
    gene_info: list
    """

    bed = pd.read_csv(gene_bed, sep="\t", header=0, index_col=[0], comment="#")
    bed['transcript'] = [x.strip().split(".")[0] for x in bed['name'].tolist()]
    bed['tss'] = bed.apply(lambda x: x['txStart'] if x['strand']=='+' else x['txEnd'], axis=1)

    ### adjacent P+GB
    bed["start"] = bed.apply(lambda x: x['txStart']-2000 if x['strand']=='+' else x['txStart'], axis=1)
    bed["end"] = bed.apply(lambda x: x['txEnd']+2000 if x['strand']=='-' else x['txEnd'], axis=1)

    bed['promoter'] = bed.apply(lambda x: tuple([x['tss']-2000, x['tss']+2000]), axis=1)
    bed['exons'] = bed.apply(lambda x: tuple([(int(i), int(j)) for i, j in zip(x['exonStarts'].strip(',').split(','), x['exonEnds'].strip(',').split(','))]), axis=1)

    ### exon length
    bed['length'] = bed.apply(lambda x: sum(list(map(lambda i: (i[1]-i[0])/1000.0, x['exons']))), axis=1)
    bed['uid'] = bed.apply(lambda x: "%s@%s@%s"%(x['name2'], x['start'], x['end']), axis=1)
    bed = bed.drop_duplicates(subset='uid', keep="first")
    gene_info = []
    for irow, x in bed.iterrows():
        gene_info.append([x['chrom'], x['start'], x['end'], x['tss'], x['promoter'], x['exons'], x['length'], 1, x['uid']])
    ### [chrom_0, start_1, end_2, tss_3, promoter_4, exons_5, length_6, 1_7, uid_8]

    return(gene_info)

def RP_AddExonRemovePromoter(peaks_info: list,
                             genes_info_full: list, 
                             genes_info_tss: list,
                             decay: int):
    """
    Multiple processing function to calculate regulation potential.
    
    Parameters
    ----------
    peaks_info: list
    genes_info_full: list
    genes_info_tss: list
    decay: int

    Returns
    -------
    genes_peaks_score_array: scipy.sparse.dok_matrix
    peaks_info_inbody: list
    peaks_info_outbody: list
    """

    Sg = lambda x: 2**(-x)
    checkInclude = lambda x, y: all([x>=y[0], x<=y[1]])
    gene_distance = 15 * decay
    genes_peaks_score_array = dok_matrix((len(genes_info_full), len(peaks_info)), dtype=np.float64)
    peaks_info_inbody = []
    peaks_info_outbody = []

    w = genes_info_full + peaks_info
    A = {}

    w.sort()
#     print(w[:100])
    for elem in w:
        if elem[-3] == 1:
            A[elem[-1]] = elem
        else:
            dlist = []
            for gene_name in list(A.keys()):
                g = A[gene_name]
                ### NOTE: main change here
                ### if peak center in the gene area
                if all([g[0]==elem[0], elem[1]>=g[1], elem[1]<=g[2]]):
                    ### if peak center in the exons
                    if any(list(map(checkInclude, [elem[1]]*len(g[5]), list(g[5])))):
                        genes_peaks_score_array[gene_name, elem[-1]] = 1.0 / g[-4]
                        peaks_info_inbody.append(elem)
                    ### if peak cencer in the promoter
                    elif checkInclude(elem[1], g[4]):
                        tmp_distance = abs(elem[1]-g[3])
                        genes_peaks_score_array[gene_name, elem[-1]] = Sg(tmp_distance / decay)
                        peaks_info_inbody.append(elem)
                    ### intron regions
                    else:
                        continue
                else:
                    dlist.append(gene_name)
            for gene_name in dlist:
                del A[gene_name]

    ### remove genes in promoters and exons
    peaks_info_set = [tuple(i) for i in peaks_info]
    peaks_info_inbody_set = [tuple(i) for i in peaks_info_inbody]
    peaks_info_outbody_set = list(set(peaks_info_set)-set(peaks_info_inbody_set))
    peaks_info_outbody = [list(i) for i in peaks_info_outbody_set]

    print("peaks number: ", len(peaks_info_set))
    print("peaks number in gene promoters and exons: ", len(set(peaks_info_inbody_set)))
    print("peaks number out gene promoters and exons:", len(peaks_info_outbody_set))

    w = genes_info_tss + peaks_info_outbody
    A = {}

    w.sort()
    for elem in w:
        if elem[-3] == 1:
            A[elem[-1]] = elem
        else:
            dlist = []
            for gene_name in list(A.keys()):
                g = A[gene_name]
                tmp_distance = elem[1] - g[1]
                if all([g[0]==elem[0], tmp_distance <= gene_distance]):
                    genes_peaks_score_array[gene_name, elem[-1]] = Sg(tmp_distance / decay)
                else:
                    dlist.append(gene_name)
            for gene_name in dlist:
                del A[gene_name]

    w.reverse()
    for elem in w:
        if elem[-3] == 1:
            A[elem[-1]] = elem
        else:
            dlist = []
            for gene_name in list(A.keys()):
                g = A[gene_name]
                tmp_distance = g[1] - elem[1]
                if all([g[0]==elem[0], tmp_distance <= gene_distance]):
                    genes_peaks_score_array[gene_name, elem[-1]] = Sg(tmp_distance / decay)
                else:
                    dlist.append(gene_name)
            for gene_name in dlist:
                del A[gene_name]

    return(genes_peaks_score_array)

def cal_gene_activity(adata: anndata.AnnData,
                      genebed: str,
                      decay: int):
    '''
    Calculate gene activity using scATAC-seq data.

    Parameters
    ----------
    adata: anndata.AnnData
        Annotated data matrix with peaks-by-cells.
    genebed: str
        Gene annotation file.
    decay: int
        Decay parameter.

    Returns
    -------
    adata_new: anndata.AnnData
        Annotated data matrix with genes-by-cells.
    '''

    # Extract features
    peaks_list = np.array(adata.var.index, dtype='|S')  # Using the default maximum length
    peaks_list = [re.sub("\W", "_", feature.decode()) for feature in peaks_list]
    peaks_list = [feature.encode() for feature in peaks_list]
    cells_list = adata.obs.index.tolist()
    cell_peaks = adata.X.T

    genes_info = []
    genes_list = []
    peaks_info = []

    print('......Extract gene information from the provided genebed file')
    # Extract gene information from the provided genebed file
    genes_info = ExtractGeneInfo(genebed)
    genes_info_tss = list()
    genes_info_full = list()  # Store complete gene information [chrom, tss, start, end, 1, unique_id]
    
    # Process and store gene information in different lists
    for igene in range(len(genes_info)):
        tmp_gene = genes_info[igene]
        genes_list.append(tmp_gene[-1])
        genes_info_full.append(tmp_gene + [igene])
        genes_info_tss.append([tmp_gene[0], tmp_gene[3], tmp_gene[1], tmp_gene[2]] + tmp_gene[4:] + [igene])
        ### Add index at the end of gene symbol
    genes = list(set([i.split("@")[0] for i in genes_list]))

    model = 'Enhanced'
    peaks_info = []

    # Process peak information
    for ipeak, peak in enumerate(peaks_list):
        peaks_tmp = peak.decode().rsplit("_", maxsplit=2)
        peaks_info.append([peaks_tmp[0], (int(peaks_tmp[1]) + int(peaks_tmp[2])) / 2.0, int(peaks_tmp[1]), int(peaks_tmp[2]), 0, peak, ipeak])
        # peaks_info [chrom, center, start, end, 0, uid, ipeak]
    
    print('......Calculate gene-peak regulatory scores based on the selected model')
    # Calculate gene-peak regulatory scores based on the selected model
    if model == "Enhanced":
        genes_peaks_score_dok = RP_AddExonRemovePromoter(peaks_info, genes_info_full, genes_info_tss, decay)

    # Convert the score matrix to CSR format, compute gene-cell score matrix
    genes_peaks_score_csr = genes_peaks_score_dok.tocsr()
    genes_cells_score_csr = genes_peaks_score_csr.dot(cell_peaks.tocsr())

    score_cells_dict = {}
    score_cells_sum_dict = {}

    print('......Store the index and total scores for each gene')
    # Store the index and total scores for each gene
    for igene, gene in enumerate(genes_list):
        score_cells_dict[gene] = igene
        score_cells_sum_dict[gene] = genes_cells_score_csr[igene, :].sum()

    score_cells_dict_dedup = {}
    score_cells_dict_max = {}

    print('......Store the maximum score for each gene symbol')
    # Store the maximum score for each gene symbol
    for gene in genes:
        score_cells_dict_max[gene] = float("-inf")

    # Choose representative symbols with the maximum score for each gene
    for gene in genes_list:
        symbol = gene.split("@")[0]
        if score_cells_sum_dict[gene] > score_cells_dict_max[symbol]:
            score_cells_dict_dedup[symbol] = score_cells_dict[gene]
            score_cells_dict_max[symbol] = score_cells_sum_dict[gene]

    # Select matrix rows based on chosen gene symbols
    gene_symbol = sorted(score_cells_dict_dedup.keys())
    matrix_row = []
    for gene in gene_symbol:
        matrix_row.append(score_cells_dict_dedup[gene])

    # Extract gene-cell score matrix based on selected rows
    score_cells_matrix = genes_cells_score_csr[matrix_row, :]

    # Create a new AnnData object to store the score matrix and set indices
    adata_new = anndata.AnnData(score_cells_matrix).T
    adata_new.obs = pd.DataFrame(index=cells_list)
    adata_new.var = pd.DataFrame(index=gene_symbol)

    return adata_new
