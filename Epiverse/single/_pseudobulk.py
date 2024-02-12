from tqdm import tqdm
import gc
import re
import os 
import gzip
from typing import Optional, Union, Dict, List
import anndata as ad
import numpy as np
import pandas as pd
import pyBigWig
import pyranges as pr

def pseudobulk(adata,chromsizes,size=None,cluster_key='celltype',clusters=None,
                  chr=['chrom','chromStart','chromEnd'],
               bigwig_path='temp',verbose=True):
    import pyrle
    if pyrle.__version__>'0.0.38':
        print('You need to use `pip install pyrle==0.0.38`')
        return

    adata.obs[cluster_key]= adata.obs[cluster_key].astype('category')
    
    if clusters==None:
        clusters=adata.obs[cluster_key].cat.categories
    print(clusters)
    for celltype in clusters:
        adata_test=adata[adata.obs[cluster_key]==celltype]
        if size!=None:
            import random 
            cell_idx=random.sample(adata_test.obs.index.tolist(),size)
            adata_test=adata_test[cell_idx]
            print(celltype,f'random select {size} cells')
            gc.collect()
        df_test=pd.DataFrame(columns=['Chromosome', 'Start', 'End', 'Name', 'Score'])
        if verbose:
            print(celltype,'chr_value')
        #df_test['Chromosome']=[i for i in adata_test.var[chr[0]] for _ in range(adata_test.shape[0])]
        df_test['Chromosome']=np.repeat(adata_test.var[chr[0]], adata_test.shape[0])
        gc.collect()
        if verbose:
            print(celltype,'chr_start')
        #df_test['Start']=[i for i in adata_test.var[chr[1]] for _ in range(adata_test.shape[0])]
        df_test['Start']=np.repeat(adata_test.var[chr[1]], adata_test.shape[0])
        gc.collect()
        if verbose:
            print(celltype,'chr_end')
        #df_test['End']=[i for i in adata_test.var[chr[2]] for _ in range(adata_test.shape[0])]
        df_test['End']=np.repeat(adata_test.var[chr[2]], adata_test.shape[0])
        gc.collect()
        
        if verbose:
            print(celltype,'Name')
        indices = np.array(adata_test.obs.index)
        repeated_indices = np.tile(indices, adata_test.shape[1])
        #df_test['Name']=adata_test.obs.index.tolist()*adata_test.shape[1]
        df_test['Name']=repeated_indices
        gc.collect()
        
        if verbose:
            print(celltype,'Score')
        df_test['Score']=adata_test.to_df().T.values.reshape(-1)
        gc.collect()
        df_test.index=np.arange(df_test.shape[0])
        
        if verbose:
            print(celltype,'write')
        group_pr=pr.PyRanges(df_test)
        group_pr.to_bigwig(
                path=f'{bigwig_path}/{celltype}.bw',
                chromosome_sizes=chromsizes,
                rpm=True,
                value_col="Score"
        )
        #return df_test
        del group_pr
        del df_test
        del adata_test
        gc.collect()

def pseudobulk_with_fragments(
    input_data: Union[pd.DataFrame, ad.AnnData],
    chromsizes: Union[pd.DataFrame, pr.PyRanges],
    cluster_key: str,
    clusters: Optional[list] = None,
    bed_path: Optional[str] = 'temp',
    bigwig_path: Optional[str] = 'temp',
    verbose=True,
    path_to_fragments: Optional[Dict[str, str]] = None,
    normalize_bigwig: Optional[bool] = True,
    remove_duplicates: Optional[bool] = True,
    split_pattern: Optional[str] = "___",
    use_polars: Optional[bool] = True,
    sample_id_col: Optional[str] = None,
    **kwargs
    ):

    """
    Create pseudobulks as bed and bigwig from single cell fragments file given a barcode annotation.

    Parameters
    ---------
    input_data: CistopicObject or pd.DataFrame
            A :class:`CistopicObject` containing the specified `variable` as a column in :class:`CistopicObject.cell_data` or a cell metadata
            :class:`pd.DataFrame` containing barcode as rows, containing the specified `variable` as a column (additional columns are
            possible) and a `sample_id` column. Index names must contain the BARCODE (e.g. ATGTCGTC-1), additional tags are possible separating with -
            (e.g. ATGCTGTGCG-1-Sample_1). The levels in the sample_id column must agree with the keys in the path_to_fragments dictionary.
            Alternatively, if the cell metadata contains a column named barcode it will be used instead of the index names.
    chromsizes: pd.DataFrame or pr.PyRanges
            A data frame or :class:`pr.PyRanges` containing size of each chromosome, containing 'Chromosome', 'Start' and 'End' columns.
    cluster_key: str
            A character string indicating the column that will be used to create the different group pseudobulk. It must be included in
            the cell metadata provided as input_data.
    cluster: list, optional
            A list of character strings indicating the groups for which pseudobulks will be created. If None, pseudobulks will be created for all groups.
    bed_path: str
            Path to folder where the fragments bed files per group will be saved. If None, files will not be generated.
    bigwig_path: str
            Path to folder where the bigwig files per group will be saved. If None, files will not be generated.
    path_to_fragments: str or dict, optional
            A dictionary of character strings, with sample name as names indicating the path to the fragments file/s from which pseudobulk profiles have to
            be created. If a :class:`CistopicObject` is provided as input it will be ignored, but if a cell metadata :class:`pd.DataFrame` is provided it
            is necessary to provide it. The keys of the dictionary need to match with the sample_id tag added to the index names of the input data frame.
    normalize_bigwig: bool, optional
            Whether bigwig files should be CPM normalized. Default: True.
    remove_duplicates: bool, optional
            Whether duplicates should be removed before converting the data to bigwig.
    split_pattern: str, optional
            Pattern to split cell barcode from sample id. Default: '___'. Note, if `split_pattern` is not None, then `export_pseudobulk` will
            attempt to infer `sample_id` from the index of `input_data` and ignore `sample_id_col`.
    use_polars: bool, optional
            Whether to use polars to read fragments files. Default: True.
        sample_id_col: str, optional
            This parameter is not necessary.
            Name of the column containing the sample name per barcode in the input :class:`CistopicObject.cell_data` or class:`pd.DataFrame`. Default: None.
    **kwargs
            Additional parameters for ray.init()

    Return
    ------
    dict
            A dictionary containing the paths to the newly created bed fragments files per group a dictionary containing the paths to the
            newly created bigwig files per group.
    """
    # check the imported package
    try:
        import pyrle
    except ImportError:
        raise  ImportError(
            'Please install the pyrle: `pip install pyrle`.'
        )
    
    
    # Get fragments file
    if path_to_fragments is None:
        print("Please, provide path_to_fragments.")
  
    # Get celltype 
    if isinstance(input_data, ad.AnnData):
        input_data = input_data.obs
    if clusters is None:
        cell_data = input_data 
    elif clusters is not None:
        cell_data = input_data[input_data[cluster_key].isin(clusters)]
    
    # Process the sample_id_col(optional)
    if sample_id_col is not None:
        try:
            sample_ids = list(set(cell_data[sample_id_col]))
        except ValueError:
            print(
            'Please, include a sample identification column (e.g. "sample_id") in your cell metadata!'
        )
            
    # Get fragments
    fragments_df_dict = {}
    for sample_id in path_to_fragments.keys():
        if sample_id_col is not None:
            if sample_id not in sample_ids:
                print(
                    "The following path_to_fragments entry is not found in the cell metadata sample_id_col: ",
                    sample_id,
                    ". It will be ignored.",
                )
            if verbose: 
                print("Reading fragments from " + path_to_fragments[sample_id])
            fragments_df = read_fragments_from_file(path_to_fragments[sample_id], use_polars=use_polars).df
            # Convert to int32 for memory efficiency
            fragments_df.Start = np.int32(fragments_df.Start)
            fragments_df.End = np.int32(fragments_df.End)
            if "Score" in fragments_df:
                fragments_df.Score = np.int32(fragments_df.Score)
            if "barcode" in cell_data:
                fragments_df = fragments_df.loc[
                    fragments_df["Name"].isin(cell_data["barcode"].tolist())
                ]
            else:
                fragments_df = fragments_df.loc[
                    fragments_df["Name"].isin(
                        prepare_tag_cells(cell_data.index.tolist(), split_pattern)
                    )
                ]
            fragments_df_dict[sample_id] = fragments_df
        else:
            if verbose: 
                print("Reading fragments from " + path_to_fragments[sample_id])
            fragments_df = read_fragments_from_file(path_to_fragments[sample_id], use_polars=use_polars).df
            # Convert to int32 for memory efficiency
            fragments_df.Start = np.int32(fragments_df.Start)
            fragments_df.End = np.int32(fragments_df.End)
            if "Score" in fragments_df:
                fragments_df.Score = np.int32(fragments_df.Score)
            if "barcode" in cell_data:
                fragments_df = fragments_df.loc[
                    fragments_df["Name"].isin(cell_data["barcode"].tolist())
                ]
            else:
                fragments_df = fragments_df.loc[
                    fragments_df["Name"].isin(
                        prepare_tag_cells(cell_data.index.tolist(), split_pattern)
                    )
                ]
            fragments_df_dict[sample_id] = fragments_df
            print(fragments_df)

    # Set groups
    if sample_id_col is not None:
        if "barcode" in cell_data:
            cell_data = cell_data.loc[:, [cluster_key,sample_id_col,"barcode"]]
        else:
            cell_data = cell_data.loc[:, [cluster_key,sample_id_col]]
    else:
        if "barcode" in cell_data:
            cell_data = cell_data.loc[:, [cluster_key, "barcode"]]
        else:
            cell_data = cell_data.loc[:, [cluster_key]]
    cell_data[cluster_key] = cell_data[cluster_key].replace(" ", "", regex=True)
    cell_data[cluster_key] = cell_data[cluster_key].replace("[^A-Za-z0-9]+", "_", regex=True)
    groups = sorted(list(set(cell_data[cluster_key])))

    # Check chromosome sizes
    if isinstance(chromsizes, pd.DataFrame):
        chromsizes = chromsizes.loc[:, ["Chromosome", "Start", "End"]]
        chromsizes = pr.PyRanges(chromsizes)
    # Check that output dir exist and generate output paths
    if isinstance(bed_path, str):
        if not os.path.exists(bed_path):
            os.makedirs(bed_path)
        bed_paths = {
            group: os.path.join(bed_path, str(group) + ".bed.gz") for group in groups
        }
    else:
        bed_paths = {}
    if isinstance(bigwig_path, str):
        if not os.path.exists(bigwig_path):
            os.makedirs(bigwig_path)
        bw_paths = {
            group: os.path.join(bigwig_path, str(group) + ".bw") for group in groups
        }
    else:
        bw_paths = {}

    # Get pseudobulk from different celltypes
    for group in groups:
        if sample_id_col is not None:
            export_pseudobulk_one_sample(
                cell_data,
                group,
                fragments_df_dict,
                chromsizes,
                bigwig_path,
                bed_path,
                normalize_bigwig,
                remove_duplicates,
                split_pattern,
                sample_id_col,
                )
        else:
            export_pseudobulk_one_sample(
                cell_data,
                group,
                fragments_df_dict,
                chromsizes,
                bigwig_path,
                bed_path,
                normalize_bigwig,
                remove_duplicates,
                split_pattern,
                 )
        

r"""
Copy from pycisTopic: https://github.com/aertslab/pycisTopic/
"""

def export_pseudobulk_one_sample(
    cell_data: pd.DataFrame,
    group: str,
    fragments_df_dict: Dict[str, pd.DataFrame],
    chromsizes: pr.PyRanges,
    bigwig_path: str,
    bed_path: str,
    normalize_bigwig: Optional[bool] = True,
    remove_duplicates: Optional[bool] = True,
    split_pattern: Optional[str] = "___",
    sample_id_col: Optional[str] = None,
):
    """
    Create pseudobulk as bed and bigwig from single cell fragments file given a barcode annotation and a group.

    Parameters
    ---------
    cell_data: pd.DataFrame
            A cell metadata :class:`pd.Dataframe` containing barcodes, their annotation and their sample of origin.
    group: str
            A character string indicating the group for which pseudobulks will be created.
    fragments_df_dict: dict
            A dictionary containing data frames as values with 'Chromosome', 'Start', 'End', 'Name', and 'Score' as columns; and sample label
            as keys. 'Score' indicates the number of times that a fragments is found assigned to that barcode.
    chromsizes: pr.PyRanges
            A :class:`pr.PyRanges` containing size of each column, containing 'Chromosome', 'Start' and 'End' columns.
    bigwig_path: str
            Path to folder where the bigwig file will be saved.
    bed_path: str
            Path to folder where the fragments bed file will be saved.
    normalize_bigwig: bool, optional
            Whether bigwig files should be CPM normalized. Default: True.
    remove_duplicates: bool, optional
            Whether duplicates should be removed before converting the data to bigwig.
    split_pattern: str
            Pattern to split cell barcode from sample id. Default: ___ .
    sample_id_col: str, optional
            Name of the column containing the sample name per barcode in the input :class:`CistopicObject.cell_data` or class:`pd.DataFrame`. Default: 'None'.
    """
    # Create logger
    print("Creating pseudobulk for " + str(group))
    group_fragments_list = []
    group_fragments_dict = {}

    if sample_id_col is not None:
        for sample_id in fragments_df_dict:
            sample_data = cell_data[cell_data.loc[:, sample_id_col].isin([sample_id])]
            if "barcode" in sample_data:
                sample_data.index = sample_data["barcode"].tolist()
            else:
                sample_data.index = prepare_tag_cells(
                    sample_data.index.tolist(), split_pattern
               )
            group_var = sample_data.iloc[:, 0]
            barcodes = group_var[group_var.isin([group])].index.tolist()
            fragments_df = fragments_df_dict[sample_id]
            group_fragments = fragments_df.loc[fragments_df["Name"].isin(barcodes)]
            if len(fragments_df_dict) > 1:
                group_fragments_dict[sample_id] = group_fragments

    else:
        for sample_id in fragments_df_dict:
            sample_data = cell_data
            if "barcode" in sample_data:
                sample_data.index = sample_data["barcode"].tolist()
            else:
                sample_data.index = prepare_tag_cells(
                    sample_data.index.tolist(), split_pattern
                )
            group_var = sample_data.iloc[:, 0]
            barcodes = group_var[group_var.isin([group])].index.tolist()
            fragments_df = fragments_df_dict[sample_id]
            group_fragments = fragments_df.loc[fragments_df["Name"].isin(barcodes)]
            if len(fragments_df_dict) > 1:
                group_fragments_dict[sample_id] = group_fragments

    if len(fragments_df_dict) > 1:
        group_fragments_list = [
            group_fragments_dict[list(group_fragments_dict.keys())[x]]
            for x in range(len(fragments_df_dict))
        ]
        group_fragments = group_fragments_list[0].append(group_fragments_list[1:])

    del group_fragments_dict
    del group_fragments_list
    del fragments_df
    gc.collect()

    group_pr = pr.PyRanges(group_fragments)
    if isinstance(bigwig_path, str):
        bigwig_path_group = os.path.join(bigwig_path, str(group) + ".bw")
        if remove_duplicates:
            group_pr.to_bigwig(
                path=bigwig_path_group,
                chromosome_sizes=chromsizes,
                rpm=normalize_bigwig,
            )
        else:
            group_pr.to_bigwig(
                path=bigwig_path_group,
                chromosome_sizes=chromsizes,
                rpm=normalize_bigwig,
                value_col="Score",
            )
    if isinstance(bed_path, str):
        bed_path_group = os.path.join(bed_path, str(group) + ".bed.gz")
        group_pr.to_bed(
            path=bed_path_group, keep=False, compression="infer", chain=False
        )
    print(str(group) + " done!")


def prepare_tag_cells(cell_names, split_pattern="___"):
    if split_pattern == "-":
        new_cell_names = [
            re.findall(r"^[ACGT]*-[0-9]+-", x)[0].rstrip("-")
            if len(re.findall(r"^[ACGT]*-[0-9]+-", x)) != 0
            else x
            for x in cell_names
        ]
        new_cell_names = [
            re.findall(r"^\w*-[0-9]*", new_cell_names[i])[0].rstrip("-")
            if (len(re.findall(r"^\w*-[0-9]*", new_cell_names[i])) != 0)
            & (new_cell_names[i] == cell_names[i])
            else new_cell_names[i]
            for i in range(len(new_cell_names))
        ]
    else:
        new_cell_names = [x.split(split_pattern)[0] for x in cell_names]

    return new_cell_names

def read_fragments_from_file(
    fragments_bed_filename, use_polars: bool = True
) -> pr.PyRanges:
    """
    Read fragments BED file to PyRanges object.

    Parameters
    ----------
    fragments_bed_filename: Fragments BED filename.
    use_polars: Use polars instead of pandas for reading the fragments BED file.

    Returns
    -------
    PyRanges object of fragments.
    """

    bed_column_names = (
        "Chromosome",
        "Start",
        "End",
        "Name",
        "Score",
        "Strand",
        "ThickStart",
        "ThickEnd",
        "ItemRGB",
        "BlockCount",
        "BlockSizes",
        "BlockStarts",
    )

    # Set the correct open function depending if the fragments BED file is gzip compressed or not.
    open_fn = gzip.open if fragments_bed_filename.endswith(".gz") else open

    skip_rows = 0
    nbr_columns = 0
    with open_fn(fragments_bed_filename, "rt") as fragments_bed_fh:
        for line in fragments_bed_fh:
            # Remove newlines and spaces.
            line = line.strip()

            if not line or line.startswith("#"):
                # Count number of empty lines and lines which start with a comment before the actual data.
                skip_rows += 1
            else:
                # Get number of columns from the first real BED entry.
                nbr_columns = len(line.split("\t"))

                # Stop reading the BED file.
                break

    if nbr_columns < 4:
        raise ValueError(
            f'Fragments BED file needs to have at least 4 columns. "{fragments_bed_filename}" contains only '
            f"{nbr_columns} columns."
        )

    if use_polars:
        import polars as pl

        # Read fragments BED file with polars.
        df = (
            pl.read_csv(
                fragments_bed_filename,
                has_header=False,
                skip_rows=skip_rows,
                separator="\t",
                use_pyarrow=True,
                new_columns=bed_column_names[:nbr_columns],
            )
            .with_columns(
                [
                    pl.col("Chromosome").cast(pl.Utf8),
                    pl.col("Start").cast(pl.Int32),
                    pl.col("End").cast(pl.Int32),
                    pl.col("Name").cast(pl.Utf8),
                ]
            )
            .to_pandas()
        )

        # Convert "Name" column to pd.Categorical as groupby operations will be done on it later.
        df["Name"] = df["Name"].astype("category")
    else:
        # Read fragments BED file with pandas.
        df = pd.read_table(
            fragments_bed_filename,
            sep="\t",
            skiprows=skip_rows,
            header=None,
            names=bed_column_names[:nbr_columns],
            doublequote=False,
            engine="c",
            dtype={
                "Chromosome": str,
                "Start'": np.int32,
                "End": np.int32,
                "Name": "category",
                "Strand": str,
            },
        )

    # Convert pandas dataframe to PyRanges dataframe.
    # This will convert "Chromosome" and "Strand" columns to pd.Categorical.
    return pr.PyRanges(df)
