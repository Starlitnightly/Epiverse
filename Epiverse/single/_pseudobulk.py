from tqdm import tqdm
import gc
import pyranges as pr
import numpy as np
import pandas as pd

def pseudobulk(adata,chromsizes,cluster_key='celltype',clusters=None,
                  chr=['chrom','chromStart','chromEnd'],
               bigwig_path='temp',verbose=True):
    adata.obs[cluster_key]= adata.obs[cluster_key].astype('category')
    
    if clusters==None:
        clusters=adata.obs[cluster_key].cat.categories
    print(clusters)
    for celltype in clusters:
        adata_test=adata[adata.obs[cluster_key]==celltype]
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
        gc.collect()