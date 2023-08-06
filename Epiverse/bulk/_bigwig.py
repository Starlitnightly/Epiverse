import pyBigWig
import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
import omicverse as ov
import pandas as pd
from tqdm import tqdm
import re
import os
from scipy.sparse import csr_matrix
import anndata

sc_color=['#7CBB5F','#368650','#A499CC','#5E4D9A','#78C2ED','#866017', '#9F987F','#E0DFED',
 '#EF7B77', '#279AD7','#F0EEF0', '#1F577B', '#A56BA7', '#E0A7C8', '#E069A6', '#941456', '#FCBC10',
 '#EAEFC5', '#01A0A7', '#75C8CC', '#F0D7BC', '#D5B26C', '#D5DA48', '#B6B812', '#9DC3C3', '#A89C92', '#FEE00C', '#FEF2A1']

class bigwig(object):
    
    def __init__(self,bw_path_dict:str):
        """
        Initialize the bigwig object.

        Arguments:
            bw_path_dict: the dictionary of bigwig file path such as: {'bw_name':'bw_path'}
        
        """
        self.bw_path_dict=bw_path_dict
        self.bw_names=list(bw_path_dict.keys())
        self.bw_tss_scores_dict={}
        self.bw_tes_scores_dict={}
        self.bw_body_scores_dict={}
        self.bw_lens=len(self.bw_names)
        self.gtf=None
    
    def read(self):
        """
        Read bigwig file from bw_path_dict.
        
        """
        self.bw_dict={}
        for bw_name in self.bw_names:
            print('......Loading {}'.format(bw_name))
            self.bw_dict[bw_name]=pyBigWig.open(self.bw_path_dict[bw_name])
            
    
    def load_gtf(self,gtf_path:str):
        """
        Load gtf file.

        Arguments:
            gtf_path: the path of gtf file.
        """


        print('......Loading gtf file')
        features=ov.utils.read_gtf(gtf_path)
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
        self.gtf=features

    def save_bw_result(self,save_path:str):
        """
        Save the computed matrix results of TSS/TES/Body.

        Arguments:
            save_path: the path of saving results.
        
        """
        for bw_name in self.bw_tss_scores_dict.keys():
            print(f"Saving {bw_name} results...")
            if not os.path.exists(save_path+'/'+bw_name):
                os.mkdir(save_path+'/'+bw_name)
                print(f"Folder '{save_path+'/'+bw_name}' created.")
            else:
                pass
            
            if self.bw_tss_scores_dict[bw_name] is None:
                print("You need to run the compute_matrix function first!")
            else:
                self.bw_tss_scores_dict[bw_name].write_h5ad(save_path+'/'+bw_name+'/tss_scores.h5ad')
            
            if self.bw_tes_scores_dict[bw_name] is None:
                print("You need to run the compute_matrix function first!")
            else:
                self.bw_tes_scores_dict[bw_name].write_h5ad(save_path+'/'+bw_name+'/tes_scores.h5ad')

            if self.bw_body_scores_dict[bw_name] is None:
                print("You need to run the compute_matrix function first!")
            else:
                self.bw_body_scores_dict[bw_name].write_h5ad(save_path+'/'+bw_name+'/body_scores.h5ad')
    
    def load_bw_result(self,load_path:str):

        """
        Load the computed matrix results of TSS/TES/Body.

        Arguments:
            load_path: the path of loading results.

        """
        for bw_name in self.bw_names:
            print(f"Loading {bw_name} results...")
            if not os.path.exists(load_path+'/'+bw_name+'/tss_scores.h5ad'):
                print("You need to run the compute_matrix function first for {}!".format(bw_name))
            else:
                self.bw_tss_scores_dict[bw_name]=anndata.read_h5ad(load_path+'/'+bw_name+'/tss_scores.h5ad')
            
            if not os.path.exists(load_path+'/'+bw_name+'/tes_scores.h5ad'):
                print("You need to run the compute_matrix function first for {}!".format(bw_name))
            else:
                self.bw_tes_scores_dict[bw_name]=anndata.read_h5ad(load_path+'/'+bw_name+'/tes_scores.h5ad')

            if not os.path.exists(load_path+'/'+bw_name+'/body_scores.h5ad'):
                print("You need to run the compute_matrix function first for {}!".format(bw_name))
            else:
                self.bw_body_scores_dict[bw_name]=anndata.read_h5ad(load_path+'/'+bw_name+'/body_scores.h5ad')


    def compute_matrix(self,bw_name:str,nbins:int=100,
                          upstream:int=3000,downstream:int=3000):
        """
        Compute the enrichment matrix of TSS/TES/Body.

        Arguments:
            bw_name: the name of bigwig file need to be computed.
            nbins: the number of bins.
            upstream: the upstream of TSS/TES.
            downstream: the downstream of TSS/TES.

        """
        if bw_name not in self.bw_tss_scores_dict.keys():
            print('......Computing {} matrix'.format(bw_name))
            #nbins=100
            features=self.gtf.loc[(self.gtf['feature']=='transcript')&(self.gtf['seqname'].str.contains('chr'))]
            gene_list=features['gene_id'].unique()

            tss_array=pd.DataFrame(columns=[i for i in range(nbins)])
            tes_array=pd.DataFrame(columns=[i for i in range(nbins)])
            body_array=pd.DataFrame(columns=[i for i in range(nbins)])

            for g in tqdm(gene_list, desc='Processing genes', unit='gene'):
                test_f=features.loc[(features['gene_id']==g)&(features['feature']=='transcript')].iloc[0]
                chrom=test_f.seqname
                if test_f.strand=='-':
                    tss_loc=test_f.end
                    tes_loc=test_f.start
                else:
                    tss_loc=test_f.start
                    tes_loc=test_f.end
                
                tss_region_start=tss_loc-upstream
                tss_region_end=tss_loc+downstream

                tes_region_start=tes_loc-upstream
                tes_region_end=tes_loc+downstream

                body_region_start=test_f.start
                body_region_end=test_f.end
                
                if tss_region_start<0:
                    tss_region_start=0
                if tss_region_end>self.bw_dict[bw_name].chroms()[chrom]:
                    tss_region_end=self.bw_dict[bw_name].chroms()[chrom]

                if tes_region_start<0:
                    tes_region_start=0
                if tes_region_end>self.bw_dict[bw_name].chroms()[chrom]:
                    tes_region_end=self.bw_dict[bw_name].chroms()[chrom]
                
                if test_f.strand=='-':
                    tss_array.loc[g]=np.array(self.bw_dict[bw_name].stats(chrom, 
                                        tss_region_start,
                                        tss_region_end, nBins=nbins,
                                        type='mean')).astype(float)[::-1]
                    tes_array.loc[g]=np.array(self.bw_dict[bw_name].stats(chrom,
                                                                          tes_region_start,
                                                                            tes_region_end, nBins=nbins,
                                                                            type='mean')).astype(float)[::-1]
                    body_array.loc[g]=np.array(self.bw_dict[bw_name].stats(chrom,
                                                                            body_region_start,
                                                                                body_region_end, nBins=nbins,
                                                                                type='mean')).astype(float)[::-1]
                                                                          
                                                                    
                else:
                    tss_array.loc[g]=np.array(self.bw_dict[bw_name].stats(chrom, 
                                        tss_region_start,
                                        tss_region_end, nBins=nbins,
                                        type='mean')).astype(float)
                    tes_array.loc[g]=np.array(self.bw_dict[bw_name].stats(chrom,
                                                                            tes_region_start,
                                                                                tes_region_end, nBins=nbins,
                                                                                type='mean')).astype(float)
                    body_array.loc[g]=np.array(self.bw_dict[bw_name].stats(chrom,
                                                                            body_region_start,
                                                                                body_region_end, nBins=nbins,
                                                                                type='mean')).astype(float)
            tss_csr=csr_matrix(tss_array.fillna(0).loc[tss_array.fillna(0).mean(axis=1).sort_values(ascending=False).index].values)
            tes_csr=csr_matrix(tes_array.fillna(0).loc[tes_array.fillna(0).mean(axis=1).sort_values(ascending=False).index].values)
            body_csr=csr_matrix(body_array.fillna(0).loc[body_array.fillna(0).mean(axis=1).sort_values(ascending=False).index].values)

            tss_adata=anndata.AnnData(tss_csr)
            tss_adata.obs.index=tss_array.fillna(0).mean(axis=1).sort_values(ascending=False).index
            tss_adata.uns['range']=[0-downstream,upstream]
            tss_adata.uns['bins']=nbins

            tes_adata=anndata.AnnData(tes_csr)
            tes_adata.obs.index=tes_array.fillna(0).mean(axis=1).sort_values(ascending=False).index
            tes_adata.uns['range']=[0-downstream,upstream]
            tes_adata.uns['bins']=nbins

            body_adata=anndata.AnnData(body_csr)
            body_adata.obs.index=body_array.fillna(0).mean(axis=1).sort_values(ascending=False).index
            body_adata.uns['range']=[0,upstream+downstream]
            body_adata.uns['bins']=nbins

            self.bw_tss_scores_dict[bw_name]=tss_adata
            self.bw_tes_scores_dict[bw_name]=tes_adata
            self.bw_body_scores_dict[bw_name]=body_adata
            print('......{} matrix finished'.format(bw_name))
            print('......{} tss matrix can be found in bw_tss_scores_dict[{}]'.format(bw_name,bw_name))
            print('......{} tes matrix can be found in bw_tes_scores_dict[{}]'.format(bw_name,bw_name))
            print('......{} body matrix can be found in bw_body_scores_dict[{}]'.format(bw_name,bw_name))
        else:
            pass

    def plot_matrix(self,bw_name:str,bw_type:str='TSS',
                    figsize:tuple=(2,8),cmap:str='Greens',
                    vmax='auto',vmin='auto',fontsize:int=12,title:str='')->tuple:
        """
        Plot the enrichment matrix of TSS/TES/Body.

        Arguments:
            bw_name: the name of bigwig file need to be computed.
            bw_type: can be set as 'TSS','TES','body' or 'all'.
            figsize: the size of figure.
            cmap: the color map of figure.
            vmax: the max value of color bar. Default the 98% percentile of data.
            vmin: the min value of color bar. Default 0.
            fontsize: the fontsize of figure.
            title: the title of figure.
        
        Returns:
            fig: the figure object.
            ax: the axis object.

        
        """
        #from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        fig, ax = plt.subplots(figsize=figsize)
        #pm=np.clip(tss_array.loc[order], 0, 11)
        if bw_type=='TSS':
            adata=self.bw_tss_scores_dict[bw_name]
        elif bw_type=='TES':
            adata=self.bw_tes_scores_dict[bw_name]
        elif bw_type=='body':
            adata=self.bw_body_scores_dict[bw_name]
        elif bw_type=='all':
            adata=anndata.concat([self.bw_tss_scores_dict[bw_name][:,:50],
                self.bw_body_scores_dict[bw_name][:,:],
               self.bw_tes_scores_dict[bw_name][:,50:]],axis=1)
            adata.uns=self.bw_tss_scores_dict[bw_name].uns.copy()
            adata.uns['bins']=adata.shape[1]

        if vmax=='auto':
            vmax=np.percentile(adata.X.toarray() ,98)
        if vmin=='auto':
            vmin=0

        smp=ax.imshow(adata.X.toarray(),
                aspect='auto',
                interpolation='bilinear',
                origin='upper',
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,)
        cax=plt.colorbar(smp,shrink=0.5)
        cax.ax.tick_params(labelsize=fontsize)
        ax.yaxis.set_visible(False)
        #ax.xaxis.set_visible(False)
        #ax.set_xlabel('TSS')
        # 设置TSS和TES位置的刻度和标签
        if bw_type=='TSS':
            x=adata.uns['bins']
            tss_position = int(adata.shape[1] / 2)  # 1/4位置
            #tes_position = int(2 * 100 / 3)  # 后1/3位置

            ax.set_xticks([0,tss_position,adata.shape[1]])  # 设置刻度位置
            ax.set_xticklabels([format_number_with_k(adata.uns['range'][0]),
                    'TSS',format_number_with_k(adata.uns['range'][1])],fontsize=fontsize)  # 设置刻度标签
        elif bw_type=='TES':
            x=adata.uns['bins']
            tes_position = int(adata.shape[1] / 2)
            ax.set_xticks([0,tes_position,adata.shape[1]])  # 设置刻度位置
            ax.set_xticklabels([format_number_with_k(adata.uns['range'][0]),
                    'TES',format_number_with_k(adata.uns['range'][1])],fontsize=fontsize)  # 设置刻度标签
        elif bw_type=='body':
            x=adata.uns['bins']
            body_position = int(adata.shape[1] / 2)
            ax.set_xticks([0,body_position,adata.shape[1]])
            ax.set_xticklabels([format_number_with_k(adata.uns['range'][0]),
                    'Peak',format_number_with_k(adata.uns['range'][1])],fontsize=fontsize)
        elif bw_type=='all':
            x=adata.uns['bins']
            tss_position = int(adata.shape[1] / 4)
            tes_position = int(adata.shape[1] / 4 *3)
            ax.set_xticks([0,tss_position,tes_position,adata.shape[1]])
            ax.set_xticklabels([format_number_with_k(adata.uns['range'][0]),
                    'TSS','TES',format_number_with_k(adata.uns['range'][1])],fontsize=fontsize)
        plt.grid(False)
        plt.tight_layout()
        if title!='':
            plt.title(title,fontsize=fontsize)
        else:
            plt.title(bw_name,fontsize=fontsize)
        return fig,ax
    
    def plot_matrix_line(self,bw_name:str,bw_type:str='TSS',
                         figsize:tuple=(3,3),color:str='#a51616',
                         linewidth:int=2,fontsize:int=13,title:str='')->tuple:
        
        """
        Plot the enrichment hist of TSS/TES/Body.

        Arguments:  
            bw_name: the name of bigwig file need to be computed.
            bw_type: can be set as 'TSS','TES','body' or 'all'.
            figsize: the size of figure.
            color: the color of figure.
            linewidth: the linewidth of figure.
            fontsize: the fontsize of figure.
            title: the title of figure.

        Returns:
            fig: the figure object.
            ax: the axis object.

        """

        fig, ax = plt.subplots(figsize=figsize)

        if bw_type=='TSS':
            adata=self.bw_tss_scores_dict[bw_name]
        elif bw_type=='TES':
            adata=self.bw_tes_scores_dict[bw_name]
        elif bw_type=='body':
            adata=self.bw_body_scores_dict[bw_name]
        elif bw_type=='all':
            adata=anndata.concat([self.bw_tss_scores_dict[bw_name][:,:50],
                self.bw_body_scores_dict[bw_name][:,:],
               self.bw_tes_scores_dict[bw_name][:,50:]],axis=1)
            adata.uns=self.bw_tss_scores_dict[bw_name].uns.copy()
            adata.uns['bins']=adata.shape[1]

        ax.plot([i for i in range(adata.shape[1])],
                adata.X.toarray().mean(axis=0),linewidth=linewidth,color=color)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)

        if bw_type=='TSS':
            x=adata.uns['bins']
            tss_position = int(adata.shape[1] / 2)  # 1/4位置
            #tes_position = int(2 * 100 / 3)  # 后1/3位置

            ax.set_xticks([0,tss_position,adata.shape[1]])  # 设置刻度位置
            ax.set_xticklabels([format_number_with_k(adata.uns['range'][0]),
                    'TSS',format_number_with_k(adata.uns['range'][1])],fontsize=fontsize)  # 设置刻度标签
        elif bw_type=='TES':
            x=adata.uns['bins']
            tes_position = int(adata.shape[1] / 2)
            ax.set_xticks([0,tes_position,adata.shape[1]])  # 设置刻度位置
            ax.set_xticklabels([format_number_with_k(adata.uns['range'][0]),
                    'TES',format_number_with_k(adata.uns['range'][1])],fontsize=fontsize)  # 设置刻度标签
        elif bw_type=='body':
            x=adata.uns['bins']
            body_position = int(adata.shape[1] / 2)
            ax.set_xticks([0,body_position,adata.shape[1]])
            ax.set_xticklabels([format_number_with_k(adata.uns['range'][0]),
                    'Peak',format_number_with_k(adata.uns['range'][1])],fontsize=fontsize)
            
        elif bw_type=='all':
            x=adata.uns['bins']
            tss_position = int(adata.shape[1] / 4)
            tes_position = int(adata.shape[1] / 4 *3)
            ax.set_xticks([0,tss_position,tes_position,adata.shape[1]])
            ax.set_xticklabels([format_number_with_k(adata.uns['range'][0]),
                    'TSS','TES',format_number_with_k(adata.uns['range'][1])],fontsize=fontsize)
        ax.set_yticklabels(ax.get_yticklabels()[1:],fontsize=fontsize)
        
        plt.grid(False)
        plt.tight_layout()
        if title!='':
            plt.title(title,fontsize=fontsize)
        else:
            plt.title(bw_name,fontsize=fontsize)
        return fig,ax
        
        
    def plot_track(self,chrom:str,chromstart:int,chromend:int,nbins:int=700,
                   value_type:str='mean',transform:str='no',
                   figwidth:int=6,figheight:int=1,plot_names=None,
                   color_dict=None,region_dict=None,
                   gtf_color:str='#000000',prefered_name:str='gene_id')->tuple:
        
        """
        Plot the peak track of bigwig file.

        Arguments:
            chrom: the chromosome of region.
            chromstart: the start position of region.
            chromend: the end position of region.
            nbins: the number of bins.
            value_type: the type of value. Can be set as 'mean','max','min','coverage','std','sum'.
            transform: the transform of value. Can be set as 'log','log2','log10','log1p','-log' or 'no'.
            figwidth: the width of figure.
            figheight: the height of figure.
            plot_names: the name of bigwig file need to be plotted.
            color_dict: the color of bigwig file need to be plotted.
            region_dict: the region of interest.
            gtf_color: the color of gtf.
            prefered_name: the prefered name of gtf.

        Returns:
            fig: the figure object.
            ax: the axis object.
        
        """
        self.scores_per_bin_dict={}
        for bw_name in self.bw_names:
            
            score_list=np.array(self.bw_dict[bw_name].stats(chrom, 
                                                       chromstart,
                                                    chromend, nBins=nbins,
                                                     type=value_type)).astype(float)
            
            if transform in ['log', 'log2', 'log10']:
                score_list=eval('np.' + transform + '(score_list)')
            elif transform == 'log1p':
                score_list=np.log1p(score_list)
            elif transform == '-log':
                - np.log(score_list)
            else:
                pass
            self.scores_per_bin_dict[bw_name]=score_list


        if color_dict is None:
            color_dict={}
            if len(self.bw_names)<=len(sc_color):
                color_dict=dict(zip(self.bw_names,sc_color))
            elif len(self.bw_names)<102:
                 color_dict=dict(zip(self.bw_names,sc.pl.palettes.default_102))

        if self.gtf is None:
            bw_lens=self.bw_lens
        else:
            bw_lens=self.bw_lens+1

        fig, axes = plt.subplots(bw_lens,1,figsize=(figwidth,bw_lens*figheight))
        if plot_names is None:
            plot_names=self.bw_names
        
        if bw_lens==1:
            axes=[axes]
        
        if self.gtf is None:
            if bw_lens>1:
                plot_axes=axes[:-1]
            else:
                plot_axes=axes
        else:
            plot_axes=axes
       

        for ax,bw_name,plot_name in zip(plot_axes,self.bw_names,plot_names):
            x_values = np.linspace(chromstart, chromend, nbins)
            ax.plot(x_values, self.scores_per_bin_dict[bw_name], '-',color=color_dict[bw_name])
            ax.fill_between(x_values, self.scores_per_bin_dict[bw_name], linewidth=0.1,
                            color=color_dict[bw_name])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            #ax.axis('off')
            ax.grid(False)
            ax.get_xaxis().get_major_formatter().set_scientific(False)

            #ax.set_xticklabels([start_region,chromEnd],fontsize=11)
            ax.set_yticklabels([])
            ax.set_xticklabels([],fontsize=11)
            ax.set_ylabel(plot_name,fontsize=12)

            if region_dict is not None:
                for region in region_dict:
                    ax.axvspan(region_dict[region][0],region_dict[region][1],alpha=0.4,color='#c2c2c2')
        
        if self.gtf is not None:
            ax=axes[-1]
            goal_gtf1=self.gtf.loc[(self.gtf['seqname']==chrom)&(self.gtf['start']>chromstart)&(self.gtf['end']<chromend)]
            for i in range(len(goal_gtf1)):
                plot_obj=goal_gtf1.iloc[i]
                if plot_obj.feature=='transcript':
                    plt.hlines(y=1,xmin=plot_obj.start,xmax=plot_obj.end, color=gtf_color, linewidth=2)
                    gene_attr=plot_obj.attribute
                    plot_text=''
                    for g in gene_attr:
                        if prefered_name in g:
                            plot_text=g.replace(prefered_name,'').replace('\"','')
                    
                    ax.text(plot_obj.end,1,plot_text,fontsize=12)
                elif plot_obj.feature=='exon':
                    rect = plt.Rectangle((plot_obj.start,0.5),plot_obj.end-plot_obj.start,1,
                                        facecolor=gtf_color,alpha=0.5)
                    ax.add_patch(rect)
                elif plot_obj.feature=='3UTR' or plot_obj.feature=='5UTR':
                    rect = plt.Rectangle((plot_obj.start,0.75),plot_obj.end-plot_obj.start,0.5,
                                        facecolor=gtf_color,alpha=1)
                    ax.add_patch(rect)
            ax.set_ylim(0,2)
            ax.axis('off')
        
        plt.suptitle('{}:{:,}-{:,}'.format(chrom,chromstart,chromend),x=0.9,fontsize=12,horizontalalignment='right')
        plt.tight_layout()
        return fig,axes



                             
def format_number_with_k(number):
    units = ['', 'kb', 'Mb', 'Gb', 'Tb']
    unit_index = 0
    
    # 处理负数情况
    sign = ""
    if number < 0:
        sign = "-"
        number = abs(number)
    
    # 依次除以1000，直到达到合适的单位
    while number >= 1000 and unit_index < len(units) - 1:
        number /= 1000.0
        unit_index += 1
    
    return f"{sign}{number:.0f}{units[unit_index]}"                    
                         
        

def plot_matrix(adata,bw_type='TSS',
                figsize=(2,8),cmap='Greens',
                vmax='auto',vmin='auto',fontsize=12,title=''):
    #from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    fig, ax = plt.subplots(figsize=figsize)
    #pm=np.clip(tss_array.loc[order], 0, 11)

    if vmax=='auto':
        vmax=np.percentile(adata.X.toarray() ,98)
    if vmin=='auto':
        vmin=0

    smp=ax.imshow(adata.X.toarray(),
            aspect='auto',
            interpolation='bilinear',
            origin='upper',
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,)
    cax=plt.colorbar(smp,shrink=0.5)
    cax.ax.tick_params(labelsize=fontsize)
    ax.yaxis.set_visible(False)
    #ax.xaxis.set_visible(False)
    #ax.set_xlabel('TSS')
    # 设置TSS和TES位置的刻度和标签
    if bw_type=='TSS':
        x=adata.uns['bins']
        tss_position = int(adata.shape[1] / 2)  # 1/4位置
        #tes_position = int(2 * 100 / 3)  # 后1/3位置
        ax.set_xticks([0,tss_position,adata.shape[1]])  # 设置刻度位置
        ax.set_xticklabels([format_number_with_k(adata.uns['range'][0]),
                'TSS',format_number_with_k(adata.uns['range'][1])],fontsize=fontsize)  # 设置刻度标签
    elif bw_type=='TES':
        x=adata.uns['bins']
        tes_position = int(adata.shape[1] / 2)
        ax.set_xticks([0,tes_position,adata.shape[1]])  # 设置刻度位置
        ax.set_xticklabels([format_number_with_k(adata.uns['range'][0]),
                'TES',format_number_with_k(adata.uns['range'][1])],fontsize=fontsize)  # 设置刻度标签
    elif bw_type=='body':
        x=adata.uns['bins']
        body_position = int(adata.shape[1] / 2)
        ax.set_xticks([0,body_position,adata.shape[1]])
        ax.set_xticklabels([format_number_with_k(adata.uns['range'][0]),
                'Peak',format_number_with_k(adata.uns['range'][1])],fontsize=fontsize)
    elif bw_type=='all':
        x=adata.uns['bins']
        tss_position = int(adata.shape[1] / 4)
        tes_position = int(adata.shape[1] / 4 *3)
        ax.set_xticks([0,tss_position,tes_position,adata.shape[1]])
        ax.set_xticklabels([format_number_with_k(adata.uns['range'][0]),
                'TSS','TES',format_number_with_k(adata.uns['range'][1])],fontsize=fontsize)
    plt.grid(False)
    plt.tight_layout()
    plt.title(title,fontsize=fontsize)
    return fig,ax

def plot_matrix_line(adata,bw_type='TSS',
                        figsize=(3,3),color='#a51616',
                        linewidth=2,fontsize=13,title=''):

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot([i for i in range(adata.shape[1])],
            adata.X.toarray().mean(axis=0),linewidth=linewidth,color=color)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)

    if bw_type=='TSS':
        x=adata.uns['bins']
        tss_position = int(adata.shape[1] / 2)  # 1/4位置
        #tes_position = int(2 * 100 / 3)  # 后1/3位置

        ax.set_xticks([0,tss_position,adata.shape[1]])  # 设置刻度位置
        ax.set_xticklabels([format_number_with_k(adata.uns['range'][0]),
                'TSS',format_number_with_k(adata.uns['range'][1])],fontsize=fontsize)  # 设置刻度标签
    elif bw_type=='TES':
        x=adata.uns['bins']
        tes_position = int(adata.shape[1] / 2)
        ax.set_xticks([0,tes_position,adata.shape[1]])  # 设置刻度位置
        ax.set_xticklabels([format_number_with_k(adata.uns['range'][0]),
                'TES',format_number_with_k(adata.uns['range'][1])],fontsize=fontsize)  # 设置刻度标签
    elif bw_type=='body':
        x=adata.uns['bins']
        body_position = int(adata.shape[1] / 2)
        ax.set_xticks([0,body_position,adata.shape[1]])
        ax.set_xticklabels([format_number_with_k(adata.uns['range'][0]),
                'Peak',format_number_with_k(adata.uns['range'][1])],fontsize=fontsize)
    elif bw_type=='all':
        x=adata.uns['bins']
        tss_position = int(adata.shape[1] / 4)
        tes_position = int(adata.shape[1] / 4 *3)
        ax.set_xticks([0,tss_position,tes_position,adata.shape[1]])
        ax.set_xticklabels([format_number_with_k(adata.uns['range'][0]),
                'TSS','TES',format_number_with_k(adata.uns['range'][1])],fontsize=fontsize)
    ax.set_yticklabels(ax.get_yticklabels()[1:],fontsize=fontsize)
    
    plt.grid(False)
    plt.tight_layout()
    plt.title(title,fontsize=fontsize)
    return fig,ax  