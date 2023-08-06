import pyBigWig
import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
import omicverse as ov

sc_color=['#7CBB5F','#368650','#A499CC','#5E4D9A','#78C2ED','#866017', '#9F987F','#E0DFED',
 '#EF7B77', '#279AD7','#F0EEF0', '#1F577B', '#A56BA7', '#E0A7C8', '#E069A6', '#941456', '#FCBC10',
 '#EAEFC5', '#01A0A7', '#75C8CC', '#F0D7BC', '#D5B26C', '#D5DA48', '#B6B812', '#9DC3C3', '#A89C92', '#FEE00C', '#FEF2A1']

class bigwig(object):
    
    def __init__(self,bw_path_dict):
        self.bw_path_dict=bw_path_dict
        self.bw_names=list(bw_path_dict.keys())
        self.bw_lens=len(self.bw_names)
        self.gtf=None
    
    def read(self):
        self.bw_dict={}
        for bw_name in self.bw_names:
            self.bw_dict[bw_name]=pyBigWig.open(self.bw_path_dict[bw_name])
            
    def close(self):
        for bw_name in self.bw_names:
            self.bw_dict[bw_name].close()
    
    def load_gtf(self,gtf_path):
        self.gtf=ov.utils.read_gtf(gtf_path)
            
    def plot_track(self,chrom,chromstart,chromend,nbins=700,
                   value_type='mean',transform='no',
                   figwidth=6,figheight=1,plot_names=None,
                   color_dict=None,region_dict=None,
                   gtf_color='#000000',prefered_name='gene_id'):
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
        
        if self.gtf is None:
            plot_axes=axes[:-1]
        else:
            plot_axes=axes
        
        if bw_lens==1:
            plot_axes=[plot_axes]

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
                    gene_attr=plot_obj.attribute.split(';')
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



                             
                             
                         
            
        