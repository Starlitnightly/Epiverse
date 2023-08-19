import multiprocessing
import anndata
import numpy as np
import pandas as pd

import multiprocessing
import anndata
import numpy as np
import pandas as pd
import gc

class ATAC_concat(object):
    """
    Concat peaks from different samples
    
    """
    def __init__(self,adata_dict,chr=None,chr_name=['chrom','chromstart','chromend'],
                near_bp=500,ncpus=None):
        """
        Initiate the ATAC_peaks object

        Arguments:
            adata_dict: the dictionary of adata object
            chr: the list of chromosomes
            chr_name: the name of chromosome, start and end
            near_bp: the distance of two peaks to be merged
            ncpus: the number of cpus to be used

        Returns:
            None
        
        """
        self.adata_dict=adata_dict
        self.chr=chr
        self.chr_name=chr_name
        self.near_bp=near_bp
        if ncpus==None:
            self.ncpus=multiprocessing.cpu_count()
        else:
            self.ncpus=ncpus
            

    def init(self):
        """
        Initiate the peaks dictionary of each chromosome

        Arguments:
            None
        
        """
        k=0
        peaks_dict={}
        for adata_key in self.adata_dict.keys():
            if k==0:
                for chr in self.chr:
                    peaks_dict[chr]=self.adata_dict[adata_key].var.loc[self.adata_dict[adata_key].var[self.chr_name[0]]==chr].copy()
            else:
                for chr in self.chr:
                    test_peaks=self.adata_dict[adata_key].var.loc[self.adata_dict[adata_key].var[self.chr_name[0]]==chr].copy()
                    mid_peaks=pd.concat([peaks_dict[chr],test_peaks],axis=0).sort_values([self.chr_name[1]])
                    mid_peaks=mid_peaks[~mid_peaks.index.duplicated(keep='first')]
                    peaks_dict[chr]=mid_peaks.sort_values([self.chr_name[1]])

            k+=1
        
        for chr in self.chr:
            peaks_dict[chr]['range']=[int(i)-int(j) for i,j in zip(peaks_dict[chr][self.chr_name[2]],
                                                                   peaks_dict[chr][self.chr_name[1]])]
            peaks_dict[chr]['range_s']=[(int(i)-int(j))//2 for i,j in zip(peaks_dict[chr][self.chr_name[2]],
                                                                          peaks_dict[chr][self.chr_name[1]])]
            peaks_dict[chr]['median']=[(int(i)-int(j))//2+j for i,j in zip(peaks_dict[chr][self.chr_name[2]],
                                                                           peaks_dict[chr][self.chr_name[1]])]
            peaks_dict[chr]=peaks_dict[chr].sort_values(['median'])

        self.peaks_dict=peaks_dict

    def concat(self,batch=1):
        """
        Concat peaks from different samples

        Arguments:
            batch: the number of chromosomes to be processed in each batch
        
        """
        chunk_size = len(self.chr) // batch  # 计算每份的大小
        chunks = [self.chr[i:i+chunk_size] for i in range(0, len(self.chr), chunk_size)]
        for chunk in chunks:
            with multiprocessing.Pool(processes=self.ncpus) as pool:
                results = pool.map(self.process_chromosome, chunk)
            for result in results:
                chr_name=result[self.chr_name[0]].iloc[0]
                self.peaks_dict[chr_name] = result
            gc.collect()

        k=0
        for chr in self.chr:
            if k==0:
                total_peak=self.peaks_dict[chr]
            else:
                total_peak=pd.concat([total_peak,self.peaks_dict[chr]],axis=0)
            k+=1
        
        split = total_peak['new_peak'].str.split(r"[:-]")
        total_peak["new_chrom"] = split.map(lambda x: x[0])
        total_peak["new_chromstart"] = split.map(lambda x: x[1]).astype(int)
        total_peak["new_chromend"] = split.map(lambda x: x[2]).astype(int)
        self.total_peak=total_peak
        return total_peak
        #for chr, peaks in results:
        #    self.peaks_dict[chr] = peaks

    def reindex(self,s=':-'):
        """
        reindex adata from different samples

        Arguments:
            s: the symbol of re in the peak name ':-': chr1:1000-2000, '__':chr1_1000_2000
        
        """
        for adata_key in self.adata_dict.keys():
            self.adata_dict[adata_key].var['old_peak']=self.adata_dict[adata_key].var.index
            self.adata_dict[adata_key].var.index=self.total_peak.loc[self.adata_dict[adata_key].var.index]['new_peak'].values
            split = self.adata_dict[adata_key].var.index.str.split(r"[{}]".format(s))
            self.adata_dict[adata_key].var["new_chrom"] = split.map(lambda x: x[0])
            self.adata_dict[adata_key].var["new_chromstart"] = split.map(lambda x: x[1]).astype(int)
            self.adata_dict[adata_key].var["new_chromend"] = split.map(lambda x: x[2]).astype(int)
            
    def merge(self,method='inner'):
        """
        Merge adata from different samples

        Arguments:
            method: 'inner' or 'outer'

        Returns:
            adata: merged adata
        
        """
        if method=='inner':
            k=0
            for adata_key in self.adata_dict.keys():
                self.adata_dict[adata_key].var_names_make_unique()
                if k==0:
                    concat_adata=self.adata_dict[adata_key]
                else:
                    concat_adata=anndata.concat([concat_adata,self.adata_dict[adata_key]],merge='same')
                k+=1
        elif method=='outer':
            k=0
            for adata_key in self.adata_dict.keys():
                self.adata_dict[adata_key].var_names_make_unique()
                if k==0:
                    concat_adata=self.adata_dict[adata_key]
                else:
                    concat_adata=anndata.concat([concat_adata,self.adata_dict[adata_key]],join="outer",fill_value=0)
                k+=1
        concat_adata.obs_names_make_unique()
        return concat_adata
    
    def process_chromosome(self, chr):
        print(chr)
        peak_pd = self.peaks_dict[chr].copy()
        return self.peak_process(peak_pd, chr)

    def peak_process(self,peak_pd,chrom='chr1'):
       
        processed_peak=[]
        new_peak=[]
        processed_num=1
        k=0
        test2=peak_pd.copy()
        for peak in peak_pd.index:
            ##判断是否重复计算
            if peak in processed_peak:
                continue
        
            #取当前peak
            peak_test=test2.loc[peak]
            peak_range_start=peak_test[self.chr_name[1]]
            peak_range_end=peak_test[self.chr_name[2]]
            peak_large_range_end=max(peak_test['median']+self.near_bp,peak_range_end)
            #print(peak_range_start,peak_range_end,peak_large_range_end)
        
            #取在范围内的所有peak
            peak_merge=test2.loc[
                (test2[self.chr_name[1]]<peak_range_end)&
                (test2[self.chr_name[2]]>peak_test['median'])&
                (test2[self.chr_name[2]]<=peak_large_range_end)
            ]
            
            #判断范围内的peak数量
            if peak_merge.shape[0]>1:
                merge_start=np.min(peak_merge[self.chr_name[1]])
                merge_end=np.max(peak_merge[self.chr_name[2]])
            else:
                merge_start=np.min(peak_merge[self.chr_name[1]])
                merge_end=np.max(peak_merge[self.chr_name[2]])
        
            #构建新的peak列表
            new_peaks=[f"{chrom}:{merge_start}-{merge_end}"]*peak_merge.shape[0]
        
            #去除已经处理的peak
            processed_num+=peak_merge.shape[0]
            processed_peaks=peak_merge.index.tolist()
            test2.drop(processed_peaks, inplace=True)
        
            #添加已处理项目
            for n_p,p_p in zip(new_peaks,processed_peaks):
                new_peak.append(n_p)
                processed_peak.append(p_p)
        peak_pd['new_peak']=new_peak
        gc.collect()
        return peak_pd
                