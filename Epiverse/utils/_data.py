import os 
import time
import requests

def download_gene_annotation_file(download_file:str = ''):
    r"""
    load gene_annotation_file
    """
    _datasets = {
        'chm13v2.0_RefSeq_Liftoff_v4':'https://figshare.com/ndownloader/files/40628072',
        'GRCH38.v44.basic.annotation':'https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_44/gencode.v44.basic.annotation.gtf.gz'
    }
     
    for datasets_name in _datasets.keys():
        print('......Gene Annotation File download start:',datasets_name)
        model_path = data_downloader(url=_datasets[datasets_name],
                                     path=os.path.join(download_file,'{}.gtf.gz'.format(datasets_name)),title=datasets_name)
    print('......Gene Annotation File download finished!')

def download_gene_activity_reference(download_file:str = ''):
    r"""
    load reference data for calculating gene activity.
    """
    _datasets = {
        'GRCm38_refgenes':'https://figshare.com/ndownloader/files/41910918',
        'T2TCHM13_refgenes':'https://figshare.com/ndownloader/files/41910924'
    }
     
    for datasets_name in _datasets.keys():
        print('......Gene Annotation File download start:',datasets_name)
        model_path = data_downloader(url=_datasets[datasets_name],
                                     path=os.path.join(download_file,'{}.txt'.format(datasets_name)),title=datasets_name)
    print('......Gene Annotation File download finished!')

def data_downloader(url,path,title):
    r"""datasets downloader
    
    Arguments
    ---------
    - url: `str`
        the download url of datasets
    - path: `str`
        the save path of datasets
    - title: `str`
        the name of datasets
    
    Returns
    -------
    - path: `str`
        the save path of datasets
    """
    if os.path.isfile(path):
        print("......Loading dataset from {}".format(path))
        return path
    else:
        print("......Downloading dataset save to {}".format(path))
        
    dirname, _ = os.path.split(path)
    try:
        if not os.path.isdir(dirname):
            print("......Creating directory {}".format(dirname))
            os.makedirs(dirname, exist_ok=True)
    except OSError as e:
        print("......Unable to create directory {}. Reason {}".format(dirname,e))
    
    start = time.time()
    size = 0
    res = requests.get(url, stream=True)

    chunk_size = 1024000
    content_size = int(res.headers["content-length"]) 
    if res.status_code == 200:
        print('......[%s Size of file]: %0.2f MB' % (title, content_size/chunk_size/10.24))
        with open(path, 'wb') as f:
            for data in res.iter_content(chunk_size=chunk_size):
                f.write(data)
                size += len(data) 
                print('\r'+ '......[Downloader]: %s%.2f%%' % ('>'*int(size*50/content_size), float(size/content_size*100)), end='')
        end = time.time()
        print('\n' + ".......Finish！%s.2f s" % (end - start))
    
    return path