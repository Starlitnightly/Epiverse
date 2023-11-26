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

def download_genome_sequence(genome_name='', download_folder=''):
    """
    Download genome sequence file for a specified genome and save it in a folder named after the genome.

    Parameters:
    - genome_name (str): The name of the genome for which to download the sequence file.
    - download_folder (str): The local folder where the downloaded file will be saved.

    Returns:
    None

    Note:
    This function uses the data_downloader function to download genome sequence files from UCSC Genome Browser.
    The downloaded file is saved in a folder named after the genome in the specified download folder.
    """
    # Mapping of genome names to their respective download URLs
    _datasets = {
        'hs1': 'https://hgdownload.soe.ucsc.edu/goldenPath/hs1/bigZips/hs1.fa.gz',
        'hg38': 'https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz',
        'hg19': 'https://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/hg19.fa.gz',
        'mm10': 'https://hgdownload.soe.ucsc.edu/goldenPath/mm10/bigZips/mm10.fa.gz',
        'mm39': 'https://hgdownload.soe.ucsc.edu/goldenPath/mm39/bigZips/mm39.fa.gz',
    }

    if genome_name in _datasets:
        # Create a folder for the genome if it doesn't exist
        download_path = os.path.join(download_folder, genome_name)
        os.makedirs(download_path, exist_ok=True)

        print('......Genome Sequence File download start:', genome_name)
        # Download the genome sequence file and save it in the designated folder
        file_path = data_downloader(url=_datasets[genome_name],
                                     path=os.path.join(download_path, f'{genome_name}.fa.gz'), title=genome_name)
        print('......Genome Sequence File download finished! Saved to:', file_path)
    else:
        print(f"Genome Sequence File '{genome_name}' not found in the dataset.")

def download_gene_activity_reference(download_file:str = ''):
    r"""
    load reference data for calculating gene activity.
    """
    _datasets = {
        'GRCm38_refgenes':'https://figshare.com/ndownloader/files/41910918',
        'T2TCHM13_refgenes':'https://figshare.com/ndownloader/files/41910924'
    }
     
    for datasets_name in _datasets.keys():
        print('......Gene Activity Reference File download start:',datasets_name)
        model_path = data_downloader(url=_datasets[datasets_name],
                                     path=os.path.join(download_file,'{}.txt'.format(datasets_name)),title=datasets_name)
    print('......Gene Activity Reference File download finished!')

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
