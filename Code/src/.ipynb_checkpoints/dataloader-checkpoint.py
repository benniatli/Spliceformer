# +
import numpy as np
import torch
from torch.utils.data import Dataset
import pickle
import pandas as pd
from scipy.sparse import load_npz
from glob import glob
from math import ceil
import h5py

OUT_MAP = np.asarray([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1],
                      [0, 0, 0]])

IN_MAP = np.asarray([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])


# -

def getData(data_dir,setType):
    if setType == 'train':
        chroms = ['chr11', 'chr13', 'chr15', 'chr17', 'chr19', 'chr21',
                               'chr2', 'chr4', 'chr6', 'chr8', 'chr10', 'chr12',
                               'chr14', 'chr16', 'chr18', 'chr20', 'chr22', 'chrX', 'chrY']
    if setType == 'test':
        chroms = ['chr1', 'chr3', 'chr5', 'chr7', 'chr9']
    if setType == 'all':
        chroms = ['chr11', 'chr13', 'chr15', 'chr17', 'chr19','chr1', 'chr21',
                               'chr2', 'chr4', 'chr6', 'chr8', 'chr10', 'chr12',
                               'chr14', 'chr16', 'chr18', 'chr20', 'chr22', 'chr3', 'chr5', 'chr7', 'chr9', 'chrX', 'chrY']
    if setType == 'all':
        with open('{}/sparse_discrete_label_data_train.pickle'.format(data_dir), 'rb') as handle:
            transcriptToLabel = pickle.load(handle)
        with open('{}/sparse_discrete_label_data_test.pickle'.format(data_dir), 'rb') as handle:
            transcriptToLabel2 = pickle.load(handle)
            transcriptToLabel.update(transcriptToLabel2)
    else:
        with open('{}/sparse_discrete_label_data_{}.pickle'.format(data_dir,setType), 'rb') as handle:
            transcriptToLabel = pickle.load(handle)
    
    if setType == 'all':
        annotation1 = pd.read_csv(data_dir+'/annotation_ensembl_v87_train.txt',sep='\t',header=None)[[0,1,2,3,4]]
        annotation2 = pd.read_csv(data_dir+'/annotation_ensembl_v87_test.txt',sep='\t',header=None)[[0,1,2,3,4]]
        annotation = pd.concat([annotation1,annotation2],axis=0)
    else:
        annotation = pd.read_csv(data_dir+'/annotation_ensembl_v87_{}.txt'.format(setType),sep='\t',header=None)[[0,1,2,3,4]]
    annotation.columns = ['name','chrom','strand','tx_start','tx_end']
    annotation['transcript'] = annotation['name'].apply(lambda x: x.split('---')[-2].split('.')[0]).values
    annotation['gene'] = annotation['name'].apply(lambda x: x.split('---')[-3].split('.')[0]).values
    #annotation['support'] = annotation['transcript'].apply(lambda x:transcriptToSupport[x])

    chrom_paths = glob(data_dir+'/sparse_sequence_data/*')
    chromToPath = {}
    for path in chrom_paths:
        chromToPath[path.split('/')[-1].split('_')[0]] = path

    seqData = {}
    for chrom in chroms:
        path = glob(data_dir+'/sparse_sequence_data/{}_*.npz'.format(chrom))[0]
        seqData[chrom] = load_npz(path).tocsr()
    return annotation,transcriptToLabel,seqData


# +
def get_GTEX_Data(data_dir,setType,anno_name = 'annotation_GTEX_v9.txt'):
    if setType == 'train':
        chroms = ['chr11', 'chr13', 'chr15', 'chr17', 'chr19', 'chr21',
                               'chr2', 'chr4', 'chr6', 'chr8', 'chr10', 'chr12',
                               'chr14', 'chr16', 'chr18', 'chr20', 'chr22', 'chrX', 'chrY']
    if setType == 'test':
        chroms = ['chr1', 'chr3', 'chr5', 'chr7', 'chr9']
    if setType == 'all':
        chroms = ['chr11', 'chr13', 'chr15', 'chr17', 'chr19','chr1', 'chr21',
                               'chr2', 'chr4', 'chr6', 'chr8', 'chr10', 'chr12',
                               'chr14', 'chr16', 'chr18', 'chr20', 'chr22', 'chr3', 'chr5', 'chr7', 'chr9', 'chrX', 'chrY']
    
    with open('{}/gene_to_label.pickle'.format(data_dir,setType), 'rb') as handle:
        gene_to_label = pickle.load(handle)
    

    annotation = pd.read_csv(data_dir+'/{}'.format(anno_name),sep='\t',header=None)[[0,1,2,3,4,5,6]]

    annotation.columns = ['name','gene','level','chrom','strand','tx_start','tx_end']
    annotation = annotation[annotation['chrom'].isin(chroms)]

    seqData = {}
    for chrom in chroms:
        seqData[chrom] = load_npz(data_dir+'/sparse_sequence_data/{}.npz'.format(chrom)).tocsr()
    return annotation,gene_to_label,seqData

def get_GTEX_v8_Data(data_dir,setType,anno_name):
    if setType == 'train':
        chroms = ['chr11', 'chr13', 'chr15', 'chr17', 'chr19', 'chr21',
                               'chr2', 'chr4', 'chr6', 'chr8', 'chr10', 'chr12',
                               'chr14', 'chr16', 'chr18', 'chr20', 'chr22']
    if setType == 'test':
        chroms = ['chr1', 'chr3', 'chr5', 'chr7', 'chr9']
    if setType == 'all':
        chroms = ['chr11', 'chr13', 'chr15', 'chr17', 'chr19','chr1', 'chr21',
                               'chr2', 'chr4', 'chr6', 'chr8', 'chr10', 'chr12',
                               'chr14', 'chr16', 'chr18', 'chr20', 'chr22', 'chr3', 'chr5', 'chr7', 'chr9']
    
    with open('{}/gene_to_label.pickle'.format(data_dir,setType), 'rb') as handle:
        gene_to_label = pickle.load(handle)
    

    annotation = pd.read_csv(data_dir+'/{}'.format(anno_name),sep='\t',header=None)[[0,1,2,3,4,5,6,7]]

    annotation.columns = ['name','gene','transcript','level','chrom','strand','tx_start','tx_end']
    annotation = annotation[annotation['chrom'].isin(chroms)]

    seqData = {}
    for chrom in chroms:
        seqData[chrom] = load_npz(data_dir+'/sparse_sequence_data/{}.npz'.format(chrom)).tocsr()
    return annotation,gene_to_label,seqData


# +
class DataPoint:
    def __init__(self, transcript,gene,chrom,strand,start,end,tx_start,tx_end,splice_loc,splice_type,SL,CL_max,shift,mask_n):
        self.transcript = transcript
        self.gene = gene
        self.chrom = chrom
        self.strand = strand
        self.start = start
        self.end = end
        self.tx_start = tx_start
        self.tx_end = tx_end
        self.splice_loc = splice_loc
        self.splice_type = splice_type
        self.SL = SL
        self.CL_max = CL_max
        self.shift = shift
        self.mask_n = mask_n
        
    def getData(self,seqData):
        X = np.zeros((4,self.SL+self.CL_max))
        Y = np.zeros((3,self.SL))
        
        low = np.max([self.tx_start,self.start-self.CL_max//2])
        low_diff = low-self.start+self.CL_max//2
        high = np.min([self.tx_end,self.end+self.CL_max//2])
        high_diff = high-(self.end+self.CL_max//2)
        X[:,low_diff:self.CL_max+self.SL+high_diff] = seqData[self.chrom][low-1:high,:4].toarray().T
        Y[0,:(self.SL-self.mask_n)] = np.ones(self.SL-self.mask_n)
        Y[:,self.splice_loc-self.shift] = OUT_MAP[np.array(self.splice_type).astype('int8')].T
        #X = seqData[self.chrom][self.start-1:self.end,:4].toarray().T
        if self.strand=='-':
            X = X[:,::-1]
            X = X[::-1,:]
        return X.copy(),Y.copy()
    
def getDataPointList(annotation,transcriptToLabel,SL,CL_max,shift):
    data = []
    for idx in range(annotation.shape[0]):
        transcript,gene,chrom,strand,tx_start,tx_end = annotation['transcript'].values[idx], annotation['gene'].values[idx],annotation['chrom'].values[idx],annotation['strand'].values[idx],annotation['tx_start'].values[idx],annotation['tx_end'].values[idx]
        length = tx_end-tx_start+1
        num_points = ceil_div(length, shift)
        label = [np.array(x) for x in transcriptToLabel[transcript]]

        for i in range(num_points):
            if strand=='+':
                start,end = tx_start+shift*i,tx_start+SL+shift*(i)-1
                if i == 0:
                    start_point = start
                inRange = [l>=start-start_point and l<=end-start_point for l in label[1]]
                mask_n = np.max([end,tx_end])-tx_end
                data.append(DataPoint(transcript,gene,chrom,strand,start,end,tx_start,tx_end,label[1][inRange],label[0][inRange],SL,CL_max,start-start_point,mask_n))
            else:
                start,end = tx_end-SL-shift*i+1,tx_end-shift*i
                if i == 0:
                    start_point = end
                inRange = [l>=start_point-end and l<= start_point-start for l in label[1]]
                mask_n = tx_start-np.min([start,tx_start])
                data.append(DataPoint(transcript,gene,chrom,strand,start,end,tx_start,tx_end,label[1][inRange],label[0][inRange],SL,CL_max,start_point-end,mask_n))
    return data


# +
class DataPointFullWithNs:
    def __init__(self, transcript,gene,chrom,strand,start,end,tx_start,tx_end,splice_loc,splice_type,SL,CL_max,shift,mask_l,mask_r,include_pos=False):
        self.transcript = transcript
        self.gene = gene
        self.chrom = chrom
        self.strand = strand
        self.start = start
        self.end = end
        self.tx_start = tx_start
        self.tx_end = tx_end
        self.splice_loc = splice_loc
        self.splice_type = splice_type
        self.SL = SL
        self.CL_max = CL_max
        self.shift = shift
        self.mask_l = mask_l
        self.mask_r = mask_r
        self.include_pos = include_pos
        
    def getData(self,seqData):
        X = np.zeros((5,self.SL+self.CL_max))
        #X[4,:] = np.ones((self.SL+self.CL_max))
        r = np.zeros((self.SL+self.CL_max))
        Y = np.zeros((3,self.SL+self.CL_max))
        
        #low = np.max([self.tx_start,self.start-self.CL_max//2])
        #low_diff = low-self.start+self.CL_max//2
        #high = np.min([self.tx_end,self.end+self.CL_max//2])
        #high_diff = high-(self.end+self.CL_max//2)
        #X[:,low_diff:self.CL_max+self.SL+high_diff] = seqData[self.chrom][low-1:high,:4].toarray().T
        if self.include_pos:
            pos = np.arange(self.start-1,self.end)
            chrm = np.repeat(self.chrom,self.SL)
            transcript = np.repeat(self.transcript,self.SL)
        
        if self.strand=='+':
            X[:,self.mask_l:self.SL+self.CL_max-self.mask_r] = seqData[self.chrom][self.start-self.CL_max//2-1+self.mask_l:self.end+self.CL_max//2-self.mask_r,:].toarray().T
            r[self.mask_l:self.SL+self.CL_max-self.mask_r] = seqData[self.chrom][self.start-self.CL_max//2-1+self.mask_l:self.end+self.CL_max//2-self.mask_r,4].toarray()[:,0]
        else:
            X[:,self.mask_r:self.SL+self.CL_max-self.mask_l] = seqData[self.chrom][self.start-self.CL_max//2-1+self.mask_r:self.end+self.CL_max//2-self.mask_l,:].toarray().T
            r[self.mask_r:self.SL+self.CL_max-self.mask_l] = seqData[self.chrom][self.start-self.CL_max//2-1+self.mask_r:self.end+self.CL_max//2-self.mask_l,4].toarray()[:,0]
            X[:4,:] = X[:4,:][::-1,:]
            X = X[:,::-1]
            r = r[::-1]
            
        Y[0,self.mask_l:(self.SL+self.CL_max-self.mask_r)] = np.ones(self.SL+self.CL_max-self.mask_r-self.mask_l)
        Y[:,self.splice_loc-self.shift+self.CL_max//2] = OUT_MAP[np.array(self.splice_type,dtype=np.int8)].T
        r_sum = np.sum(r)
        if r_sum>0:
            Y[:,r==1] =  OUT_MAP[3*np.ones(int(r_sum),dtype=np.int8)].T
        
        if self.include_pos:
            return X.copy(),Y.copy(),pos,chrm,transcript
        else:
            return X.copy(),Y.copy()
        
def getDataPointListFullWithNs(annotation,transcriptToLabel,SL,CL_max,shift,include_pos=False):
    data = []
    for idx in range(annotation.shape[0]):
        transcript,gene,chrom,strand,tx_start,tx_end = annotation['transcript'].values[idx], annotation['gene'].values[idx],annotation['chrom'].values[idx],annotation['strand'].values[idx],annotation['tx_start'].values[idx],annotation['tx_end'].values[idx]
        length = tx_end-tx_start+1
        num_points = ceil_div(length, shift)
        label = [np.array(x) for x in transcriptToLabel[transcript]]

        for i in range(num_points):
            if strand=='+':
                start,end = tx_start+shift*i,tx_start+SL+shift*(i)-1
                if i == 0:
                    start_point = start
                inRange = [l>=start-start_point-CL_max//2 and l<=end-start_point+CL_max//2 for l in label[1]]
                mask_l = tx_start-np.min([start-CL_max//2,tx_start])
                mask_r = np.max([end+CL_max//2,tx_end])-tx_end
                data.append(DataPointFullWithNs(transcript,gene,chrom,strand,start,end,tx_start,tx_end,label[1][inRange],label[0][inRange],SL,CL_max,start-start_point,mask_l,mask_r,include_pos))
            else:
                start,end = tx_end-SL-shift*i+1,tx_end-shift*i
                if i == 0:
                    start_point = end
                inRange = [l>=start_point-end-CL_max//2 and l<= start_point-start+CL_max//2 for l in label[1]]
                mask_l =  np.max([end+CL_max//2,tx_end])-tx_end
                mask_r = tx_start-np.min([start-CL_max//2,tx_start])
                data.append(DataPointFullWithNs(transcript,gene,chrom,strand,start,end,tx_start,tx_end,label[1][inRange],label[0][inRange],SL,CL_max,start_point-end,mask_l,mask_r,include_pos))
    return data   


# +
class DataPointFull:
    def __init__(self, transcript,gene,chrom,strand,start,end,tx_start,tx_end,splice_loc,splice_type,SL,CL_max,shift,mask_l,mask_r,include_pos=False):
        self.transcript = transcript
        self.gene = gene
        self.chrom = chrom
        self.strand = strand
        self.start = start
        self.end = end
        self.tx_start = tx_start
        self.tx_end = tx_end
        self.splice_loc = splice_loc
        self.splice_type = splice_type
        self.SL = SL
        self.CL_max = CL_max
        self.shift = shift
        self.mask_l = mask_l
        self.mask_r = mask_r
        self.include_pos = include_pos
        
    def getData(self,seqData):
        X = np.zeros((4,self.SL+self.CL_max))
        r = np.zeros((self.SL+self.CL_max))
        Y = np.zeros((3,self.SL+self.CL_max))
        
        #low = np.max([self.tx_start,self.start-self.CL_max//2])
        #low_diff = low-self.start+self.CL_max//2
        #high = np.min([self.tx_end,self.end+self.CL_max//2])
        #high_diff = high-(self.end+self.CL_max//2)
        #X[:,low_diff:self.CL_max+self.SL+high_diff] = seqData[self.chrom][low-1:high,:4].toarray().T
        if self.include_pos:
            pos = np.arange(self.start-1,self.end)
            chrm = np.repeat(self.chrom,self.SL)
            transcript = np.repeat(self.transcript,self.SL)
        
        if self.strand=='+':
            X[:,self.mask_l:self.SL+self.CL_max-self.mask_r] = seqData[self.chrom][self.start-self.CL_max//2-1+self.mask_l:self.end+self.CL_max//2-self.mask_r,:4].toarray().T
            r[self.mask_l:self.SL+self.CL_max-self.mask_r] = seqData[self.chrom][self.start-self.CL_max//2-1+self.mask_l:self.end+self.CL_max//2-self.mask_r,4].toarray()[:,0]
        else:
            X[:,self.mask_r:self.SL+self.CL_max-self.mask_l] = seqData[self.chrom][self.start-self.CL_max//2-1+self.mask_r:self.end+self.CL_max//2-self.mask_l,:4].toarray().T
            r[self.mask_r:self.SL+self.CL_max-self.mask_l] = seqData[self.chrom][self.start-self.CL_max//2-1+self.mask_r:self.end+self.CL_max//2-self.mask_l,4].toarray()[:,0]
            X = X[:,::-1]
            r = r[::-1]
            X = X[::-1,:]
        Y[0,self.mask_l:(self.SL+self.CL_max-self.mask_r)] = np.ones(self.SL+self.CL_max-self.mask_r-self.mask_l)
        Y[:,self.splice_loc-self.shift+self.CL_max//2] = OUT_MAP[np.array(self.splice_type,dtype=np.int8)].T
        r_sum = np.sum(r)
        if r_sum>0:
            Y[:,r==1] =  OUT_MAP[3*np.ones(int(r_sum),dtype=np.int8)].T
        
        if self.include_pos:
            return X.copy(),Y.copy(),pos,chrm,transcript
        else:
            return X.copy(),Y.copy()
    
def getDataPointListFull(annotation,transcriptToLabel,SL,CL_max,shift,include_pos=False):
    data = []
    for idx in range(annotation.shape[0]):
        transcript,gene,chrom,strand,tx_start,tx_end = annotation['transcript'].values[idx], annotation['gene'].values[idx],annotation['chrom'].values[idx],annotation['strand'].values[idx],annotation['tx_start'].values[idx],annotation['tx_end'].values[idx]
        length = tx_end-tx_start+1
        num_points = ceil_div(length, shift)
        label = [np.array(x) for x in transcriptToLabel[transcript]]

        for i in range(num_points):
            if strand=='+':
                start,end = tx_start+shift*i,tx_start+SL+shift*(i)-1
                if i == 0:
                    start_point = start
                inRange = [l>=start-start_point-CL_max//2 and l<=end-start_point+CL_max//2 for l in label[1]]
                mask_l = tx_start-np.min([start-CL_max//2,tx_start])
                mask_r = np.max([end+CL_max//2,tx_end])-tx_end
                data.append(DataPointFull(transcript,gene,chrom,strand,start,end,tx_start,tx_end,label[1][inRange],label[0][inRange],SL,CL_max,start-start_point,mask_l,mask_r,include_pos))
            else:
                start,end = tx_end-SL-shift*i+1,tx_end-shift*i
                if i == 0:
                    start_point = end
                inRange = [l>=start_point-end-CL_max//2 and l<= start_point-start+CL_max//2 for l in label[1]]
                mask_l =  np.max([end+CL_max//2,tx_end])-tx_end
                mask_r = tx_start-np.min([start-CL_max//2,tx_start])
                data.append(DataPointFull(transcript,gene,chrom,strand,start,end,tx_start,tx_end,label[1][inRange],label[0][inRange],SL,CL_max,start_point-end,mask_l,mask_r,include_pos))
    return data


# -

def getDataPointListSeqFull(annotation,transcriptToLabel,SL,CL_max,shift,tokenizer,include_pos=False):
    data = []
    for idx in range(annotation.shape[0]):
        transcript,gene,chrom,strand,tx_start,tx_end = annotation['transcript'].values[idx], annotation['gene'].values[idx],annotation['chrom'].values[idx],annotation['strand'].values[idx],annotation['tx_start'].values[idx],annotation['tx_end'].values[idx]
        length = tx_end-tx_start+1
        num_points = ceil_div(length, shift)
        label = [np.array(x) for x in transcriptToLabel[transcript]]

        for i in range(num_points):
            if strand=='+':
                start,end = tx_start+shift*i,tx_start+SL+shift*(i)-1
                if i == 0:
                    start_point = start
                inRange = [l>=start-start_point-CL_max//2 and l<=end-start_point+CL_max//2 for l in label[1]]
                mask_l = tx_start-np.min([start-CL_max//2,tx_start])
                mask_r = np.max([end+CL_max//2,tx_end])-tx_end
                data.append(DataPointSeqFull(transcript,gene,chrom,strand,start,end,tx_start,tx_end,label[1][inRange],label[0][inRange],SL,CL_max,start-start_point,mask_l,mask_r,tokenizer,include_pos))
            else:
                start,end = tx_end-SL-shift*i+1,tx_end-shift*i
                if i == 0:
                    start_point = end
                inRange = [l>=start_point-end-CL_max//2 and l<= start_point-start+CL_max//2 for l in label[1]]
                mask_l =  np.max([end+CL_max//2,tx_end])-tx_end
                mask_r = tx_start-np.min([start-CL_max//2,tx_start])
                data.append(DataPointSeqFull(transcript,gene,chrom,strand,start,end,tx_start,tx_end,label[1][inRange],label[0][inRange],SL,CL_max,start_point-end,mask_l,mask_r,tokenizer,include_pos))
    return data


class DataPointSeqFull:
    def __init__(self, transcript,gene,chrom,strand,start,end,tx_start,tx_end,splice_loc,splice_type,SL,CL_max,shift,mask_l,mask_r,tokenizer,include_pos=False):
        self.transcript = transcript
        self.gene = gene
        self.chrom = chrom
        self.strand = strand
        self.start = start
        self.end = end
        self.tx_start = tx_start
        self.tx_end = tx_end
        self.splice_loc = splice_loc
        self.splice_type = splice_type
        self.SL = SL
        self.CL_max = CL_max
        self.shift = shift
        self.mask_l = mask_l
        self.mask_r = mask_r
        self.include_pos = include_pos
        self.tokenizer = tokenizer
        
    def one_hot_to_sequence(self,one_hot_array):
        # Define mapping of indices to bases
        bases = np.array(['A', 'C', 'G', 'T', 'N'])

        # Convert transposed one-hot array to sequence indices
        sequence_indices = np.argmax(one_hot_array, axis=0)

        # Map indices to bases to get the sequence
        sequence = bases[sequence_indices]
        # Convert the array to a string
        #sequence_string = ''.join(sequence)
        return sequence

        
    def getData(self,seqData):
        X = np.zeros((4,self.SL+self.CL_max))
        r = np.zeros((self.SL+self.CL_max))
        Y = np.zeros((3,self.SL+self.CL_max))
        
        #low = np.max([self.tx_start,self.start-self.CL_max//2])
        #low_diff = low-self.start+self.CL_max//2
        #high = np.min([self.tx_end,self.end+self.CL_max//2])
        #high_diff = high-(self.end+self.CL_max//2)
        #X[:,low_diff:self.CL_max+self.SL+high_diff] = seqData[self.chrom][low-1:high,:4].toarray().T
        if self.include_pos:
            pos = np.arange(self.start-1,self.end)
            chrm = np.repeat(self.chrom,self.SL)
            transcript = np.repeat(self.transcript,self.SL)
        
        if self.strand=='+':
            X[:,self.mask_l:self.SL+self.CL_max-self.mask_r] = seqData[self.chrom][self.start-self.CL_max//2-1+self.mask_l:self.end+self.CL_max//2-self.mask_r,:4].toarray().T
            r[self.mask_l:self.SL+self.CL_max-self.mask_r] = seqData[self.chrom][self.start-self.CL_max//2-1+self.mask_l:self.end+self.CL_max//2-self.mask_r,4].toarray()[:,0]
        else:
            X[:,self.mask_r:self.SL+self.CL_max-self.mask_l] = seqData[self.chrom][self.start-self.CL_max//2-1+self.mask_r:self.end+self.CL_max//2-self.mask_l,:4].toarray().T
            r[self.mask_r:self.SL+self.CL_max-self.mask_l] = seqData[self.chrom][self.start-self.CL_max//2-1+self.mask_r:self.end+self.CL_max//2-self.mask_l,4].toarray()[:,0]
            X = X[:,::-1]
            r = r[::-1]
            X = X[::-1,:]
        Y[0,self.mask_l:(self.SL+self.CL_max-self.mask_r)] = np.ones(self.SL+self.CL_max-self.mask_r-self.mask_l)
        Y[:,self.splice_loc-self.shift+self.CL_max//2] = OUT_MAP[np.array(self.splice_type,dtype=np.int8)].T
        r_sum = np.sum(r)
        if r_sum>0:
            Y[:,r==1] =  OUT_MAP[3*np.ones(int(r_sum),dtype=np.int8)].T
        
        X = self.one_hot_to_sequence(X)
        #X = self.tokenizer(X)
        
        if self.include_pos:
            return X.copy(), Y.copy(), pos, chrm, transcript
        else:
            return X.copy(),Y.copy()


# +
class DataPointGTEX:
    def __init__(self, gene,chrom,strand,start,end,tx_start,tx_end,splice_loc,splice_type,SL,CL_max,shift,mask_l,mask_r):
        self.gene = gene
        self.chrom = chrom
        self.strand = strand
        self.start = start
        self.end = end
        self.tx_start = tx_start
        self.tx_end = tx_end
        self.splice_loc = splice_loc
        self.splice_type = splice_type
        self.SL = SL
        self.CL_max = CL_max
        self.shift = shift
        self.mask_l = mask_l
        self.mask_r = mask_r
        
    def getData(self,seqData):
        X = np.zeros((4,self.SL+self.CL_max))
        r = np.zeros((self.SL+self.CL_max))
        Y = np.zeros((3,self.SL+self.CL_max))
        
        if self.strand=='+':
            X[:,self.mask_l:self.SL+self.CL_max-self.mask_r] = seqData[self.chrom][self.start-self.CL_max//2-1+self.mask_l:self.end+self.CL_max//2-self.mask_r,:4].toarray().T
            r[self.mask_l:self.SL+self.CL_max-self.mask_r] = seqData[self.chrom][self.start-self.CL_max//2-1+self.mask_l:self.end+self.CL_max//2-self.mask_r,4].toarray()[:,0]
        else:
            X[:,self.mask_r:self.SL+self.CL_max-self.mask_l] = seqData[self.chrom][self.start-self.CL_max//2-1+self.mask_r:self.end+self.CL_max//2-self.mask_l,:4].toarray().T
            r[self.mask_r:self.SL+self.CL_max-self.mask_l] = seqData[self.chrom][self.start-self.CL_max//2-1+self.mask_r:self.end+self.CL_max//2-self.mask_l,4].toarray()[:,0]
            X = X[:,::-1]
            r = r[::-1]
            X = X[::-1,:]
        Y[0,self.mask_l:(self.SL+self.CL_max-self.mask_r)] = np.ones(self.SL+self.CL_max-self.mask_r-self.mask_l)
        Y[:,self.splice_loc-self.shift+self.CL_max//2] = self.splice_type.T
        r_sum = np.sum(r)
        if r_sum>0:
            Y[:,r==1] =  OUT_MAP[3*np.ones(int(r_sum),dtype=np.int8)].T

        return X.copy(),Y.copy()
    
def label_map(val,stype):
    label = np.zeros(3)
    label[stype] = val
    label[0] = 1-val
    return label

def getDataPointListGTEX(annotation,gene_to_label,SL,CL_max,shift):
    data = []
    for idx in range(annotation.shape[0]):
        gene,chrom,strand,tx_start,tx_end = annotation['gene'].values[idx],annotation['chrom'].values[idx],annotation['strand'].values[idx],annotation['tx_start'].values[idx],annotation['tx_end'].values[idx]
        length = tx_end-tx_start+1
        num_points = ceil_div(length, shift)
        junction_start, junction_end = gene_to_label[gene]
        junction_start = junction_start.items()
        junction_end = junction_end.items()
        junction_start = [x for x in junction_start if (x[0] > tx_start and x[0] < tx_end)]
        junction_end = [x for x in junction_end if (x[0] > tx_start and x[0] < tx_end)]
        if strand=='+':
            #tx_end = np.max([x[0] for x in junction_start]+[x[0] for x in junction_end])
            loc = [x[0]-tx_start for x in junction_start]+[x[0]-tx_start for x in junction_end]
            stype = [label_map(x[1],2) for x in junction_start]+[label_map(x[1],1) for x in junction_end]
        else:
            #tx_start = np.min([x[0] for x in junction_start]+[x[0] for x in junction_end])
            #tx_end = np.max([x[0] for x in junction_start]+[x[0] for x in junction_end])
            loc = [tx_end-x[0] for x in junction_start]+[tx_end-x[0] for x in junction_end]
            stype = [label_map(x[1],1) for x in junction_start]+[label_map(x[1],2) for x in junction_end]
            
        
        label = [np.array(stype).astype(int),np.array(loc).astype(int)]
        
        for i in range(num_points):
            if strand=='+':
                start,end = tx_start+shift*i,tx_start+SL+shift*(i)-1
                if i == 0:
                    start_point = start
                inRange = [l>=start-start_point-CL_max//2 and l<=end-start_point+CL_max//2 for l in label[1]]
                mask_l = tx_start-np.min([start-CL_max//2,tx_start])
                mask_r = np.max([end+CL_max//2,tx_end])-tx_end
                data.append(DataPointGTEX(gene,chrom,strand,start,end,tx_start,tx_end,label[1][inRange],label[0][inRange],SL,CL_max,start-start_point,mask_l,mask_r))
            else:
                start,end = tx_end-SL-shift*i+1,tx_end-shift*i
                if i == 0:
                    start_point = end
                inRange = [l>=start_point-end-CL_max//2 and l<= start_point-start+CL_max//2 for l in label[1]]
                mask_l =  np.max([end+CL_max//2,tx_end])-tx_end
                mask_r = tx_start-np.min([start-CL_max//2,tx_start])
                data.append(DataPointGTEX(gene,chrom,strand,start,end,tx_start,tx_end,label[1][inRange],label[0][inRange],SL,CL_max,start_point-end,mask_l,mask_r))
    return data


# -

class spliceDataset(Dataset):
    def __init__(self, annotation, transform=None, target_transform=None):
        self.annotation = annotation
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, idx):
        X,Y = self.annotation[idx].getData(self.seqData)
        return X,Y

def ceil_div(x, y):
    return int(ceil(float(x)/y))

class DataLoaderWrapper:

    def __init__(self, data_loader, steps_per_epoch: int = 1000):
        self.data_loader = data_loader
        self.iterator = iter(self.data_loader)
        self.steps_per_epoch = steps_per_epoch
        self.current_step = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_step < self.steps_per_epoch:
            self.current_step += 1
            try:
                return next(self.iterator)
            except StopIteration:
                self.iterator = iter(self.data_loader)
                return next(self.iterator)
        else:
            self.current_step = 0
            raise StopIteration


# +
#def collate_fn(data):
#    """
#       data: is a list of tuples with (example, label)
#             where 'example' is a tensor of arbitrary shape
#             and label/length are scalars
#    """
#    #unfold1 = nn.Unfold((SL*3,1),SL,CL_max//2)
#    #unfold2 = nn.Unfold((SL,1),SL,0)
#    features = []
#    labels = []
#   for i in range(len(data)):
#        features.append(torch.Tensor(data[i][0]))
#        labels.append(torch.Tensor(data[i][1]))
#        #features.append(tmp.unfold(0,SL*3,CL_max//2))
#        #labels.append(torch.Tensor(data[i][1]).unfold(0,SL,CL_max//2))
#    return torch.cat(features,dim=0).float(), torch.cat(labels,dim=0).float()
# -

class h5pyDataset(Dataset):
    def __init__(self, h5f, idxs, transform=None, target_transform=None):
        self.data = h5f
        self.idxs = idxs
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.idxs)
    
    def clip_datapoints(self,X, Y, CL, N_GPUS):
    # This function is necessary to make sure of the following:
    # (i) Each time model_m.fit is called, the number of datapoints is a
    # multiple of N_GPUS. Failure to ensure this often results in crashes.
    # (ii) If the required context length is less than CL_max, then
    # appropriate clipping is done below.
    # Additionally, Y is also converted to a list (the .h5 files store 
    # them as an array).

        rem = X.shape[0]%N_GPUS
        clip = (CL_max-CL)//2

        if rem != 0 and clip != 0:
            return X[:-rem, clip:-clip], [Y[t][:-rem] for t in range(1)]
        elif rem == 0 and clip != 0:
            return X[:, clip:-clip], [Y[t] for t in range(1)]
        elif rem != 0 and clip == 0:
            return X[:-rem], [Y[t][:-rem] for t in range(1)]
        else:
            return X, [Y[t] for t in range(1)]

    def __getitem__(self, idx):
        X = self.data['X' + str(self.idxs[idx])][:].astype(np.float32)
        Y = self.data['Y' + str(self.idxs[idx])][:].astype(np.float32)
        #X, Y = self.clip_datapoints(X, Y, CL, N_GPUS) 

        if self.transform:
            X = self.transform(X)
        if self.target_transform:
            label = self.target_transform(label)
        return X, [Y[t] for t in range(1)]
