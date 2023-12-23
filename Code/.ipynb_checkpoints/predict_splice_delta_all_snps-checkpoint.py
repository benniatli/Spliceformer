import numpy as np
import sys
import time
import h5py
from tqdm import tqdm

import numpy as np
import re
from math import ceil
from sklearn.metrics import average_precision_score

import pandas as pd
import matplotlib.pyplot as plt
import pickle
#import pickle5 as pickle

from sklearn.model_selection import train_test_split

from scipy.sparse import load_npz
from glob import glob

from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.linear_model import LogisticRegression

from torch.utils.data import Dataset
import torch
import torch.nn as nn
from transformers import get_constant_schedule_with_warmup
from sklearn.metrics import precision_score,recall_score,accuracy_score

from src.train import trainModel
from src.dataloader import get_GTEX_v8_Data,spliceDataset,h5pyDataset,getDataPointList,getDataPointListGTEX,DataPointGTEX
from src.weight_init import keras_init
from src.losses import categorical_crossentropy_2d
from src.model import SpliceFormer
from src.evaluation_metrics import print_topl_statistics
import copy
import pyfastx
import gffutils
#import tensorflow as tf

def generate_point_mutations(input_tensor, i, j,base_map,inside_gene):
    """
    Generate all point mutations within the range [i, j] for a batch of one-hot encoded genomic sequences.

    Args:
    input_tensor (torch.Tensor): Batch of one-hot encoded genomic sequences with shape (4, n).
    i (int): Start index of the mutation range.
    j (int): End index of the mutation range.

    Returns:
    torch.Tensor: Batch of point mutations with shape (b, 4, n).
    """

    # Extract batch size and sequence length
    n_in_gene = torch.sum(inside_gene[0,20000:25000]).cpu().numpy()
    #_,_, n = input_tensor[:,:,n_in_gene].shape

    # Create a copy of the input tensor to store point mutations

    # Loop through the mutation range [i, j]
    d = 0
    for batch in np.array_split(np.arange(i, i+n_in_gene),(n_in_gene)//(BATCH_SIZE//3)+1):
        ref_base = []
        alt_base = []
        idx = []
        c = 0
        

        mutated_sequence = input_tensor.clone().repeat((batch.shape[0]*3,1,1))
        for pos in batch:
            original_base = torch.argmax(input_tensor[0,:, pos], dim=0).cpu().numpy()
            possible_bases = [0,1,2,3]
            possible_bases.remove(original_base)
            for base in possible_bases:
                if inside_gene[0, pos]==1:
                    mutated_sequence[c,:, pos] = 0
                    mutated_sequence[c,base, pos] = 1
                    ref_base.append(base_map[original_base])
                    alt_base.append(base_map[base])
                    idx.append(d)
                c += 1 
            d += 1
        yield mutated_sequence,alt_base,ref_base,np.array(idx)


def getDeltas(delta,idx,inside_gene):
    acceptorDelta = delta[:,1,:]*inside_gene
    donorDelta = delta[:,2,:]*inside_gene
    pos_gain_a = torch.argmax(acceptorDelta,dim=1)
    pos_gain_d = torch.argmax(donorDelta,dim=1)
    pos_loss_a = torch.argmax(-acceptorDelta,dim=1)
    pos_loss_d = torch.argmax(-donorDelta,dim=1)

    delta_gain_a = acceptorDelta.gather(1,pos_gain_a.unsqueeze(1)).cpu().numpy()[:,0]
    delta_gain_d = donorDelta.gather(1,pos_gain_d.unsqueeze(1)).cpu().numpy()[:,0]
    delta_loss_a = -acceptorDelta.gather(1,pos_loss_a.unsqueeze(1)).cpu().numpy()[:,0]
    delta_loss_d = -donorDelta.gather(1,pos_loss_d.unsqueeze(1)).cpu().numpy()[:,0]
    pos_gain_a = pos_gain_a.cpu().numpy()
    pos_gain_d = pos_gain_d.cpu().numpy()
    pos_loss_a = pos_loss_a.cpu().numpy()
    pos_loss_d = pos_loss_d.cpu().numpy()
    return delta_gain_a,delta_loss_a,delta_gain_d,delta_loss_d,pos_gain_a-idx,pos_loss_a-idx,pos_gain_d-idx,pos_loss_d-idx

def ceil_div(x, y):

    return int(ceil(float(x)/y))


IN_MAP = np.asarray([[0, 0, 0, 0],
                     [1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])

def one_hot_encode(Xd):
    return IN_MAP[Xd.astype('int8')]

def reformat_data(X0):
    # This function converts X0, Y0 of the create_datapoints function into
    # blocks such that the data is broken down into data points where the
    # input is a sequence of length SL+CL_max corresponding to SL nucleotides
    # of interest and CL_max context nucleotides, the output is a sequence of
    # length SL corresponding to the splicing information of the nucleotides
    # of interest. The CL_max context nucleotides are such that they are
    # CL_max/2 on either side of the SL nucleotides of interest.

    num_points = ceil_div(len(X0)-CL_max, SL)
    Xd = np.zeros((num_points, SL+CL_max))
    X0 = np.pad(X0, [0, SL], 'constant', constant_values=0)

    for i in range(num_points):
        Xd[i] = X0[SL*i:CL_max+SL*(i+1)]

    return Xd

def seqToArray(seq,strand):
    seq = 'N'*(CL_max//2) + seq + 'N'*(CL_max//2)
    seq = seq.upper()
    seq = re.sub(r'[^AGTC]', '0',seq)
    seq = seq.replace('A', '1').replace('C', '2')
    seq = seq.replace('G', '3').replace('T', '4').replace('N', '0')
    if strand == '+':
        X0 = np.asarray([int(x) for x in seq])
            
    elif strand == '-':
        X0 = (5-np.asarray([int(x) for x in seq[::-1]])) % 5  # Reverse complement
        
    Xd = reformat_data(X0)
    return  np.swapaxes(one_hot_encode(Xd),1,2)

rev_comp_dict = {'A':'T','T':'A','C':'G','G':'C'}

def predictMutations(ref,strand,chrom,shift):

    batch_features = torch.Tensor(ref).type(torch.FloatTensor).unsqueeze(dim=0).to(device)

    ref_pred = ([models[i](batch_features)[0].detach() for i in range(n_models)])
    ref_pred = torch.stack(ref_pred)
    ref_pred = torch.mean(ref_pred,dim=0)
    inside_gene = (batch_features.sum(axis=(1)) >= 1)

    mutations = generate_point_mutations(batch_features, CL_max//2, CL_max//2+SL,base_map,inside_gene)
    for i,(mutation,alt_base,ref_base,idx) in tqdm(enumerate(mutations)):
        
        mutation = mutation.type(torch.FloatTensor).to(device)
            
        alt_pred = ([models[i](mutation)[0].detach() for i in range(n_models)])
        alt_pred = torch.stack(alt_pred)
        alt_pred = torch.mean(alt_pred,dim=0)

        delta = alt_pred-ref_pred
        a1,b1,c1,d1,a2,b2,c2,d2 = getDeltas(delta,CL_max//2+idx,inside_gene)

        if strand == '+':
            df = pd.DataFrame({'CHROM':chrom,'POS':start+shift+idx,'REF':ref_base,'ALT':alt_base,'DS_AG':a1,'DS_AL':b1,'DS_DG':c1,'DS_DL':d1,'DP_AG':a2,'DP_AL':b2,'DP_DG':c2,'DP_DL':d2})
        else:
            df = pd.DataFrame({'CHROM':chrom,'POS':end-shift-idx,'REF':[rev_comp_dict[x] for x in ref_base],'ALT':[rev_comp_dict[x] for x in alt_base],'DS_AG':a1,'DS_AL':b1,'DS_DG':c1,'DS_DL':d1,'DP_AG':-a2,'DP_AL':-b2,'DP_DG':-c2,'DP_DL':-d2})
        if i==0:
            results = df
        else:
            results = pd.concat([results,df],axis=0)
    return results


def main():
    rng = np.random.default_rng(23673)
    ii = int(sys.argv[1])
    L = 32
    N_GPUS = 1
    k = 1
    NUM_ACCUMULATION_STEPS=1
    # Hyper-parameters:
    # L: Number of convolution kernels
    # W: Convolution window size in each residual unit
    # AR: Atrous rate in each residual unit

    W = np.asarray([11, 11, 11, 11, 11, 11, 11, 11,
                    21, 21, 21, 21, 41, 41, 41, 41])
    AR = np.asarray([1, 1, 1, 1, 4, 4, 4, 4,
                     10, 10, 10, 10, 25, 25, 25, 25])
    BATCH_SIZE = 16*k*N_GPUS

    k = NUM_ACCUMULATION_STEPS*k

    CL = 2 * np.sum(AR*(W-1))

    SL=5000
    CL_max=40000

    from collections import OrderedDict

    temp = 1
    n_models = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_m = SpliceFormer(CL_max,bn_momentum=0.01/NUM_ACCUMULATION_STEPS,depth=4,heads=4,n_transformer_blocks=2,determenistic=True,crop=False)
    model_m.apply(keras_init)
    model_m = model_m.to(device)

    if torch.cuda.device_count() > 1:
        model_m = nn.DataParallel(model_m)

    output_class_labels = ['Null', 'Acceptor', 'Donor']

    #for output_class in [1,2]:
    models = [copy.deepcopy(model_m) for i in range(n_models)]

    for i,model in enumerate(models):
            state_dict = torch.load('../Results/PyTorch_Models/transformer_encoder_40k_finetune_rnasplice-blood_all_050623_{}'.format(i))
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)

    #[model.load_state_dict(torch.load('../Results/PyTorch_Models/transformer_encoder_40k_finetune_rnasplice-blood_all_050623_{}'.format(i))) for i,model in enumerate(models)]
    #nr = [0,2,3]
    #[model.load_state_dict(torch.load('../Results/PyTorch_Models/transformer_encoder_40k_201221_{}'.format(nr[i]))) for i,model in enumerate(models)]
    #chunkSize = num_idx/10
    for model in models:
        model.eval()

    Y_true_acceptor, Y_pred_acceptor = [],[]
    Y_true_donor, Y_pred_donor = [],[]

    #targets_list = []
    #outputs_list = []
    ce_2d = []
    gtf = gffutils.FeatureDB("/odinn/tmp/benediktj/Data/Gencode_V44/gencode.v44.annotation.db")

    fasta = pyfastx.Fasta('/odinn/tmp/benediktj/Data/Gencode_V44/GRCh38.p14.genome.fa')

    df = []
    for gene in tqdm(gtf.features_of_type('gene')):
        if gene['gene_type'][0] == "protein_coding" and gene[0] != 'chrM':
            df.append([gene[0],gene[3],gene[4],gene[6],gene['gene_name'][0],gene['gene_id'][0]])
        #print(gene)
          #  print()

    df = pd.DataFrame(df)
    df.columns = ['CHROM','START','END','STRAND','NAME','ID']

    df = np.array_split(df,30)[ii]



    base_map = 'ACGT'
    for i in range(df.shape[0]):
        gene_name,chrom,strand,start,end,gene_id = df.iloc[i,:][['NAME','CHROM','STRAND','START','END','ID']]
        ref = fasta[chrom][start-1:end].seq
        ref = seqToArray(ref,strand)
        for j in range(ref.shape[0]):
            tmp = predictMutations(ref[j,:,:],strand,chrom,j*SL)
            if j==0:
                results = tmp
            else:
                results = pd.concat([results,tmp],axis=0)
        results.sort_values(['CHROM','POS','REF','ALT']).to_csv('/odinn/tmp/benediktj/Data/splice_variant_delta_scores/raw_snps/{}_{}_{}.tsv'.format(chrom,gene_name,gene_id),sep='\t',index=False)


if __name__ == "__main__":
    main()
