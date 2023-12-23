import numpy as np
import collections
from tqdm import tqdm
import pandas as pd
import io
import os
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import pickle
import re
import sys
import h5py
from math import ceil
from collections import defaultdict
from scipy.sparse import lil_matrix,csr_matrix,coo_matrix,dok_matrix, save_npz
import pyfastx
import gffutils

def create_datapoints(seq, strand, tx_start, tx_end):
    # This function first converts the sequence into an integer array, where
    # A, C, G, T, Missing are mapped to 1, 2, 3, 4, 5 respectively. If the strand is
    # negative, then reverse complementing is done. . It then calls reformat_data and one_hot_encode

    seq = seq.upper()
    seq = re.sub(r'[^AGTC]', '5',seq)
    seq = seq.replace('A', '1').replace('C', '2')
    seq = seq.replace('G', '3').replace('T', '4')

    tx_start = int(tx_start)
    tx_end = int(tx_end) 

    Y_idx = []
    
    X0 = np.asarray([int(x) for x in seq])

    X = one_hot_encode(X0)

    return X

def ceil_div(x, y):
    return int(ceil(float(x)/y))


IN_MAP = np.asarray([[0, 0, 0, 0,0],
                     [1, 0, 0, 0,0],
                     [0, 1, 0, 0,0],
                     [0, 0, 1, 0,0],
                     [0, 0, 0, 1,0],
                    [0, 0, 0, 0,1]])
# One-hot encoding of the inputs: 0 is for padding, and 1, 2, 3, 4 correspond
# to A, C, G, T, Missing respectively.

OUT_MAP = np.asarray([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1],
                      [0, 0, 0]])

def one_hot_encode(Xd):
    return IN_MAP[Xd.astype('int8')]

def getJunctions(gtf,transcript_id):
    transcript = gtf[transcript_id.split('.')[0]]
    strand = transcript[6]
    exon_junctions = []
    tx_start = int(transcript[3])
    tx_end = int(transcript[4])
    exons = gtf.children(transcript, featuretype="exon")
    for exon in exons:
        exon_start = int(exon[3])
        exon_end = int(exon[4])
        exon_junctions.append((exon_start,exon_end))

    intron_junctions = []

    if strand=='+':
        intron_start = exon_junctions[0][1]
        for i,exon_junction in enumerate(exon_junctions[1:]):
            intron_end = exon_junction[0]
            intron_junctions.append((intron_start,intron_end))
            if i+1 != len(exon_junctions[1:]):
                intron_start = exon_junction[1]

    elif strand=='-':
        exon_junctions.reverse()
        intron_start = exon_junctions[0][1]
        for i,exon_junction in enumerate(exon_junctions[1:]):
            intron_end = exon_junction[0]
            intron_junctions.append((intron_start,intron_end))
            if i+1 != len(exon_junctions[1:]):
                intron_start = exon_junction[1]

    jn_start = [x[0] for x in intron_junctions]
    jn_end = [x[1] for x in intron_junctions]
    Y_type, Y_idx = [],[]
    if strand == '+':
        Y0 = -np.ones(tx_end-tx_start+1)
        if len(jn_start) > 0:
            Y0 = np.zeros(tx_end-tx_start+1)
            for c in jn_start:
                if tx_start <= c <= tx_end:
                    Y_type.append(2)
                    Y_idx.append(c-tx_start)
            for c in jn_end:
                if tx_start <= c <= tx_end:
                    Y_type.append(1)
                    Y_idx.append(c-tx_start)

    elif strand == '-':
        Y0 = -np.ones(tx_end-tx_start+1)

        if len(jn_start) > 0:
            Y0 = np.zeros(tx_end-tx_start+1)
            for c in jn_end:
                if tx_start <= c <= tx_end:
                    Y_type.append(2)
                    Y_idx.append(tx_end-c)
            for c in jn_start:
                if tx_start <= c <= tx_end:
                    Y_type.append(1)
                    Y_idx.append(tx_end-c)

    return jn_start,jn_end,Y_type, Y_idx


def createDataset(setType,data_dir):
    genes = gtf.features_of_type('gene')

    if setType == 'train':
            CHROM_GROUP = ['chr11', 'chr13', 'chr15', 'chr17', 'chr19', 'chr21',
                           'chr2', 'chr4', 'chr6', 'chr8', 'chr10', 'chr12',
                           'chr14', 'chr16', 'chr18', 'chr20', 'chr22', 'chrX', 'chrY']
    elif setType == 'test':
        CHROM_GROUP = ['chr1', 'chr3', 'chr5', 'chr7', 'chr9']
    else:
        CHROM_GROUP = ['chr1', 'chr3', 'chr5', 'chr7', 'chr9',
                       'chr11', 'chr13', 'chr15', 'chr17', 'chr19', 'chr21',
                       'chr2', 'chr4', 'chr6', 'chr8', 'chr10', 'chr12',
                       'chr14', 'chr16', 'chr18', 'chr20', 'chr22', 'chrX', 'chrY']

    idx = 0
    
    prev_start = None
    prev_end = None
    prev_chrom = None
    
    seqData = {}
    geneToLabel = {}
    transcriptToLabel = {}
    
    for chrom in CHROM_GROUP:
        seqData[chrom] = dok_matrix((len(fasta[chrom]), 5), dtype=np.int8)
    
    if os.path.exists('{}/annotation_ensembl_v87_{}.txt'.format(data_dir,setType)):
        os.remove('{}/annotation_ensembl_v87_{}.txt'.format(data_dir,setType))

    for gene in tqdm(genes): 
        chrom = 'chr' + gene[0]
        
        if chrom not in CHROM_GROUP:
            continue
        current_chrom = chrom
        if prev_chrom is None:
            prev_chrom = current_chrom
        strand = gene[6]
        gene_start = gene[3]
        gene_end = gene[4]
        transcripts = gtf.children(gene, featuretype="transcript")
        for transcript in transcripts:
            transcript_id = transcript['transcript_id'][0]
            if transcript['transcript_biotype'][0]!='protein_coding':
                continue
            if transcript['transcript_support_level'][0] not in ['1']:
                continue

            jn_start,jn_end,Y_type, Y_idx = getJunctions(gtf,transcript_id)
            
            tx_start = int(transcript[3])
            tx_end = int(transcript[4])

            if (gene_start!=prev_start and gene_end!=prev_end):
                try:
                    seq = fasta[chrom][int(gene_start)-1:int(gene_end)]
                    seq = seq.seq
                except:
                    print('Failed reading fasta file for {}:{}-{}'.format(chrom,gene_start,gene_end))
                    print('SKIPPING')
                    break
                X = create_datapoints(seq, strand, gene_start, gene_end)

                seqData[chrom][int(gene_start)-1:int(gene_end)] = X
                prev_start,prev_end = gene_start,gene_end               

            transcriptToLabel[transcript_id] = (Y_type, Y_idx)
            
            if prev_chrom != current_chrom:
                save_npz('{}/sparse_sequence_data/{}_{}.npz'.format(data_dir,prev_chrom,setType), seqData[prev_chrom].tocoo())
                seqData[prev_chrom] = None

            prev_chrom = current_chrom

            name = '{}---{}.{}---{}.{}---{}'.format(gene['gene_name'][0],gene['gene_id'][0],gene['gene_version'][0],transcript['transcript_id'][0],transcript['transcript_version'][0],transcript['transcript_biotype'][0])
            
            if strand=='+':
                with open('{}/annotation_ensembl_v87_{}.txt'.format(data_dir,setType), 'a') as the_file:
                    the_file.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(name,chrom,strand,tx_start,tx_end,','.join([str(x) for x in jn_start]),','.join([str(x) for x in jn_end])))
            if strand=='-':
                with open('{}/annotation_ensembl_v87_{}.txt'.format(data_dir,setType), 'a') as the_file:
                    the_file.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(name,chrom,strand,tx_start,tx_end,','.join([str(x) for x in jn_end]),','.join([str(x) for x in jn_start])))

                    
    save_npz('{}/sparse_sequence_data/{}_{}.npz'.format(data_dir,prev_chrom,setType), seqData[prev_chrom].tocoo())

    with open('{}/sparse_discrete_label_data_{}.pickle'.format(data_dir,setType), 'wb') as handle:
        pickle.dump(transcriptToLabel, handle, protocol=pickle.HIGHEST_PROTOCOL)
