#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import os
import logging

from torch_geometric.data import Data

from utils.utils import *

def get_train_and_test_cpi(args):

    cpi_list = pd.read_csv(os.path.join(args.data_path,'CPI_with_negative_sample.csv')).to_numpy()
    n = len(cpi_list)
    # if args.do_predict:
    #     train_cpi = cpi_list
    #     valid_cpi = cpi_list[int(len(cpi_list)*0.99):]
    #     test_cpi = np.array([])
    if args.do_predict:
        train_cpi,valid_cpi = train_test_split(cpi_list,train_size=0.99,shuffle=False)
        test_cpi = np.array([])
    else:
        test_start_idx = args.fold * n//10
        test_end_idx = (args.fold+1) * n//10
        train_cpi = np.concatenate((cpi_list[0:test_start_idx,],cpi_list[test_end_idx:,]))
        test_cpi = cpi_list[test_start_idx:test_end_idx,]
        train_cpi,valid_cpi = train_test_split(train_cpi,train_size=0.99,shuffle=False)

    compound_idx = train_cpi[:,0]
    protein_idx = train_cpi[:,1]
    values = train_cpi[:,2]
    CPI_train = np.zeros((args.n_compound,args.n_protein))
    CPI_train[compound_idx,protein_idx] = values
    CPI_train_mask = np.zeros((args.n_compound,args.n_protein))
    CPI_train_mask[compound_idx,protein_idx] = 1

    logging.info(f"train_triples:{len(train_cpi)}")
    logging.info(f"valid_triples:{len(valid_cpi)}")
    logging.info(f"test_triples:{len(test_cpi)}")
    return CPI_train,CPI_train_mask,valid_cpi, test_cpi

def load_kge_data(args):
    with open(os.path.join(args.data_path, 'entities.dict')) as fin:
        entity2id = dict()
        for line in fin:
            eid, entity = line.strip().split('\t')
            entity2id[entity] = int(eid)

    with open(os.path.join(args.data_path, 'relations.dict')) as fin:
        relation2id = dict()
        for line in fin:
            rid, relation = line.strip().split('\t')
            relation2id[relation] = int(rid)
    
    nentity = len(entity2id)
    nrelation = len(relation2id)

    CPI_train,_, _, _ = get_train_and_test_cpi(args)

    CCS = np.load(os.path.join(args.data_path,'CCS.npy'))
    PPS = np.load(os.path.join(args.data_path,'PPS.npy'))
    se = np.load(os.path.join(args.data_path,'compound_se.npy'))

    cpi_train_triples = adjacency_matrix_to_triples(CPI_train,relation2id['CPI'],col_offset=args.n_compound)
    ccs_triples = adjacency_matrix_to_triples(CCS,relation2id['CCS'],args.ccs_threshold)
    pps_triples = adjacency_matrix_to_triples(PPS,relation2id['PPS'],args.pps_threshold,args.n_compound,args.n_compound)
    se_triples = adjacency_matrix_to_triples(se,relation2id['side_effect'],col_offset=args.n_compound+args.n_protein)

    train_triples = np.concatenate([ccs_triples, pps_triples, se_triples, cpi_train_triples])

    logging.info('#train: %d' % len(train_triples))

    train_dataloader_head = DataLoader(
        TrainDataset(train_triples, nentity, nrelation, args.kg_negative_sample_size, 'head-batch'), 
        batch_size=args.kg_batch_size,
        shuffle=True, 
        collate_fn=TrainDataset.collate_fn
    )
    
    train_dataloader_tail = DataLoader(
        TrainDataset(train_triples, nentity, nrelation, args.kg_negative_sample_size, 'tail-batch'), 
        batch_size=args.kg_batch_size,
        shuffle=True, 
        collate_fn=TrainDataset.collate_fn
    )
    
    train_iterator = BidirectionalOneShotIterator(train_dataloader_head, train_dataloader_tail)
    return train_iterator, nentity, nrelation


def load_graph_data(args):
    CCS = np.load(os.path.join(args.data_path,'CCS.npy'))
    PPS = np.load(os.path.join(args.data_path,'PPS.npy'))

    CPI_train,CPI_train_mask, _, _ = get_train_and_test_cpi(args)

    ccs_edge_index, ccs_edge_weight = adjacency_matrix_to_edge_index_and_weight(CCS,args.pna_ccs_threshold)
    pps_edge_index, pps_edge_weight = adjacency_matrix_to_edge_index_and_weight(PPS,args.pna_pps_threshold,args.n_compound,args.n_compound)
    cpi_edge_index,cpi_edge_weight = adjacency_matrix_to_edge_index_and_weight(CPI_train,col_offset=args.n_compound)
    pci_edge_index,pci_edge_weight = adjacency_matrix_to_edge_index_and_weight(CPI_train.T,row_offset=args.n_compound)

    graph_data = Data(num_nodes=args.n_compound+args.n_protein)

    graph_data.edge_index = torch.tensor(np.concatenate([ccs_edge_index,pps_edge_index,cpi_edge_index,pci_edge_index],axis=1)).long()
    graph_data.edge_weight = torch.tensor(np.concatenate([ccs_edge_weight,pps_edge_weight,cpi_edge_weight,pci_edge_weight]).reshape(-1,1)).float()
    graph_data.edge_type = torch.tensor([0]*len(ccs_edge_weight)+[1]*len(pps_edge_weight)+[2]*len(cpi_edge_weight)+[3]*len(pci_edge_weight),dtype=torch.long)
    graph_data.deg = calc_deg(graph_data)

    return graph_data,CPI_train,CPI_train_mask

class TrainDataset(Dataset):
    def __init__(self, triples, nentity, nrelation, negative_sample_size, mode):
        self.len = len(triples)
        self.triples = triples
        self.nentity = nentity
        self.nrelation = nrelation
        self.negative_sample_size = negative_sample_size
        self.mode = mode
        self.count = self.count_frequency(triples)
        self.true_head, self.true_tail = self.get_true_head_and_tail(self.triples)
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        positive_sample = self.triples[idx]

        head, relation, tail = positive_sample

        subsampling_weight = self.count[(head, relation)] + self.count[(tail, -relation-1)]
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))
        
        negative_sample_list = []
        negative_sample_size = 0

        while negative_sample_size < self.negative_sample_size:
            negative_sample = np.random.randint(self.nentity, size=self.negative_sample_size*2)
            if self.mode == 'head-batch':
                mask = np.in1d(
                    negative_sample, 
                    self.true_head[(relation, tail)], 
                    assume_unique=True, 
                    invert=True
                )
            elif self.mode == 'tail-batch':
                mask = np.in1d(
                    negative_sample, 
                    self.true_tail[(head, relation)], 
                    assume_unique=True, 
                    invert=True
                )
            else:
                raise ValueError('Training batch mode %s not supported' % self.mode)
            negative_sample = negative_sample[mask]
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size
        
        negative_sample = np.concatenate(negative_sample_list)[:self.negative_sample_size]

        negative_sample = torch.LongTensor(negative_sample)

        positive_sample = torch.LongTensor(positive_sample)
            
        return positive_sample, negative_sample, subsampling_weight, self.mode
    
    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        subsample_weight = torch.cat([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return positive_sample, negative_sample, subsample_weight, mode
    
    @staticmethod
    def count_frequency(triples, start=4):
        '''
        Get frequency of a partial triple like (head, relation) or (relation, tail)
        The frequency will be used for subsampling like word2vec
        '''
        count = {}
        for head, relation, tail in triples:
            if (head, relation) not in count:
                count[(head, relation)] = start
            else:
                count[(head, relation)] += 1

            if (tail, -relation-1) not in count:
                count[(tail, -relation-1)] = start
            else:
                count[(tail, -relation-1)] += 1
        return count
    
    @staticmethod
    def get_true_head_and_tail(triples):
        '''
        Build a dictionary of true triples that will
        be used to filter these true triples for negative sampling
        '''
        
        true_head = {}
        true_tail = {}

        for head, relation, tail in triples:
            if (head, relation) not in true_tail:
                true_tail[(head, relation)] = []
            true_tail[(head, relation)].append(tail)
            if (relation, tail) not in true_head:
                true_head[(relation, tail)] = []
            true_head[(relation, tail)].append(head)

        for relation, tail in true_head:
            true_head[(relation, tail)] = np.array(list(set(true_head[(relation, tail)])))
        for head, relation in true_tail:
            true_tail[(head, relation)] = np.array(list(set(true_tail[(head, relation)])))                 

        return true_head, true_tail

   
class BidirectionalOneShotIterator(object):
    def __init__(self, dataloader_head, dataloader_tail):
        self.iterator_head = self.one_shot_iterator(dataloader_head)
        self.iterator_tail = self.one_shot_iterator(dataloader_tail)
        self.step = 0
        
    def __next__(self):
        self.step += 1
        if self.step % 2 == 0:
            data = next(self.iterator_head)
        else:
            data = next(self.iterator_tail)
        return data
    
    @staticmethod
    def one_shot_iterator(dataloader):
        '''
        Transform a PyTorch Dataloader into python iterator
        '''
        while True:
            for data in dataloader:
                yield data

