#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch

from torch.utils.data import Dataset

from pandas import DataFrame
import pdb

def filter_relations(triples, verbose = True):
    drug_sideeffect = np.stack([np.array(triples['head_type']) == 'drug',
                            np.array(triples['tail_type'])=='sideeffect']).all(0)
    drug_disease = np.stack([np.array(triples['head_type'])=='drug',
                         np.array(triples['tail_type'])=='disease']).all(0)
    drug_protein = np.stack([np.array(triples['head_type'])=='drug',
                         np.array(triples['tail_type'])=='protein']).all(0)
    disease_protein = np.stack([np.array(triples['head_type'])=='disease',
                         np.array(triples['tail_type'])=='protein']).all(0)
    idx = np.stack([drug_disease, drug_sideeffect, drug_protein, disease_protein]).any(0)
    if verbose:
        print("filtering relation types ", np.unique(triples['relation'][idx]))
    return idx

def isin(a, dset):
    for i in range(len(a)):
        dset = dset[a[i] == dset[:, i]]
    return len(dset) > 0 

class TrainDataset(Dataset):
    def __init__(self, triples, nentity, nrelation, 
                 batch_size, negative_sample_size, 
                 entity_dict,
                 filter_idx = None, 
                 train_triples = None,
                 shuffle = True):
        self.len = len(triples['head'])
        
        ht = [entity_dict[t][0] for t in triples['head_type']]
        tt = [entity_dict[t][0] for t in triples['tail_type']]
        self.triples = DataFrame({"head": triples['head']+ht,
                                  "relation": triples['relation'],
                                  "tail": triples['tail']+tt}).to_numpy()
        
        if filter_idx is not None:
            self.triples = self.triples[filter_idx]
            print(np.unique(self.triples[:, 1]))
            self.len = len(self.triples)
        if shuffle:
            np.random.shuffle(self.triples)
            
        self.train_triples = None
        if train_triples is not None:
            ht = [entity_dict[t][0] for t in train_triples['head_type']]
            tt = [entity_dict[t][0] for t in train_triples['tail_type']]
            self.train_triples = np.vstack([train_triples['head']+ht,train_triples['relation'], train_triples['tail']+tt]).T
        
        self.nentity = nentity
        self.nrelation = nrelation
        self.entity_dict = entity_dict

        self.batch_size = batch_size        
        self.negative_sample_size = negative_sample_size
        self.positive_sample_size = batch_size - negative_sample_size
        
        self.n_batch = int(self.len/self.positive_sample_size) -1
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        idx = idx%self.n_batch
        s = idx*self.positive_sample_size
        e = (idx+1)*self.positive_sample_size
            
        samples = self.triples[s:e] 
        Y = np.ones(self.positive_sample_size,)
        
        if self.negative_sample_size > 0:
            negative_sample = self.get_n_negative_samples(self.negative_sample_size)
            samples = np.vstack([samples, negative_sample])
            Y = np.append(Y, -np.ones(self.negative_sample_size,))
        
        return torch.tensor(samples), torch.tensor(Y)
    
    def get_n_negative_samples(self, n):
        samples = self.triples[np.random.randint(self.triples.shape[0], size=n)]
        data = self.triples
        if self.train_triples is not None:
            data = np.append(data, self.train_triples, axis=0)

        idx = 0 if np.random.rand() > 0.5 else 2
        samples[:, idx] = np.random.randint(self.nentity, size=n)
        y = np.array([isin(x, data) for x in samples])
        while any(y):
            samples[y==1, idx] = np.random.randint(self.nentity, size=sum(y))
            y[y==1] = np.array([isin(x, data) for x in samples[y==1]])
        return samples
    
class DatasetIterator(object):
    def __init__(self, dataloader):
        self.iterator = self.one_shot_iterator(dataloader)
        self.step = 0
        self.epoch_size = int(dataloader.len / dataloader.batch_size)
        
    def __next__(self):
        self.step += 1
        data = next(self.iterator)
        return data
    
    @staticmethod
    def one_shot_iterator(dataloader):
        '''
        Transform a PyTorch Dataloader into python iterator
        '''
        while True:
            for data in dataloader:
                yield data

# eof