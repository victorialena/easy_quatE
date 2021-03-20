#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch

from torch.utils.data import Dataset

from pandas import DataFrame

# def isin(a, dset, i=0):
#     if len(dset) == 0:
#         return False
#     if i == len(a)-1: #last idx
#         return any(a[i] == dset[:, i])
#     return isin(a, dset[a[i] == dset[:, i]] ,i+1)

def isin(a, dset):
    for i in range(len(a)):
        dset = dset[a[i] == dset[:, i]]
    return len(dset) > 0 

class TrainDataset(Dataset):
    def __init__(self, triples, nentity, nrelation, 
                 batch_size, negative_sample_size, 
                 entity_dict,
                 shuffle = True):
        self.len = len(triples['head'])
        
        ht = [entity_dict[t][0] for t in triples['head_type']]
        tt = [entity_dict[t][0] for t in triples['tail_type']]
        self.triples = DataFrame({"head": triples['head']+ht,
                                  "relation": triples['relation'], 
                                  "tail": triples['tail']+tt})
        
        if shuffle:
            self.triples.sample(frac=1).reset_index(drop=True)
        
        self.nentity = nentity
        self.nrelation = nrelation
        self.entity_dict = entity_dict

        self.batch_size = batch_size        
        self.negative_sample_size = negative_sample_size
        self.positive_sample_size = batch_size - negative_sample_size
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        s = idx*self.positive_sample_size
        e = (idx+1)*self.positive_sample_size-1
        samples = self.triples.loc[s:e].squeeze().to_numpy()
        Y = np.ones(self.positive_sample_size,)
        
        if self.negative_sample_size > 0:
            negative_sample = self.get_n_negative_samples(self.negative_sample_size)
            samples = np.vstack([samples, negative_sample])
            Y = np.append(Y, -np.ones(self.negative_sample_size,))
        
        return torch.tensor(samples), torch.tensor(Y)
    
    def get_n_negative_samples(self, n):
        data = self.triples.to_numpy()
        samples = data[np.random.randint(data.shape[0], size=n), :]
        idx = 0 if np.random.rand() > 0.5 else 2
        samples[:, idx] = np.random.randint(self.nentity, size=n)
        y = np.array([isin(x, data) for x in samples])
        while any(y):
            samples[y==1, idx] = np.random.randint(self.nentity, size=sum(y))
            y[y==1] = np.array([isin(x, data) for x in samples[y==1]])
        return samples
    
class TestDataset(Dataset):
    def __init__(self, triples, nentity, nrelation, 
                 batch_size, entity_dict,shuffle = True):
        self.len = len(triples['head'])
        
        ht = [entity_dict[t][0] for t in triples['head_type']]
        tt = [entity_dict[t][0] for t in triples['tail_type']]
        self.triples = DataFrame({"head": triples['head']+ht,
                                  "relation": triples['relation'], 
                                  "tail": triples['tail']+tt})
        
        if shuffle:
            self.triples.sample(frac=1).reset_index(drop=True)
        
        self.nentity = nentity
        self.nrelation = nrelation
        self.entity_dict = entity_dict

        self.batch_size = 2*batch_size        
        self.negative_sample_size = batch_size
        self.positive_sample_size = batch_size
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        s = idx*self.positive_sample_size
        e = (idx+1)*self.positive_sample_size-1
        samples = self.triples.loc[s:e].squeeze().to_numpy()
        Y = np.ones(self.positive_sample_size,)
        
        negative_sample = self.get_negative_samples(samples)
        samples = np.vstack([samples, negative_sample])
        Y = np.append(Y, -np.ones(self.negative_sample_size,))
        
        return torch.tensor(samples), torch.tensor(Y)
    
    def get_negative_samples(self, samples):
        data = self.triples.to_numpy()        
        idx = 0 if np.random.rand() > 0.5 else 2 # head vs tail samples
        print(samples[0:3])
        samples[:, idx] = np.random.randint(self.nentity, size=self.negative_sample_size)
        print(samples[0:3])
        y = np.array([isin(x, data) for x in samples])
        while any(y):
            print(sum(y))
            samples[y==1, idx] = np.random.randint(self.nentity, size=sum(y))
            print(samples[0:3])
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