#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch

from torch.utils.data import Dataset

from pandas import DataFrame

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
        samples = [self.sample_negative() for _ in range(n)]        
        return np.stack(samples, axis=0)
    
    def sample_negative(self):
        new = self.triples.sample(1).squeeze().to_numpy()
        idx = 0 if np.random.rand() > 0.5 else 2
        while (df == new).all(1).any():
            new = self.triples.sample(1).squeeze().to_numpy()
            other = df.sample(1).squeeze().to_numpy()
            new[idx] = other[idx]
        return new
    
class DatasetIterator(object):
    def __init__(self, dataloader):
        self.iterator = self.one_shot_iterator(dataloader)
        self.step = 0
        
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