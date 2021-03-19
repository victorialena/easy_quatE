from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from dataloader import TestDataset
from collections import defaultdict

from ogb.linkproppred import Evaluator

class KGEModel(nn.Module):
    def __init__(self, model_name, nentity, nrelation, hidden_dim, evaluator):
        super(KGEModel, self).__init__()
        
        self.model_name = 'QuatE' # add comparison : 'TransE', 'DistMult', 'ComplEx', 'RotatE'
        
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        
        # In general, the embedding size k is tuned amongst {50, 100, 200, 250, 300}.
        # Regularization rate λ1 and λ2 are searched in {0, 0.01, 0.05, 0.1, 0.2}.
        
        self.lambda_r = 0.05 
        self.lambda_e = 0.01        
        self.embedding_range = 1. / hidden_dim
        
        self.entity_dim = hidden_dim*4 
        self.relation_dim = hidden_dim*4
        
        self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim))
        nn.init.uniform_(
            tensor=self.entity_embedding, 
            a=-self.embedding_range, 
            b=self.embedding_range)
        
        self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
        nn.init.uniform_(
            tensor=self.relation_embedding, 
            a=-self.embedding_range, 
            b=self.embedding_range)
        
        self.evaluator = evaluator
        
    def forward(self, sample, Y = torch.ones(batch_size,)):
        '''
        sample : a batch of triple.
        IDEA : should we use negative samples?
        '''

        batch_size, negative_sample_size = sample.size(0), torch.sum(Y == -1).item()

        head = torch.index_select(
            self.entity_embedding, 
            dim=0, 
            index=sample[:,0]
        ).unsqueeze(1)

        relation = torch.index_select(
            self.relation_embedding, 
            dim=0, 
            index=sample[:,1]
        ).unsqueeze(1)

        tail = torch.index_select(
            self.entity_embedding, 
            dim=0, 
            index=sample[:,2]
        ).unsqueeze(1)
        
        score = QuatE(head, relation, tail, Y)
        
        return score
    
    def hamilton_product(A, B):
        assert A.shape == B.shape
        
        n, k, d = A.shape
        assert d == 4
        
        p = torch.sum(torch.mul(torch.mul(A, B), [1, -1, -1, -1]), dim=2)
        q = torch.sum(torch.mul(torch.mul(A, B[:, :, [1, 0, 3, 2]]), [1, 1, 1, -1]), dim=2)
        u = torch.sum(torch.mul(torch.mul(A, B[:, :, [2, 3, 0, 1]]), [1, -1, 1, 1]), dim=2)
        v = torch.sum(torch.mul(torch.mul(A, B[:, :, [3, 2, 1, 0]]), [1, 1, -1, 1]), dim=2)
        
        return torch.stack([p,q,u,v], dim=2)
    
    def QuatE(self, head, relation, tail, Y):
        '''
        head : head node embedding (N, k*4)
        relation : relation embedding (N, k*4)
        tail : tail node embedding (N, k*4)
        Y : (N, ) 1 if positive sample, -1 is negative sample
        '''
                
        #p_head, q_head, u_head, v_head = torch.chunk(head, 4, dim=2)
        #p_tail, q_tail, u_tail, v_tail = torch.chunk(tail, 4, dim=2)
        
        head = torch.reshape(head, (self.nentity, self.hidden_dim, 4))
        tail = torch.reshape(tail, (self.nentity, self.hidden_dim, 4))
        relation = torch.reshape(relation, (self.nrelation, self.hidden_dim, 4))
        
        # normalize W_r
        normalized_relation = relation / torch.norm(relation, dim=2)
        
        headp = hamilton_product(head, normalized_relation)
        phi = torch.sum(torch.mul(headp, tail), dim=(2,1)) # (N,)
        
        loss = torch.sum(torch.log(1+torch.exp(-Y*phi)))
        loss = loss + self.lambda_e * torch.norm(self.entity_embedding) + self.lambda_r * torch.norm(self.relation_embedding)
        
        return loss

    # @staticmethod
    def train_step(model, optimizer, train_iterator, args):
        '''
        SINGLE training step. Evaluate back-propation and return the loss
        '''

        model.train()
        optimizer.zero_grad()
        samples, Y = next(train_iterator)

        if args.cuda:
            samples = samples.cuda()
            Y = Y.cuda()
        
        loss = model(samples, Y)
        loss.backward()
        optimizer.step()
        
        return loss