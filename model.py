from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from dataloader import TrainDataset
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
        
    def forward(self, sample, Y):
        '''
        sample : a batch of triple.
        IDEA : should we use negative samples?
        '''

        batch_size, negative_sample_size = sample.shape[0], torch.sum(Y == -1).item()

        head = torch.index_select(
            self.entity_embedding, 
            dim=0, 
            index=sample[:,0])#.unsqueeze(1)
       

        relation = torch.index_select(
            self.relation_embedding, 
            dim=0, 
            index=sample[:,1])#.unsqueeze(1)

        tail = torch.index_select(
            self.entity_embedding, 
            dim=0, 
            index=sample[:,2])#.unsqueeze(1)
        
        return self.QuatE(head, relation, tail, Y)
    
    def hamilton_product(self, A, B):
        assert A.shape == B.shape
        
        n, k, d = A.shape
        assert d == 4
        
        p = torch.sum(torch.mul(torch.mul(A, B), torch.tensor([1, -1, -1, -1])), dim=2)
        q = torch.sum(torch.mul(torch.mul(A, B[:, :, [1, 0, 3, 2]]), torch.tensor([1, 1, 1, -1])), dim=2)
        u = torch.sum(torch.mul(torch.mul(A, B[:, :, [2, 3, 0, 1]]), torch.tensor([1, -1, 1, 1])), dim=2)
        v = torch.sum(torch.mul(torch.mul(A, B[:, :, [3, 2, 1, 0]]), torch.tensor([1, 1, -1, 1])), dim=2)
        
        return torch.stack([p,q,u,v], dim=2)
    
    def QuatE(self, head, relation, tail, Y):
        '''
        head : head node embedding (N, k*4)
        relation : relation embedding (N, k*4)
        tail : tail node embedding (N, k*4)
        Y : (N, ) 1 if positive sample, -1 is negative sample
        '''
        
        batch_size = head.shape[0]
        
        head = torch.reshape(head, (batch_size, self.hidden_dim, 4))
        tail = torch.reshape(tail, (batch_size, self.hidden_dim, 4))
        relation = torch.reshape(relation, (batch_size, self.hidden_dim, 4))
        
        # normalize W_r
        normalized_relation = F.normalize(relation, p=2, dim=2)
        
        headp = self.hamilton_product(head, normalized_relation)
        phi = torch.sum(torch.mul(headp, tail), dim=(2,1)) # (N,)
        
        loss = torch.log(1+torch.exp(-Y*phi))
        
        return loss, Y

    # @staticmethod
    def train_step(model, optimizer, train_iterator, args):
        '''
        SINGLE training step. Evaluate back-propation and return the loss
        '''

        model.train()
        optimizer.zero_grad()
        samples, Y = next(train_iterator)

        if args["cuda"]: # args.cuda:
            samples = samples.cuda()
            Y = Y.cuda()
        
        scores, _ = model(samples, Y)
        loss = torch.mean(scores) + model.lambda_e * torch.norm(model.entity_embedding) + model.lambda_r * torch.norm(model.relation_embedding)
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    def test_step(model, data_iterator, args):
        '''
        Evaluate the model on test or valid datasets
        '''

        model.eval()
        test_logs = defaultdict(list)

        with torch.no_grad():
            for step in range(data_iterator.epoch_size):
                samples, Y = next(data_iterator)
                if args["cuda"]:
                    samples = samples.cuda()
                    Y = Y.cuda()

                batch_size = samples.shape[0]
                scores, _ = model(samples, Y)
                loss = torch.mean(scores).item()

                batch_results = model.evaluator.eval({'y_pred_pos': scores[Y==1], 'y_pred_neg': scores[Y==-1].unsqueeze(1)})
                
                for metric in batch_results:
                    test_logs[metric].append(batch_results[metric])
                
                if step % args["test_log_steps"] == 0:
                    print('Evaluating the model... ', step, ',', data_iterator.epoch_size)
                    print('loss = ', loss)
                    
            metrics = {}
            for metric in test_logs:
                metrics[metric] = torch.cat(test_logs[metric]).mean().item()

        return loss, metrics