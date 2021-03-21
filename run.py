import argparse
import json
import logging
import os
import random

import numpy as np
import torch

from pandas import DataFrame

from torch.utils.data import DataLoader
import torch.nn.functional as F

from model import KGEModel

from dataloader import TrainDataset, filter_relations
from dataloader import DatasetIterator

from ogb.linkproppred import LinkPropPredDataset, Evaluator
from collections import defaultdict

import time
import pdb

import datetime

def now():
    d = datetime.datetime.now()
    x = d - datetime.timedelta(microseconds=d.microsecond)
    return x

d_name = "ogbl-biokg"
dataset = LinkPropPredDataset(name = d_name) 

split_edge = dataset.get_edge_split()
train_triples, valid_triples, test_triples = split_edge["train"], split_edge["valid"], split_edge["test"]

nrelation = int(max(train_triples['relation']))+1 #4
nentity = sum(dataset[0]['num_nodes_dict'].values())

entity_dict = dict()
cur_idx = 0
for key in dataset[0]['num_nodes_dict']: #['drug', 'sideeffect', 'protein', 'disease', 'function']:
    entity_dict[key] = (cur_idx, cur_idx + dataset[0]['num_nodes_dict'][key])
    cur_idx += dataset[0]['num_nodes_dict'][key]
nentity = sum(dataset[0]['num_nodes_dict'].values()) # entity_dict['disease'][1]+1 

#pdb.set_trace()

evaluator = Evaluator(name = d_name)

args = {"cuda" : True,
        "lr" : 1e-4, 
        "n_epoch" : 5, 
        "hidden_dim" : 500, 
        "save_checkpoint_steps" : 1000, 
        "log_steps" : 50,
        "valid_steps" : 300,
        "test_log_steps" : 100}

validation_iterator = DatasetIterator(
    TrainDataset(valid_triples, nentity, nrelation, 
                 1024, 512, entity_dict, train_triples= train_triples))#, filter_idx = filter_relations(valid_triples)))
test_iterator = DatasetIterator(
    TrainDataset(test_triples, nentity, nrelation, 
                 1024, 512, entity_dict, train_triples= train_triples))#, filter_idx = filter_relations(test_triples)))
train_iterator = DatasetIterator(
    TrainDataset(train_triples, nentity, nrelation, 
                 1024, 256, entity_dict))#, filter_idx = filter_relations(train_triples)))

kge_model = KGEModel(
        model_name="QuatE",
        nentity=nentity,
        nrelation=nrelation,
        hidden_dim=args["hidden_dim"],
        evaluator=evaluator)
if args["cuda"]:
    kge_model.cuda()

learning_rate = args["lr"] #learning_rate
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, kge_model.parameters()), 
                             lr=learning_rate)

training_logs = []
valid_logs = []
for step in range(args["n_epoch"]*train_iterator.epoch_size):

    loss = kge_model.train_step(optimizer, train_iterator, args)
    training_logs.append(('train', loss))

    if step % args["save_checkpoint_steps"] == 0 and step > 0:
        torch.save({'step': step,
                    'loss': loss,
                    'model': kge_model.state_dict()}, "checkpoint_"+str(now))

    if step % args["log_steps"] == 0:
        print("step:", step, "loss:", loss)

    if step % args["valid_steps"] == 0:
        logging.info('Evaluating on Valid Dataset...')
        valid_loss, metrics = kge_model.test_step(validation_iterator, args)
        training_logs.append(('validation', valid_loss))
        valid_logs.append(metrics)

        # save progress 
        DataFrame(valid_logs).to_csv("valid_logs.csv")
        DataFrame(training_logs, columns =['type', 'loss']).to_csv("training_logs.csv")