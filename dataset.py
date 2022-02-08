# -*- coding: utf-8 -*-
"""
Created on Sat May 22 20:03:31 2021

@author: user
"""
import torch
from torch_geometric.data import Data, DataLoader
import torch_geometric.nn as pyg_nn
import os
import numpy as np
from utils_dag import *
from queue import Queue
from torch_geometric.utils import degree
import copy

def prepare_dataset_subsegmental_hg_pred(path,node_weight=1, train=False,test=False):
    file = os.listdir(path)
    file.sort()
    num = len(file)//5
    dataset = []
    
    pred_path = ""
    
    for i in range(num):
        edge = np.load(os.path.join(path, file[i*5]))
        edge_prop = np.load(os.path.join(path, file[i*5+1]))
        x = np.load(os.path.join(path, file[i*5+3]))
        y = np.load(os.path.join(path, file[i*5+4]))
        patient = file[i*5].split('.')[0][:-5]
        edge = edge[:,edge_prop>0]
        
        if test:
            pred = np.load(os.path.join(pred_path, file[i*5+2]))
        else:
            pred = np.load(os.path.join(path, file[i*5+2]))
        
        parse_num = 10
        weight = np.ones(y.shape)
        if train:
            pred, weight = data_augmentation_pred_sub(pred, weight, edge, node_weight)
            parse_num = parse_num + np.random.randint(low = -2,high = 3)
        
        parent_map, children_map = parent_children_map(edge, x.shape[0])
        hypertree = hyper_tree(parent_map, children_map)
        hypergraph = hypergraph_airwaytree_sub(hypertree, children_map, pred, num=parse_num)
        edge_hg = hypergraph_edge_detection(hypergraph) 
        hypergraph = (torch.from_numpy(hypergraph)).float()
        edge_hg = (torch.from_numpy(edge_hg)).long()
        
        edge_prop = (torch.from_numpy(edge_prop)).float()
        edge_index = (torch.from_numpy(edge)).long()
        weight = (torch.from_numpy(weight)).float()
        
        pred111 = np.zeros((pred.shape[0],5))
        pred111[:,0] = pred//16
        pred = pred-pred111[:,1]*16
        pred111[:,1] = pred//8
        pred = pred-pred111[:,1]*8
        pred111[:,2] = pred//4
        pred = pred-pred111[:,1]*4
        pred111[:,3] = pred//2
        pred = pred-pred111[:,1]*2
        pred111[:,4] = pred//1
       
        x = np.concatenate([x, pred111], axis=1)
        
        x = (torch.from_numpy(x)).float()
        y = (torch.from_numpy(y)).float()
        data = Data(x = x, edge_index = edge_index, y = y, edge_attr=edge_prop, patient=patient, weights = weight, hypergraph=hypergraph, edge_hg = edge_hg)
        
        edge_node_he = []
        node_num = x.shape[0]
        index = np.arange(0,node_num)
        loc = np.where(hypergraph.data.numpy()==1)
        for m in range(len(loc[0])):
            edge_node_he.append((loc[0][m]+node_num,loc[1][m]))
        edge_node_he = np.array(edge_node_he)
        index = np.arange(0,node_num)
        edge_node_he = np.transpose(edge_node_he, (1,0))
        edge_node_he_index = (torch.from_numpy(edge_node_he)).long()
        data.__setattr__("edge_node_he", edge_node_he_index)
        data.__setattr__("node_num", node_num)
        index = (torch.from_numpy(index)).long()
        data.__setattr__("index", index)
        
        if x.shape[0]==y.shape[0]:
            dataset.append(data)
        else:
            print(file[i*4])
    return dataset

def hypergraph_airwaytree_sub(hypertree, children_map, y, num=10):
    hypertree = hypertree + np.eye(y.shape[0])
    y_lobar = y
    hypergraph = []
    clique_lobar = (y_lobar>=18).astype(np.uint8)
    hypergraph.append(clique_lobar)
    lobar_nodes = np.where(y_lobar>=18)[0]
    start_nodes = Queue()
    for i in range(len(lobar_nodes)):
        children = np.where(children_map[lobar_nodes[i],:]==1)[0]
        if children is not None:
            for child in children:
                if child not in lobar_nodes:
                    start_nodes.put(child)
                    hypertree[child, lobar_nodes[i]] = 1
                    
    while(not start_nodes.empty()):
        cur = start_nodes.get()
        hypergraph.append(hypertree[cur,:])
        children = np.where(children_map[cur,:]==1)[0]
        if children is not None:
            for child in children:
                hypertree[child, cur] = 1
                hypergraph.append(hypertree[child,:])
                num_children = np.sum(hypertree[child,:])
                if num_children>num:
                    start_nodes.put(child)
  
    hypergraph = np.array(hypergraph)
    return hypergraph

def data_augmentation_pred_sub(pred, weight, edge, node_weight):
    main_list = []
    child_list = []
    for i in range(edge.shape[1]):
        if pred[edge[0,i]] != pred[edge[1,i]] and pred[edge[0,i]]>=18:
            main_list.append(edge[0,i])
            child_list.append(edge[1,i])
    for i in range(len(main_list)):
        mode = np.random.randint(10)
        if mode<6:
            weight[main_list[i]] = node_weight
            weight[child_list[i]] = node_weight
        elif mode<8:
            pred[main_list[i]] = pred[child_list[i]]
            weight[main_list[i]] = node_weight
        else:
            if pred[main_list[i]] >= 18:
                pred[child_list[i]] = 18
                weight[child_list[i]] = node_weight
    return pred, weight

def parent_children_map(edge, N):
    parent_map = np.zeros(N, dtype = np.uint16)
    children_map = np.zeros((N,N), dtype = np.uint8)
    for i in range(edge.shape[1]):
        parent_map[edge[1,i]] = edge[0,i]
        children_map[edge[0,i], edge[1,i]] = 1
    return parent_map, children_map

def hyper_tree(parent_map, children_map):
    hypertree = children_map.copy()
    children_map_copy = children_map.copy()
    N = len(parent_map)
    ends = np.zeros(N)
    while(children_map_copy.sum() != 0):
        cur_children = np.sum(children_map_copy, axis=1)
        for i in range(N):
            if ends[i]==0 and cur_children[i]==0:
                hypertree[parent_map[i],:] += hypertree[i,:]
                children_map_copy[parent_map[i],i] = 0
                ends[i] = 1                
    return hypertree

def hypergraph_edge_detection(hypergraph):
    node_num = hypergraph.shape[0]
    edge = []
    for i in range(1, node_num-1):
        for j in range(i+1, node_num):
            if (hypergraph[i,:] * hypergraph[j,:]).sum() == 1:
                edge.append([i,j])
                edge.append([j,i])
    edge = np.array(edge)
    if len(edge)>0:
        edge = np.transpose(edge, (1,0))
    else:
        edge = np.array([[0],[0]])
    return edge