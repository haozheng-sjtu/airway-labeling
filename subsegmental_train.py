# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 16:12:39 2021

@author: user
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
import torch_geometric.nn as pyg_nn
import os
import numpy as np
from dataset import prepare_dataset_subsegmental_hg_pred
from network import TNN
from loss_functions import LabelSmoothCrossEntropyLoss
torch.manual_seed(666) # cpu
torch.cuda.manual_seed(666) #gpu
np.random.seed(666) #numpy
import time
import shutil
import sys
from utils import *

train_path = ""
test_path = ""
epochs = 800

dataset3 = prepare_dataset_subsegmental_hg_pred(train_path)
train_loader_val = DataLoader(dataset3,batch_size = 1,shuffle = False, num_workers = 0,pin_memory=True)
dataset2 = prepare_dataset_subsegmental_hg_pred(test_path)
test_loader_case = DataLoader(dataset2,batch_size = 1,shuffle = False, num_workers = 0,pin_memory=True)


save_dir = "checkpoints/"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
logfile = os.path.join(save_dir,'log')
sys.stdout = Logger(logfile)
pyfiles = [f for f in os.listdir('./') if f.endswith('.py')]
for f in pyfiles:
    shutil.copy(f,os.path.join(save_dir,f))
    
my_net = TNN(nfeat=28, nhid=64, nclass=127, nlayer=3, nhead=1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
my_net = my_net.to(device)  
optimizer = torch.optim.SGD(my_net.parameters(), lr=1e-3, momentum=0.9)  
   
my_net.train()
for epoch in range(epochs):
    time1 = time.time()
    test_accuracy = []
    train_accuracy = []
    train_loss = []
    dataset1 = prepare_dataset_subsegmental_hg_pred(train_path, node_weight=10, train=True)
    train_loader_case = DataLoader(dataset1,batch_size = 1,shuffle = True, num_workers = 0,pin_memory=True)
    for case in train_loader_case:
        edge = case.edge_index.to(device)
        edge_prop = case.edge_attr
        x = case.x.to(device)
        y = case.y.to(device)
        y = y.long()
        H = case.hypergraph.to(device)
        E = case.edge_hg.to(device)
    
        optimizer.zero_grad()  
    
        output = my_net(x,H,E,case.to(device))  
        
        weight = case.weights.to(device) 
        loss_function = LabelSmoothCrossEntropyLoss(weight = weight, smoothing = 0.02)
        loss = loss_function(output, y)
        loss.backward()
        
        pred = output.max(dim = 1)
        label = y.cpu().data.numpy()
        pred = pred[1].cpu().data.numpy()
        acc = np.sum((label==pred).astype(np.uint8))/(label.shape[0])
        train_accuracy.append(acc)
        train_loss.append(loss.item())

        optimizer.step()  
        
    train_accuracy = np.array(train_accuracy)
    train_loss = np.array(train_loss)
    train_mean_acc = np.mean(train_accuracy)
    train_mean_loss = np.mean(train_loss)
    print("epoch:", epoch + 1, "loss:", train_mean_loss, "acc:", train_mean_acc, "time:",time.time()-time1)
        
    if (epoch+1)%10 == 0:
        for case in test_loader_case:
            edge = case.edge_index.to(device)
            x = case.x.to(device)
            y = case.y.to(device)
            edge_prop = case.edge_attr
            H = case.hypergraph.to(device)
            E = case.edge_hg.to(device)
            
            pred,_ = my_net(x,H,E,case.to(device))
            pred = pred.max(dim = 1)
            
            y = y.cpu().data.numpy()
            pred = pred[1].cpu().data.numpy()
            acc = np.sum((y==pred).astype(np.uint8))/(y.shape[0])
            test_accuracy.append(acc)
        test_accuracy = np.array(test_accuracy)
        mean_acc = test_accuracy.mean()
        print("Accuracy of Test Samples:{}%".format(mean_acc))
    
    if (epoch+1)%100 == 0:
        state_dict = my_net.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].cpu()
        torch.save({
        'epoch': epoch+1,
        'save_dir': save_dir,
        'state_dict': state_dict},
        os.path.join(save_dir, '%04d.ckpt' % (epoch+1)))


my_net.eval()

accuracy = []
sen_class = []
pre_class = []
for case in train_loader_val:
    edge = case.edge_index.to(device)
    x = case.x.to(device)
    y = case.y.to(device)
    edge_prop = case.edge_attr
    H = case.hypergraph.to(device)
    E = case.edge_hg.to(device)
    
    pred,_ = my_net(x, H, E,case.to(device))
    pred = pred.max(dim = 1)

    y = y.cpu().data.numpy()
    pred = pred[1].cpu().data.numpy()

    acc = np.sum((y==pred).astype(np.uint8))/(y.shape[0])
    accuracy.append(acc)
    
    sen_c = np.zeros(127)
    pre_c = np.zeros(127)
    for j in range(127):
        pred_c = (pred==j).astype(np.uint8)
        label_c = (y==j).astype(np.uint8)
        sen_c[j] = (pred_c*label_c).sum()/label_c.sum() if label_c.sum()>0 else 1
        pre_c[j] = (pred_c*label_c).sum()/pred_c.sum() if pred_c.sum()>0 else 0
    sen_class.append(sen_c)
    pre_class.append(pre_c)
      
accuracy = np.array(accuracy)
mean_acc = accuracy.mean()
print("Accuracy of Train Samples:{}%".format(mean_acc))
print("class-wise sensitivity:")
sen_class = np.array(sen_class)
print(sen_class.mean(axis=0))
print("class-wise precision:")
pre_class = np.array(pre_class)
print(pre_class.mean(axis=0))

accuracy = []
sen_class = []
pre_class = []
for case in test_loader_case:
    edge = case.edge_index.to(device)
    x = case.x.to(device)
    y = case.y.to(device)
    edge_prop = case.edge_attr
    H = case.hypergraph.to(device)
    E = case.edge_hg.to(device)
    
    pred,_ = my_net(x,H,E,case.to(device))
    pred = pred.max(dim = 1)

    y = y.cpu().data.numpy()
    pred = pred[1].cpu().data.numpy()
    acc = np.sum((y==pred).astype(np.uint8))/(y.shape[0])
    accuracy.append(acc)
    
    sen_c = np.zeros(127)
    pre_c = np.zeros(127)
    for j in range(127):
        pred_c = (pred==j).astype(np.uint8)
        label_c = (y==j).astype(np.uint8)
        sen_c[j] = (pred_c*label_c).sum()/label_c.sum() if label_c.sum()>0 else 1
        pre_c[j] = (pred_c*label_c).sum()/pred_c.sum() if pred_c.sum()>0 else 0
    sen_class.append(sen_c)
    pre_class.append(pre_c)
      
accuracy = np.array(accuracy)
mean_acc = accuracy.mean()
print("Accuracy of Test Samples:{}%".format(mean_acc))
print("class-wise sensitivity:")
sen_class = np.array(sen_class)
print(sen_class.mean(axis=0))
print("class-wise precision:")
pre_class = np.array(pre_class)
print(pre_class.mean(axis=0))
      
