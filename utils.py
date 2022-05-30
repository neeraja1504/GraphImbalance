import numpy as np
import scipy.sparse as sp
import torch
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix,classification_report
from copy import deepcopy
from scipy.spatial.distance import pdist,squareform
import argparse
import scipy.sparse as sp
import numpy as np
import torch
import ipdb
from scipy.io import loadmat
import networkx as nx
import multiprocessing as mp
import torch.nn.functional as F
from functools import partial
import random


def encode_onehot(labels):
    # The classes must be sorted before encoding to enable static class encoding.
    # In other words, make sure the first class always maps to index 0.
    classes = sorted(list(set(labels)))
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def load_data(path="./data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize_features(features)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    adj = torch.FloatTensor(np.array(adj.todense()))
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def print_class_acc(output, labels, class_num_list, pre='valid'):
    pre_num = 0
    #print class-wise performance
    
    # for i in range(labels.max()+1):
        
    #     cur_tpr = accuracy(output[pre_num:pre_num+class_num_list[i]], labels[pre_num:pre_num+class_num_list[i]])
    #     print(str(pre)+" class {:d} True Positive Rate: {:.3f}".format(i,cur_tpr.item()))
    #     index_negative = labels != i
    #     labels_negative = labels.new(labels.shape).fill_(i)
        
    #     cur_fpr = accuracy(output[index_negative,:], labels_negative[index_negative])
    #     print(str(pre)+" class {:d} False Positive Rate: {:.3f}".format(i,cur_fpr.item()))
    #     pre_num = pre_num + class_num_list[i]
    

    #ipdb.set_trace()
    if labels.max() > 1:
        auc_score = roc_auc_score(labels.cpu().detach(), F.softmax(output.cpu(), dim=-1).detach(), average='macro', multi_class='ovr')
    else:
        auc_score = roc_auc_score(labels.cpu().detach(), F.softmax(output.cpu(), dim=-1)[:,1].detach(), average='macro')

    macro_F = f1_score(labels.cpu().detach(), torch.argmax(output.cpu(), dim=-1).detach(), average='macro')
    print(str(pre)+' current auc-roc score: {:f}, current macro_F score: {:f}'.format(auc_score,macro_F))

    # cr=classification_report(labels.cpu().detach(), torch.argmax(output.cpu(), dim=-1).detach())
    # print("Classification report is",cr)


    return  macro_F  

def print_class_acc_test(output, labels, class_num_list, pre='test'):
    pre_num = 0
    # print class-wise performance
    
    # for i in range(labels.max()+1):
        
    #     cur_tpr = accuracy(output[pre_num:pre_num+class_num_list[i]], labels[pre_num:pre_num+class_num_list[i]])
    #     print(str(pre)+" class {:d} True Positive Rate: {:.3f}".format(i,cur_tpr.item()))
    #     index_negative = labels != i
    #     labels_negative = labels.new(labels.shape).fill_(i)
        
    #     cur_fpr = accuracy(output[index_negative,:], labels_negative[index_negative])
    #     print(str(pre)+" class {:d} False Positive Rate: {:.3f}".format(i,cur_fpr.item()))
    #     pre_num = pre_num + class_num_list[i]
    

    #ipdb.set_trace()
    if labels.max() > 1:
        auc_score = roc_auc_score(labels.cpu().detach(), F.softmax(output.cpu(), dim=-1).detach(), average='macro', multi_class='ovr')
    else:
        auc_score = roc_auc_score(labels.cpu().detach(), F.softmax(output.cpu(), dim=-1)[:,1].detach(), average='macro')

    macro_F = f1_score(labels.cpu().detach(), torch.argmax(output.cpu(), dim=-1).detach(), average='macro')
    print(str(pre)+' current auc-roc score: {:f}, current macro_F score: {:f}'.format(auc_score,macro_F))

    weight_F= f1_score(labels.cpu().detach(), torch.argmax(output.cpu(), dim=-1).detach(), average='weighted')
    print("weighted_F1 is",weight_F)

    cm=confusion_matrix(labels.cpu().detach(), torch.argmax(output.cpu(), dim=-1).detach())
    print("Confusion matrix is \n",cm)

    cr=classification_report(labels.cpu().detach(), torch.argmax(output.cpu(), dim=-1).detach())
    print("Classification report is \n",cr)

    return  macro_F    
