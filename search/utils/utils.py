import os
import csv
import math
import gzip
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler


def split_list(origin_list, n):
    res_list = []
    L = len(origin_list)
    N = int(math.ceil(L / float(n)))
    begin = 0
    end = begin + N
    while begin < L:
        if end < L:
            temp_list = origin_list[begin:end]
            res_list.append(temp_list)
            begin = end
            end += N
        else:
            temp_list = origin_list[begin:]
            res_list.append(temp_list)
            break
    return res_list


def load_embedding(filename, sep = '\t'):
    '''
    load embedding from file
    :param filename: embedding file name
    :param sep: the char used as separation symbol
    :return: a dict with item name as key and embedding vector as value
    '''
    with open(filename, 'r') as fp:
        result = {}
        for l in fp:
            l = l.strip()
            if l == '':
                continue
            sp = l.split(sep)
            vals = [float(sp[i]) for i in range(1, len(sp))]
            result[sp[0]] = vals
        return result
    

def get_rel_feat(path):
    rel_feat = pd.read_csv(path)
    rel_feat_names = list(sorted(set(rel_feat.columns) - {'query', 'doc'}))
    rel_feat[rel_feat_names] = StandardScaler().fit_transform(rel_feat[rel_feat_names])
    rel_feat = dict(zip(map(lambda x: tuple(x), rel_feat[['query', 'doc']].values),
            rel_feat[rel_feat_names].values.tolist()))
    return rel_feat


def read_rel_feat(path):
    rel_feat = {}
    f = csv.reader(open(path, 'r'), delimiter = ',')
    next(f)
    for line in f:
        if line[0] not in rel_feat:
            rel_feat[line[0]] = {}
        if line[1] not in rel_feat[line[0]]:
            rel_feat[line[0]][line[1]] = np.array([float(val) for val in line[2:]])
    return rel_feat


def pkl_load(filename):
    if not os.path.exists(filename):
        print('filename={} not exists!')
        return
    with gzip.open(filename, 'rb') as f:
        data_dict = pickle.load(f)
    return data_dict


def pkl_save(data_dict, filename):
    with gzip.open(filename, 'wb') as f:
        pickle.dump(data_dict, f)

