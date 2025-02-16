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
    """
    Splits the input list into smaller sublists of size n (or close to n).

    :param origin_list: The original list to be split.
    :param n: The number of sublists to split into.
    :return: A list of sublists.
    """
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
    """
    Load embedding from file
    :param filename: embedding file name
    :param sep: the char used as separation symbol
    :return: a dict with item name as key and embedding vector as value
    """

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
    """
    Loads and scales the relevance features from a CSV file.

    :param path: Path to the CSV file containing the relevance features.
    :return: A dictionary where the key is a tuple (query, doc) and the value is a list of features.
    """
    rel_feat = pd.read_csv(path)
    rel_feat_names = list(sorted(set(rel_feat.columns) - {'query', 'doc'}))
    rel_feat[rel_feat_names] = StandardScaler().fit_transform(rel_feat[rel_feat_names])
    rel_feat = dict(zip(map(lambda x: tuple(x), rel_feat[['query', 'doc']].values),
            rel_feat[rel_feat_names].values.tolist()))
    return rel_feat


def read_rel_feat(path):
    """
    Reads relevance features from a CSV file and returns them in a nested dictionary format.

    :param path: Path to the CSV file containing the relevance features.
    :return: A nested dictionary where the key is a query and the value is another dictionary of documents and features.
    """
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
    """
    Loads a pickle file and returns the data inside it.

    :param filename: Path to the pickle file.
    :return: The loaded data from the pickle file.
    """
    if not os.path.exists(filename):
        print('filename={} not exists!')
        return
    with gzip.open(filename, 'rb') as f:
        data_dict = pickle.load(f)
    return data_dict


def pkl_save(data_dict, filename):
    """ Saves a dictionary to a compressed pickle file.

    :param data_dict: The dictionary to be saved.
    :param filename: The path where the pickle file should be saved.
    """
    with gzip.open(filename, 'wb') as f:
        pickle.dump(data_dict, f)


def remove_duplicate(input_path, output_path):
    """ Removes duplicate documents in the ranking list.

    :param input_path: The path to the input file containing the ranking list.
    :param output_path: The path where the cleaned ranking list will be saved.
    """
    unique_records = set()
    output_lines = []

    with open(input_path, "r") as file:
        for line in file:
            parts = line.strip().split()
            topic_id = parts[0]
            document_name = parts[2]
            
            key = (topic_id, document_name)
            
            if key not in unique_records:
                unique_records.add(key)
                output_lines.append(line)

    with open(output_path, "w") as file:
        for line in output_lines:
            file.write(line)


def restore_doc_ids(order_str, id_dict):
    """ Restores document IDs based on an ordered list of indices and a dictionary of document IDs.

    :param order_str: A string representing the order of document indices.
    :param id_dict: A dictionary mapping indices to document IDs.
    :return: A list of document IDs in the restored order.
    """
    order = [int(x) for x in order_str.replace(" ", "").replace("[", "").replace("]", "").split(">")]
    reversed_dict = {v: k for k, v in id_dict.items()}
    return [reversed_dict[num] for num in order if num in reversed_dict]


