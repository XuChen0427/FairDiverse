import json
import os
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

def convert_keys_values_to_int(data):

    if isinstance(data, dict):
        return {int(k): convert_keys_values_to_int(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_keys_values_to_int(element) for element in data]
    elif isinstance(data, str) and data.isdigit():
        return int(data)
    else:
        return data

def Init_Group_AdjcentMatrix(dataset_name):
    dir = os.path.join("recommendation", "processed_dataset", dataset_name)
    if not os.path.exists(dir):
        raise ValueError("do not processed such data, please run the ranking phase to generate data for re-ranking")

    with open(os.path.join(dir, "iid2pid.json"), "r") as file:
        iid2pid = json.load(file)
        iid2pid = convert_keys_values_to_int(iid2pid)

    return iid2pid

def Build_Adjecent_Matrix(config):
    iid2pid = Init_Group_AdjcentMatrix(config['dataset'])
    row = list(iid2pid.keys())
    col = list(iid2pid.values())
    data = np.ones_like(row)
    M = coo_matrix((data, (row, col)), shape=(config['item_num'], config['group_num']))
    M = M.toarray()

    for i in range(len(M)):
        if np.sum(M[i]) == 0:
            M[i][0] = 1
            iid2pid[i] = 0

    return M, iid2pid