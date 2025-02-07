import json
import os
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
import json
import torch
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

def get_iid2text(dataset_name):
    dir = os.path.join("recommendation", "processed_dataset", dataset_name)
    if not os.path.exists(dir):
        raise ValueError("do not processed such data, please run the ranking phase to generate data for re-ranking")

    with open(os.path.join(dir, "iid2text.json"), "r") as file:
        iid2text = json.load(file)
        iid2text = convert_keys_values_to_int(iid2text)

    return iid2text

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

def load_json(file_path):
    """load json"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: The file at {file_path} is not a valid JSON file.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None



def get_cos_similar_torch(v1, v2, device='cuda'):
    import torch.nn.functional as F
    if device == 'cuda':
        v1 = torch.tensor(v1).cuda()
        v2 = torch.tensor(v2).cuda()
        cos_sim = F.cosine_similarity(v1, v2)
        return cos_sim.to(torch.float).cpu().numpy()
    else:
        v1 = torch.tensor(v1).cpu().float()
        v2 = torch.tensor(v2).cpu().float()
        cos_sim = F.cosine_similarity(v1, v2)
        return cos_sim.numpy()

