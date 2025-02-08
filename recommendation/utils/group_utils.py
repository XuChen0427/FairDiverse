import json
import os
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
import json
import torch


def convert_keys_values_to_int(data):
    """
       Recursively converts dictionary keys and values, as well as string digits, into integers.

       This function will process a dictionary or list and convert all keys and values to integers where applicable.
       If the data is a string that represents a digit, it will be converted to an integer.

       :param data: The input data to be processed, which can be a dictionary, list, or string.
       :return: The input data with keys and values converted to integers if applicable.
   """

    if isinstance(data, dict):
        return {int(k): convert_keys_values_to_int(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_keys_values_to_int(element) for element in data]
    elif isinstance(data, str) and data.isdigit():
        return int(data)
    else:
        return data

def Init_Group_AdjcentMatrix(dataset_name):
    """
        Initializes the group adjacency matrix by loading the processed dataset corresponding to the given dataset name.

        This function checks if the processed dataset exists for the provided dataset name, and if not, raises an error.
        It then loads the `iid2pid.json` file, converts the keys and values to integers, and returns the result.

        :param dataset_name: The name of the dataset to load.
        :return: A dictionary mapping item IDs (iid) to product IDs (pid).
    """


    dir = os.path.join("recommendation", "processed_dataset", dataset_name)
    if not os.path.exists(dir):
        raise ValueError("do not processed such data, please run the ranking phase to generate data for re-ranking")

    with open(os.path.join(dir, "iid2pid.json"), "r") as file:
        iid2pid = json.load(file)
        iid2pid = convert_keys_values_to_int(iid2pid)

    return iid2pid

def get_iid2text(dataset_name):
    """
        Loads and returns the mapping of item IDs (iid) to text descriptions from the processed dataset.

        This function checks if the processed dataset exists for the given dataset name, and if not, raises an error.
        It then loads the `iid2text.json` file, converts the keys and values to integers, and returns the result.

        :param dataset_name: The name of the dataset to load.
        :return: A dictionary mapping item IDs (iid) to their corresponding text descriptions.
    """


    dir = os.path.join("recommendation", "processed_dataset", dataset_name)
    if not os.path.exists(dir):
        raise ValueError("do not processed such data, please run the ranking phase to generate data for re-ranking")

    with open(os.path.join(dir, "iid2text.json"), "r") as file:
        iid2text = json.load(file)
        iid2text = convert_keys_values_to_int(iid2text)

    return iid2text

def Build_Adjecent_Matrix(config):
    """
        Builds an adjacency matrix based on the group-item mapping, initializing it with ones,
        and adjusting rows with no connections.

        This function uses the `Init_Group_AdjcentMatrix` to retrieve a mapping of item IDs (iid) to product IDs (pid),
        constructs an adjacency matrix, and ensures that rows with no connections are assigned a default value.

        :param config: A configuration dictionary containing dataset information and matrix dimensions.
        :return: A tuple containing:
            - A 2D NumPy array representing the adjacency matrix.
            - A dictionary mapping item IDs (iid) to product IDs (pid).
    """


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
    """
    Loads a JSON file from the specified file path.

    This function attempts to open and parse a JSON file. If successful, it returns the parsed data.
    If the file is not found, or the file content is not valid JSON, an error message is printed,
    and `None` is returned.

    :param file_path: The path to the JSON file to be loaded.
    :return: The parsed data from the JSON file if successful, otherwise `None`.
    """

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
    """
        Computes the cosine similarity between two vectors using PyTorch.

        This function calculates the cosine similarity between two input vectors, `v1` and `v2`, using the specified device
        (either `cuda` for GPU or `cpu`). The function utilizes PyTorch's `cosine_similarity` function and returns the result as a NumPy array.

        :param v1: The first vector for cosine similarity calculation.
        :param v2: The second vector for cosine similarity calculation.
        :return: The cosine similarity between `v1` and `v2`.
    """


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

