import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random
import json
import os
import ast


def convert_keys_values_to_int(data):

    if isinstance(data, dict):
        return {int(k): convert_keys_values_to_int(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_keys_values_to_int(element) for element in data]
    elif isinstance(data, str) and data.isdigit():
        return int(data)
    else:
        return data

class AbstractDataset(Dataset):
    def __init__(self, df, config):
        uid, iid, pid, label = config['user_id'], config['item_id'], config['group_id'], config['label_id']
        history_column = 'history_behaviors'

        with open(os.path.join("recommendation", "processed_dataset", config['dataset'], "iid2pid.json"), "r") as file:
            self.item2pid = json.load(file)
            self.item2pid = convert_keys_values_to_int(self.item2pid)

        #df[history_column] = df[history_column].apply(lambda x: x.apply(ast.literal_eval))
        #a = np.array(df[history_column].values)
        #print(type(a[0]))

        self.user_ids = torch.tensor(df[uid].values, dtype=torch.long)
        self.item_ids = torch.tensor(df[iid].values, dtype=torch.long)
        self.label = torch.tensor(df[label].values, dtype=torch.float32)
        self.group_id = torch.tensor(df[pid].values, dtype=torch.float32)
        self.history_ids = torch.stack([torch.tensor(row,  dtype=torch.long) for row in df[history_column]])
        #self.history_ids = torch.tensor(df[history_column].values, dtype=torch.long)


    def get_i2p_dict(self):
        return self.item2pid

    def __len__(self):
        return len(self.label)

class PointWiseDataset(AbstractDataset):

    def __getitem__(self, idx):
        return self.user_ids[idx], self.item_ids[idx], self.group_id[idx], self.label[idx]

class SequentialDataset(AbstractDataset):
    def __init__(self, df, config):
        filter_df = df[df[config['label_id']] == 1]  ### for pair, we only utilize the positive label and sample some neg labels
        super().__init__(filter_df, config)

    def __getitem__(self, idx):
        return self.user_ids[idx], self.history_ids[idx], self.item_ids[idx], self.group_id[idx]


class PairWiseDataset(AbstractDataset):
    def __init__(self, df, config):
        filter_df = df[df[config['label_id']] == 1] ### for pair, we only utilize the positive label and sample some neg labels
        super().__init__(filter_df, config)

        uid, iid, pid, label = config['user_id'], config['item_id'], config['group_id'], config['label_id']
        self.user2pos = {}
        for row in df[[uid, iid, label, pid]].itertuples(index=True):
            user_id, item_id, label_id, pid = row.uid, row.iid, row.label, row.pid
            if user_id not in self.user2pos.keys():
                self.user2pos[user_id] = []
            if label_id == 1:
                self.user2pos[user_id].append(item_id)


    def __getitem__(self, idx):
        neg_idx = idx
        user_id = self.user_ids[idx]
        #pos_label = self.label[idx]

        if self.label[idx] == 0:
            pos_index = random.randint(0, len(user_id)-1)
            pos_item = self.user2pos[user_id][pos_index]
            pos_group = torch.tensor(self.item2pid[pos_item])
            pos_item = torch.tensor(pos_item, dtype=torch.long)
            return self.user_ids[idx], pos_item, self.item_ids[idx], pos_group, self.group_id[idx]

        else:
            neg_idx = 0
            while self.item_ids[neg_idx] in self.user2pos[user_id]:
                neg_idx = random.randint(0, len(self.item_ids)-1)
            neg_item = self.item_ids[neg_idx]
            neg_group = torch.tensor(self.item2pid[neg_item])
            neg_item = torch.tensor(neg_item, dtype=torch.long)

        return self.user_ids[idx], self.item_ids[idx], neg_item, self.group_id[idx], neg_group
        #return self.user_ids[idx], self.item_ids[idx], self.label[idx], self.group_id[idx]

class RankingTestDataset(Dataset):
    def __init__(self, df, config):
        with open(os.path.join("recommendation", "processed_dataset", config['dataset'], "iid2pid.json"), "r") as file:
            self.item2pid = json.load(file)

        #df['items'] = df['items'].apply(lambda x: np.array(ast.literal_eval(x)))
        df['items'] = df['items'].apply(eval)
        #print(df['items'].values[0])
        # a = np.array(df["items"].values)
        # print(a)
        # print(a.shape)

        self.user_ids = torch.tensor(df["user_id"].values, dtype=torch.long)
        self.items = torch.tensor(np.array(df["items"].tolist()), dtype=torch.long)
        self.pos_length = torch.tensor(df["pos_length"].values, dtype=torch.long)
        self.history_ids = torch.stack([torch.tensor(row, dtype=torch.long) for row in df["history_behaviors"]])

    def get_i2p_dict(self):
        return self.item2pid

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        return self.user_ids[idx], self.history_ids[idx], self.items[idx], self.pos_length[idx]