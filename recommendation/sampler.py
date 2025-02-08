import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random
import json
import os
import ast
from .utils import Build_Adjecent_Matrix, convert_keys_values_to_int


class AbstractDataset(Dataset):
    def __init__(self, df, config):
        """
            Abstract dataset utilized for training

            :param df: The input DataFrame.
            :param config: The merged config file.
        """


        self.config = config
        self.uid_field, self.iid_field, self.pid_field, self.label_field = config['user_id'], config['item_id'], config['group_id'], config['label_id']
        history_column = 'history_behaviors'

        self.M, self.iid2pid = Build_Adjecent_Matrix(config)

        self.user_ids = torch.tensor(df[self.uid_field].values, dtype=torch.long)
        self.item_ids = torch.tensor(df[self.iid_field].values, dtype=torch.long)
        self.label = torch.tensor(df[self.label_field].values, dtype=torch.float32)
        self.group_id = torch.tensor(df[self.pid_field].values, dtype=torch.long)
        self.history_ids = torch.stack([torch.tensor(row,  dtype=torch.long) for row in df[history_column]])
        #self.history_ids = torch.tensor(df[history_column].values, dtype=torch.long)


    def __len__(self):
        return len(self.label)

class PointWiseDataset(AbstractDataset):

    def __getitem__(self, idx):
        """
           getitem function for pointwise dataset

           :param idx: The index of the sample
           :return: The utilized features of the dataset
        """
        return self.user_ids[idx], self.history_ids[idx], self.item_ids[idx], self.group_id[idx], self.label[idx]

class SequentialDataset(AbstractDataset):

    def __init__(self, df, config):
        """
          SequentialDataset initialization:we only utilize the positive label and sample some neg labels

          :param df: The input DataFrame.
          :param config: The merged config file.
       """
        filter_df = df[df[config['label_id']] == 1]
        super().__init__(filter_df, config)

    def __getitem__(self, idx):
        """
           getitem function for sequential dataset

           :param idx: The index of the sample
           :return: The utilized features of the sequential dataset
        """
        return self.user_ids[idx], self.history_ids[idx], self.item_ids[idx], self.group_id[idx]


class PairWiseDataset(AbstractDataset):
    def __init__(self, df, config):
        """
           PairWiseDataset initialization: here we construct the positive samples for each items

           :param df: The input DataFrame.
           :param config: The merged config file.
        """
        filter_df = df[df[config['label_id']] == 1]
        super().__init__(filter_df, config)
        self.user2pos = {}
        for row in filter_df[[self.uid_field, self.iid_field, self.pid_field]].itertuples(index=True):
            #print(row)
            user_id, item_id, pid = row._1, row._2, row._3
            if user_id not in self.user2pos.keys():
                self.user2pos[user_id] = []
            self.user2pos[user_id].append(item_id)
        for key in self.user2pos.keys():
            self.user2pos[key] = list(set(self.user2pos[key]))


    def __getitem__(self, idx):
        """
          getitem function for sequential dataset with negative sampling operations

          :param idx: The index of the sample
          :return: The utilized features of the sequential dataset
       """
        user_id = int(self.user_ids[idx].numpy())
        neg_item = 0
        while neg_item in self.user2pos[user_id]:
            neg_item = random.randint(0, self.config['item_num']-1)

        neg_group = torch.tensor(self.iid2pid[neg_item], dtype=torch.long)
        neg_item = torch.tensor(neg_item, dtype=torch.long)

        return self.user_ids[idx], self.history_ids[idx], self.item_ids[idx], neg_item, self.group_id[idx], neg_group
        #return self.user_ids[idx], self.item_ids[idx], self.label[idx], self.group_id[idx]

class RankingTestDataset(Dataset):
    def __init__(self, df, config):
        with open(os.path.join("recommendation", "processed_dataset", config['dataset'], "iid2pid.json"), "r") as file:
            self.item2pid = json.load(file)
        df['items'] = df['items'].apply(lambda x: np.array(ast.literal_eval(x)))
        self.user_ids = torch.tensor(df["user_id"].values, dtype=torch.long)
        self.items = torch.stack([torch.tensor(row, dtype=torch.long) for row in df["items"]])
        self.pos_length = torch.tensor(df["pos_length"].values, dtype=torch.long)
        self.history_ids = torch.stack([torch.tensor(row, dtype=torch.long) for row in df["history_behaviors"]])

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        """
             getitem function for RankingDataset dataset under test dataset since in test set we already finish negative samples

             :param idx: The index of the sample
             :return: The utilized features of the RankingDataset
        """
        return self.user_ids[idx], self.history_ids[idx], self.items[idx], self.pos_length[idx]