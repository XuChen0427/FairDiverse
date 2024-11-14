import numpy as np
#from
import os

import pandas as pd
import yaml

from .process_dataset import Process
from .base_model import MF, GRU4Rec
from .rank_model import IPS

import torch.optim as optim

from .sampler import PointWiseDataset, PairWiseDataset, RankingTestDataset, SequentialDataset
from .evaluator import CTR_Evaluator, Ranking_Evaluator

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm,trange
from datetime import datetime
import torch
import json
from scipy.sparse import save_npz, load_npz
import torch.nn as nn
import ast



class RecTrainer(object):
    def __init__(self, dataset, stage, train_config):
        self.dataset = dataset
        self.stage = stage
        self.train_config = train_config


        #self.topk = topk

    # def process_dataset(self):
    #     if not os.path.exists(os.path.join("processed_dataset", str(self.dataset))):
    #         Process(self.dataset)
    def load_configs(self, dir):
        print("start to load config...")
        with open(os.path.join(dir, "process_config.yaml"), 'r') as f:
            config = yaml.safe_load(f)


        # print(train_data_df.head())

        print("start to load model...")
        with open(os.path.join("recommendation", "properties", "models.yaml"), 'r') as f:
            model_config = yaml.safe_load(f)

        with open(os.path.join("recommendation", "properties", "models", self.train_config['model'] + ".yaml"),
                  'r') as f:
            model_config.update(yaml.safe_load(f))

        if self.train_config['fair-rank'] == True:
            with open(os.path.join("recommendation", "properties", "models", self.train_config['rank_model'] + ".yaml"),
                      'r') as f:
                model_config.update(yaml.safe_load(f))

        config.update(model_config)

        with open(os.path.join("recommendation", "properties", "evaluation.yaml"), 'r') as f:
            config.update(yaml.safe_load(f))

        config.update(self.train_config)  ###train_config has highest rights
        print("your loading config is:")
        print(config)

        return config

    def Set_Dataset(self, data_type, config, train_data_df, val_data_df, test_data_df):
        if data_type == 'point':
            train = PointWiseDataset(train_data_df, config)
        elif data_type == 'pair':
            train = PairWiseDataset(train_data_df, config)
        elif data_type == 'sequential':
            train = SequentialDataset(train_data_df, config)
        else:
            raise NotImplementedError("train_type only supports in [point, pair, sequential]")

        if config['eval_type'] == 'CTR':
            valid = PointWiseDataset(val_data_df, config)
            test = PointWiseDataset(test_data_df, config)
        elif config['eval_type'] == 'ranking':
            valid = RankingTestDataset(val_data_df, config)
            test = RankingTestDataset(test_data_df, config)
        else:
            raise NotImplementedError("We only support the eval type as [CTR, ranking]")

        return train, valid, test

    def train(self):
        dir = os.path.join("recommendation", "processed_dataset", self.dataset)
        config = self.load_configs(dir)

        state = Process(self.dataset)
        print(state)
        #exit(0)

        print("start to load dataset......")

        self.device = config['device']

        if config['model'] == 'mf':
            self.Model = MF(config).to(self.device)
        elif config['model'] == 'gru4rec':
            self.Model = GRU4Rec(config).to(self.device)
        else:
            raise NotImplementedError(f"Not supported model type: {config['model']}")

        self.group_weight = np.ones(config['group_num'])

        if config['fair-rank'] == True:
            if config['rank_model'] == "IPS":
                self.Fair_Ranker = IPS(config, self.group_weight)

            else:
                NotImplementedError(f"Not supported fair rank model type:{config['rank_model']}")


        if config['data_type'] not in self.Model.type:
            raise ValueError(f"The tested data type does not align with the model type: input is {config['data_type']}, "
                             f"the model only support: {self.Model.type}")
        if config['stage'] not in self.Model.IR_type:
            raise ValueError(f"The tested stage does not align with the model stage: input is {config['stage']}, "
                             f"the model only support: {self.Model.IR_type}")

        train_data_df = pd.read_csv(os.path.join(dir, self.dataset + ".train"), sep='\t')
        val_data_df = pd.read_csv(os.path.join(dir, self.dataset + ".valid." + config['eval_type']), sep='\t')
        test_data_df = pd.read_csv(os.path.join(dir, self.dataset + ".test." + config['eval_type']), sep='\t')

        print(test_data_df.head())
        train_data_df["history_behaviors"] = train_data_df["history_behaviors"].apply(lambda x: np.array(ast.literal_eval(x)))
        val_data_df["history_behaviors"] = val_data_df["history_behaviors"].apply(lambda x: np.array(ast.literal_eval(x)))
        test_data_df["history_behaviors"] = test_data_df["history_behaviors"].apply(lambda x: np.array(ast.literal_eval(x)))

        optimizer = optim.Adam(self.Model.parameters(), lr= config['learning_rate'])
        data_type = config['data_type']



        train, valid, test = self.Set_Dataset(data_type, config, train_data_df, val_data_df, test_data_df)

        train_loader = DataLoader(train, batch_size=config['batch_size'], shuffle=True)
        valid_loader = DataLoader(valid, batch_size=config['eval_batch_size'], shuffle=False)
        test_loader = DataLoader(test, batch_size=config['eval_batch_size'], shuffle=False)

        if config['eval_type'] == 'CTR':
            evaluator = CTR_Evaluator(config)
        elif config['eval_type'] == 'ranking':
            evaluator = Ranking_Evaluator(config)
        else:
            raise NotImplementedError("we only support eval type in [CTR, ranking] !")

        today = datetime.today()
        today_str = f"{today.year}-{today.month}-{today.day}"
        log_dir = os.path.join("recommendation", "log", f"{today_str}_{config['log_name']}")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        print("start to train...")

        for epoch in trange(config['epoch']):
            total_loss = 0
            self.Model.train()
            best_result = -1

            #for user_ids, item_ids, group_ids, label in train_loader:
            for train_datas in train_loader:

                if data_type == "point":
                    interaction = {"user_ids": train_datas[0].to(self.device), "item_ids": train_datas[1].to(self.device),
                                   "group_ids": train_datas[2].to(self.device), "label": train_datas[3].to(self.device)}
                elif data_type == "pair":
                    interaction = {"user_ids": train_datas[0].to(self.device), "pos_item_ids": train_datas[1].to(self.device),
                                   "pos_group_ids": train_datas[2].to(self.device), "neg_item_ids": train_datas[3].to(self.device),
                                   "neg_group_ids": train_datas[4].to(self.device)
                                   }
                else: ###squential format
                    interaction = {"user_ids": train_datas[0].to(self.device), "history_ids": train_datas[1].to(self.device),
                                   "item_ids": train_datas[2].to(self.device), "group_ids": train_datas[3].to(self.device), }

                optimizer.zero_grad()
                loss = self.Model.compute_loss(interaction)
                if config['fair-rank'] == True and self.Fair_Ranker.fair_type == 're-weight':
                    input_dict = {'items': train_datas[2].detach().numpy(), 'loss': loss.detach().cpu().numpy()}
                    weight = self.Fair_Ranker.reweight(input_dict=input_dict)
                    if epoch != 0 and epoch % config['update_epoch'] == 0:
                        self.Fair_Ranker.reset_parameters()
                    #print(weight)
                    #exit(0)
                    loss = torch.mean(torch.tensor(weight).to(self.device)*loss)
                else:
                    loss = torch.mean(loss)

                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if epoch % config['eval_step'] == 0:
                eval_result = evaluator.eval(valid_loader, self.Model)
                watch_eval_value = eval_result[config['watch_metric']]
                if watch_eval_value >= best_result:
                    best_result = watch_eval_value
                    torch.save(self.Model.state_dict(), os.path.join(log_dir, "best_model.pth"))
                print(f"eval result: {eval_result}, best result: {best_result}")
                print()


            print("epoch: %d loss: %.3f" %(epoch, total_loss/ len(train_loader)))



        print(f"training complete! start to save the config and model...")
        print(f" config files are dump in {log_dir}")
        with open(os.path.join(log_dir, "config.yaml"), 'w') as f:
            yaml.dump(config, f)

        print("start to testing...")
        self.Model.load_state_dict(torch.load(os.path.join(log_dir, "best_model.pth")))  # load state_dict
        self.Model.eval()  # change to eval model
        if config['store_scores'] == False:
            test_result = evaluator.eval(test_loader, self.Model)
        else:
            test_result, coo_matrix = evaluator.eval(test_loader, self.Model, store_scores=True)
            save_npz(os.path.join(log_dir, 'ranking_scores.npz'), coo_matrix) ##prepared for re-ranking stage


        with open(os.path.join(log_dir, 'test_result.json'), 'w') as file:
            json.dump(test_result, file)
        print(test_result)
        print(f"dump in {log_dir}")


