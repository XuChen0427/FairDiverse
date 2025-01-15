import numpy as np
import os

import yaml
import random

import time
from tqdm import tqdm,trange
from datetime import datetime
import torch
import json
from scipy.sparse import save_npz, load_npz
import torch.nn as nn
import ast



class SRDTrainer(object):
    def __init__(self, dataset, stage, train_config):
        self.dataset = dataset
        self.stage = stage
        self.train_config = train_config

    def load_configs(self, dir):
        print("start to load dataset config...")
        with open(os.path.join(self.train_config['task'], "properties", "dataset", self.train_config['dataset'] + ".yaml"), 'r') as f:
            config = yaml.safe_load(f)
        config.update({'data_dir': dir})

        print("start to load model config...")

        with open(os.path.join(self.train_config['task'], "properties", "models", self.train_config['model'] + ".yaml"),
                  'r') as f:
            model_config = yaml.safe_load(f)

        config.update(model_config)

        with open(os.path.join("recommendation", "properties", "evaluation.yaml"), 'r') as f:
            config.update(yaml.safe_load(f))

        config.update(self.train_config)  ###train_config has highest rights
        print("your loading config is:")
        print(config)

        return config


    def train(self):
        dir = os.path.join(self.train_config['task'], "processed_dataset", self.dataset)
        config = self.load_configs(dir)

        if os.path.exists(os.path.join(config['task'], "processed_dataset", config['dataset'], config['model'])) and config['reprocess'] == False:
            print("Data has been processed, start to load the dataset...")
        else:
            print("start to process data...")
            if config['model'].lower() == 'desa':
                from .utils.process_dataset import data_process
                from .utils.div_type import div_dataset
                from .datasets.desa import divide_five_fold_train_test
                data_process(config)
                D = div_dataset(config)
                D.get_listpair_train_data()
                divide_five_fold_train_test(config)
            elif config['model'].lower() == 'daletor':
                pass
            elif config['model'].lower() == 'graph4div':
                pass
            elif config['model'].lower() == 'llm':
                pass
            else:
                raise NotImplementedError(f"Not supported model type: {config['model']}")

        print("start to load dataset......")

        self.device = config['device']
        if config['mode'] == 'test' and config['best_model_list'] != []:
            print("start to test the model...")
            from .evaluator import get_global_fullset_metric
            get_global_fullset_metric(config)
        elif config['mode'] == 'train':
            if config['model'].lower() == 'desa':
                from .datasets.desa import DESA_run
                DESA_run(config)
            elif config['model'].lower() == 'daletor':
                pass
            elif config['model'].lower() == 'graph4div':
                pass
            elif config['model'].lower() == 'llm':
                pass
            else:
                raise NotImplementedError(f"Not supported model type: {config['model']}")

        '''
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
        '''


