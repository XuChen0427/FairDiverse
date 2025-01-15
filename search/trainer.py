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
    def __init__(self, train_config):
        self.train_config = train_config

    def load_configs(self, dir):
        print("start to load dataset config...")
        with open(os.path.join(self.train_config['task'], "properties", "dataset", self.train_config['dataset'].lower() + ".yaml"), 'r') as f:
            config = yaml.safe_load(f)
        config.update({'data_dir': dir})

        print("start to load model config...")

        with open(os.path.join(self.train_config['task'], "properties", "models", self.train_config['model'].lower() + ".yaml"),
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
        dir = os.path.join(self.train_config['task'], "processed_dataset", self.train_config['dataset'])
        config = self.load_configs(dir)

        if os.path.exists(os.path.join(config['task'], "processed_dataset", config['dataset'], config['model'])) and config['reprocess'] == False:
            print("Data has been processed, start to load the dataset...")
        else:
            print("start to process data...")
            if os.path.join(config['data_dir'], 'div_query.data') not in config['data_dir']:
                from .utils.process_dataset import data_process
                data_process(config)
            if config['model'].lower() == 'desa':
                from .utils.div_type import div_dataset
                from .datasets.DESA import divide_five_fold_train_test
                D = div_dataset(config)
                D.get_listpair_train_data()
                divide_five_fold_train_test(config)
            elif config['model'].lower() == 'daletor':
                from .utils.process_daletor import Process
                Process(config)
            elif config['model'].lower() == 'xquad':
                from .utils.process_bm25 import generate_bm25_scores_for_query
                generate_bm25_scores_for_query(config)
            elif config['model'].lower() == 'pm2':
                from .utils.process_bm25 import generate_bm25_scores_for_query
                generate_bm25_scores_for_query(config)
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
                from .datasets.DESA import DESA_run
                DESA_run(config)
            elif config['model'].lower() == 'daletor':
                from .datasets.DALETOR import DALETOR_run
                DALETOR_run(config)
            elif config['model'].lower() == 'xquad':
                from .rerank_model.xQuAD import xQuAD
                xquad = xQuAD()
                xquad.run(config)
            elif config['model'].lower() == 'pm2':
                from .rerank_model.PM2 import PM2
                pm2 = PM2()
                pm2.run(config)
            elif config['model'].lower() == 'llm':
                from .llm_model.llm_run import llm_run
                llm_run(config)
            else:
                raise NotImplementedError(f"Not supported model type: {config['model']}")



