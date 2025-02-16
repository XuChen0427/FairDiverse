import os
import gzip
import time
import torch
import pickle
import argparse
import numpy as np
import sys
import json

from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader

from ..postprocessing_model.DALETOR import DALETOR
from ..utils.loss import ndcg_loss
from ..evaluator import evaluate_test_qids_DALETOR


class TrainDataset(Dataset):
    def __init__(self, train_list):
        self.data = train_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        X = self.data[idx][0].clone().detach().float()
        rel_feat = self.data[idx][1].clone().detach().float()
        div_feat = self.data[idx][2].clone().detach().float()
        X.requires_grad=False
        rel_feat.requires_grad=False
        div_feat.requires_grad=False
        return X, rel_feat, div_feat


def get_train_loader(fold, train_ids, config):
    data_list = []
    data_dir=os.path.join(config['data_dir'], config['model'], 'train/')
    doc_tensor_dict={}
    for qid in tqdm(train_ids,desc='load train data', ncols=80):
        file_path = os.path.join(data_dir,str(qid)+'.pkl.gz')
        with gzip.open(file_path,'rb') as f:
            try:
                sample_dict = pickle.load(f)
            except EOFError:
                continue
        data_list.extend(sample_dict[str(qid)])
    train_dataset=TrainDataset(data_list)
    loader=DataLoader(
        dataset=train_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    return loader


def get_test_dataset(fold, test_qids, config):
    data_dir = os.path.join(config['data_dir'], config['model'], 'test/')
    test_dataset = {}
    for qid in tqdm(test_qids, desc="load test data",ncols=80):
        file_path = os.path.join(data_dir,str(qid)+'.pkl.gz')
        with gzip.open(file_path,'rb') as f:
            try:
                temp_test_dict=pickle.load(f)
            except EOFError:
                continue
        test_dataset[str(qid)]=temp_test_dict[str(qid)]
    return test_dataset


def DALETOR_run(config):
    if not os.path.exists(os.path.join(config['model_save_dir'], config['model'])):
        os.makedirs(os.path.join(config['model_save_dir'], config['model']))
    all_qids=np.load(os.path.join(config['data_dir'], 'all_qids.npy'))
    qd=pickle.load(open(os.path.join(config['data_dir'], 'div_query.data'),'rb'))
    
    final_metrics=[]
    best_model_list=[]
    fold_time=0
    
    test_qids_list=[]
    for train_ids, test_ids in KFold(5).split(all_qids):
        fold_time+=1
        max_metric = 0
        train_ids.sort()
        test_ids.sort()
        train_qids=[str(all_qids[i]) for i in train_ids]
        test_qids=[str(all_qids[i]) for i in test_ids]
        test_qids_list.append(test_qids)

        train_data_loader = get_train_loader(fold_time, train_qids, config)
        test_dataset_dict = get_test_dataset(fold_time, test_qids, config)

        model = DALETOR(config['dropout'])
        if torch.cuda.is_available():
            model=model.cuda()
        opt = torch.optim.Adagrad(model.parameters(), lr=config['learning_rate'])
        params = list(model.parameters())
        if fold_time == 1:
            print('model={}'.format(model))
            print('len params={}'.format(len(params)))
            for param in params:
                print('{}'.format(param.size()))
            n_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
            print('* number of parameters: %d' % n_params)

        all_steps = len(train_data_loader)
        patience = 0
        
        for epoch in range(config['epoch']):
            print('Start Training...')
            epoch_step=0
            model.train()
            
            for step, train_data in enumerate(tqdm(train_data_loader, desc='BATCH', ncols=80)):
                X, rel_feat, div_feat = train_data
                if torch.cuda.is_available():
                    X = X.cuda()
                    rel_feat = rel_feat.cuda()
                    div_feat = div_feat.cuda()
                    
                score = model.fit(X, rel_feat, True)
                loss = ndcg_loss(score, div_feat)

                opt.zero_grad()
                loss.backward()
                clip_grad_norm_(model.parameters(), max_norm=1)
                opt.step()

                epoch_step+=1
                if (step + 1) % config['eval_steps'] == 0:
                    model.eval()
                    metrics = []
                    for qid in test_qids:
                        metric = evaluate_test_qids_DALETOR(model, test_dataset_dict[str(qid)], qd[str(qid)], 'metric')
                        metrics.append(metric)
                    avg_alpha_NDCG = np.mean(metrics)

                    if max_metric < avg_alpha_NDCG:
                        max_metric = avg_alpha_NDCG
                        print('max avg_alpha_NDCG updated: {}'.format(max_metric))
                        model_filename = os.path.join(config['model_save_dir'], config['model'], 'TOTAL_EPOCH_' + str(config['epoch']) + '_FOLD_' + str(fold_time) + '_EPOCH_' + str(epoch) + '_LR_' + str(config['learning_rate']) + '_BATCHSIZE_' + str(
                        config['batch_size']) + '_DROPOUT_' + str(config['dropout']) + '_' + str(config['embedding_type']) + '.pickle')
                        torch.save(model.state_dict(),model_filename)
                        best_model = model_filename
                        patience = 0
                    else: 
                        patience += 1
                    model.train()

                    if epoch > 0 and patience > 2:
                        new_lr = 0.0
                        for param_group in opt.param_groups:
                            param_group['lr'] = param_group['lr'] * 0.5
                            new_lr = param_group['lr']
                        patience = 0

            model.eval()
            metrics = []
            for qid in test_qids:
                metric = evaluate_test_qids_DALETOR(model, test_dataset_dict[str(qid)], qd[str(qid)], 'metric')
                metrics.append(metric)
            avg_alpha_NDCG = np.mean(metrics)
            if max_metric < avg_alpha_NDCG:
                max_metric = avg_alpha_NDCG
                model_filename = os.path.join(config['model_save_dir'], config['model'], 'TOTAL_EPOCH_' + str(config['epoch']) + '_FOLD_' + str(fold_time) + '_EPOCH_' + str(epoch) + '_LR_' + str(config['learning_rate']) + '_BATCHSIZE_' + str(
                        config['batch_size']) + '_DROPOUT_' + str(config['dropout']) + '_' + str(config['embedding_type']) + '.pickle')
                torch.save(model.state_dict(),model_filename)
                best_model = model_filename
            if epoch == (config['epoch']-1):
                final_metrics.append(max_metric)
                best_model_list.append(best_model)
    with open(os.path.join(config['model_save_dir'], config['model'], 'fold_qid.json'), 'w') as f:
        json.dump(test_qids_list, f)
    print('alpha-nDCG = {}, best model = {}'.format(sum(final_metrics)/len(final_metrics), best_model_list))
