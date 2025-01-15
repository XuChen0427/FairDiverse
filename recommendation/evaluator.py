import torch
from tqdm import tqdm, trange
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from .metric import NDCG, HR, MRR, AUC_score, MMF, Gini, Entropy
import os
import json
from .utils import Build_Adjecent_Matrix




class Abstract_Evaluator(object):
    def __init__(self, config):
        self.config = config
        self.M, self.iid2pid = Build_Adjecent_Matrix(config)


    def eval(self, dataloader, model, store_scores = False):
        pass


class CTR_Evaluator(Abstract_Evaluator):
    def __int__(self,config):
        super().__init__(config=config)

    def eval(self, dataloader, model, store_scores = False):
        model.eval()
        y_scores = []
        y_true = []

        row = []
        col = []
        data = []

        with torch.no_grad():
            for user_ids, item_ids, group_ids, label in tqdm(dataloader):
                row.extend(user_ids.numpy().tolist())
                col.extend(item_ids.numpy().tolist())
                user_ids, item_ids = user_ids.to(self.config['device']), item_ids.to(self.config['device'])

                score = model(user_ids, item_ids).cpu().numpy().tolist()
                data.extend(score)
                label = label.cpu().numpy().tolist()
                y_scores.extend(score)
                y_true.extend(label)

        auc_score = AUC_score(y_scores=y_scores, y_true=y_true)
        result_dict = {}
        result_dict["auc"] = np.round(auc_score,self.config['decimals'])

        if store_scores == False:
            return result_dict
        else:
            coo = coo_matrix((data, (row, col)), shape=(self.config['user_num'], self.config['item_num']))
            csr = coo.tocsr() #to remove the zero rows
            csr_eliminated = csr[csr.getnnz(1) > 0]
            coo = csr_eliminated.tocoo()
            return result_dict, coo


class Ranking_Evaluator(Abstract_Evaluator):
    def __int__(self,config):
        super().__init__(config=config)

    def eval(self, dataloader, model, store_scores = False):
        model.eval()
        y_scores = []
        y_true = []

        result_dict = {f"ndcg@{k}":0 for k in self.config['topk']}
        result_dict.update({f"mrr@{k}":0 for k in self.config['topk']})
        result_dict.update({f"hr@{k}":0 for k in self.config['topk']})
        result_dict.update({f"mmf@{k}": 0 for k in self.config['topk']})
        result_dict.update({f"gini@{k}": 0 for k in self.config['topk']})
        result_dict.update({f"entropy@{k}": 0 for k in self.config['topk']})
        exposure_dict = {f"top@{k}":np.zeros(self.config['group_num']) for k in self.config['topk']}
        index = 0

        #UI_matrix = np.zeros((self.config['user_num'], self.config['item_num']))
        row = []
        col = []
        data = []

        with torch.no_grad():

            #for user_ids, history_behavior, items, pos_length in tqdm(dataloader):
            for eval_data in tqdm(dataloader):
                user_ids, history_behavior, items, pos_length = eval_data
                batch_size, sample_size = items.shape #item
                #print(items.shape)
                #exit(0)
                pos_length = pos_length.cpu().numpy()

                for b in range(batch_size):
                    row.extend([index]*sample_size)
                    index = index + 1
                    real_item_ids = items[b].numpy().tolist()
                    col.extend(real_item_ids)
                    #print(model.IR_type)
                    if 'retrieval' not in model.IR_type:
                        #if self.config['data_type'] == 'point' or self.config['data_type'] == 'pair':
                        repeat_user_tensor = user_ids[b].repeat(sample_size).unsqueeze(0).to(self.config['device'])
                        #else:
                        repeat_history_tensor = history_behavior[b].repeat(sample_size, 1).unsqueeze(0).to(self.config['device'])

                        user_dict = {"user_ids": repeat_user_tensor,
                                     "history_ids": repeat_history_tensor}
                        i = items[b].to(self.config['device'])
                        score = model(user_dict, i.unsqueeze(0)).cpu().numpy()[0]

                    else:
                        user_dict = {"user_ids":user_ids[b].unsqueeze(0).to(self.config['device']),
                                     "history_ids":history_behavior[b].unsqueeze(0).to(self.config['device'])}
                        i = items[b].to(self.config['device'])

                        score = model.full_predict(user_dict, i.unsqueeze(0)).cpu().numpy()[0]

                    data.extend(score.tolist())
                    #ranked_score = np.sort(score)[::-1]
                    label_list = [1] * pos_length[b] + [0] * (sample_size - pos_length[b])
                    label_list = np.array(label_list)
                    ranked_args = np.argsort(score)[::-1]
                    rank_list = label_list[ranked_args]
                    for k in self.config['topk']:
                        result_dict[f"ndcg@{k}"] += NDCG(rank_list, label_list, k)
                        result_dict[f"mrr@{k}"] += MRR(rank_list, k)
                        result_dict[f"hr@{k}"] += HR(rank_list, label_list, k)

                        ######count the exposures for the computing fairness degree#############
                        ids = ranked_args[:k]
                        rank_items = np.array(real_item_ids)[ids]
                        #print(rank_items)
                        #print(ids)
                        #exit(0)
                        for iid in rank_items:
                            group_id = self.iid2pid[iid]
                            exposure_dict[f"top@{k}"][group_id] += 1




        for k in self.config['topk']:
            #print(exposure_dict[f"top@{k}"])
            result_dict[f"mmf@{k}"] = MMF(exposure_dict[f"top@{k}"], ratio=self.config['mmf_eval_ratio']) * index
            result_dict[f"gini@{k}"] = Gini(exposure_dict[f"top@{k}"]) * index
            result_dict[f"entropy@{k}"] = Entropy(exposure_dict[f"top@{k}"]) * index


        for key in result_dict.keys():
            result_dict[key] = np.round(result_dict[key]/index, self.config['decimals'])

        if store_scores == False:
            return result_dict
        else:
            return result_dict, coo_matrix((data, (row, col)), shape=(index, self.config['item_num']))

