import torch
from tqdm import tqdm, trange
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from .metric import NDCG, HR, MRR, AUC_score, MMF, Gini, Entropy
import os
import json
from .utils import Build_Adjecent_Matrix
from .metric import *




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
        result_dict.update({f"maxminratio@{k}": 0 for k in self.config['topk']})
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
            result_dict[f"maxminratio@{k}"] = MinMaxRatio(exposure_dict[f"top@{k}"]) * index


        for key in result_dict.keys():
            result_dict[key] = np.round(result_dict[key]/index, self.config['decimals'])

        if store_scores == False:
            return result_dict
        else:
            return result_dict, coo_matrix((data, (row, col)), shape=(index, self.config['item_num']))



class LLM_Evaluator(Abstract_Evaluator):
    def __init__(self, config):
        super().__init__(config=config)
        self.topk_list = config['topk']

    def get_data(self, data):
        # ground_truths = [i['positive_items'] for i in data]
        # sens_feat = [i['sensitiveAttribute'] for i in data]
        label_lists = []
        # ranking_lists = []
        score_lists = []
        predict_lists = []
        for user in data:
            p = user['predict_list']
            predict_lists.append(p)
            label_list = [1 if m in user['positive_items'] else 0 for m in p]
            # label_list = [1 if m in user['positive_items'] else 0 for m in user['item_candidates']]
            score = user['scores']
            score_lists.append(score)
            label_lists.append(label_list)
            # ranking_lists.append(ranking_list)

        return predict_lists, label_lists, score_lists

    def get_cates_value(self, iid2pid, predict, topk):
        cates_name = self.get_categories(iid2pid)
        predict = [i[:topk] for i in predict]
        from collections import defaultdict
        cates_count = defaultdict(int)
        for p in predict:
            for prediction in p:
                c = iid2pid.get(prediction, -1)
                cates_count[c] += 1  # not score-based scores[idx][k]
        values = [cates_count[i] for i in cates_name]
        return values

    def cal_acc_score(self, label_lists, score_lists, topk):
        score = {}
        ndcgs = []
        hrs = []
        mrrs = []
        for lab, sco in zip(label_lists, score_lists):
            ndcg = NDCG(lab, lab, topk)
            hr = HR(lab, lab, topk)
            mrr = MRR(lab, topk)
            ndcgs.append(ndcg)
            hrs.append(hr)
            mrrs.append(mrr)

        # compute metrics
        score[f'NDCG@{topk}'] = np.round(np.mean(ndcgs), 4)
        score[f'HR@{topk}'] = np.round(np.mean(hrs), 4)
        score[f'MRR@{topk}'] = np.round(np.mean(mrrs), 4)
        return score

    def get_categories(self, iid2pid):
        return list(set(iid2pid.values()))

    def cal_fair_score(self, iid2pid, predict, topk):
        # provider fairness的评估指标 exposure
        score = {}
        cates_value = self.get_cates_value(iid2pid, predict, topk)
        # print(cates_value)
        mmf = MMF(cates_value)
        cate_gini = Gini(cates_value)
        maxmin_ratio = MinMaxRatio(cates_value)
        # cv = (cates_value)
        entropy = Entropy(cates_value)
        score[f'MMF@{topk}'] = np.round(mmf, 4)
        score[f'Gini@{topk}'] = np.round(cate_gini, 4)
        score[f'MMR@{topk}'] = np.round(maxmin_ratio, 4)
        # score[f'cv@{topk}'] = np.round(cv, 4)
        score[f'Entropy@{topk}'] = np.round(entropy, 4)
        return score

    def llm_eval(self, grounding_result, iid2pid):
        predict_lists, label_lists, score_lists = self.get_data(grounding_result)
        eval_result = {}
        for topk in self.topk_list:
            acc_score = self.cal_acc_score(label_lists, score_lists, topk)
            fair_score = self.cal_fair_score(iid2pid, predict_lists, topk)
            acc_score.update(fair_score)
            eval_result.update({f'Top{topk}': acc_score})
        print(f'Evaluate_result:{eval_result}')
        return eval_result