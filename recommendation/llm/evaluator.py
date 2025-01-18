
from ..metric import *

class Evaluator(object):
    def __init__(self, metric_list, TopK):
        self.TopK = TopK
        self.metric_list = metric_list

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

    def get_cates_value(self, iid2pid, predict):
        cates_name = self.get_categories(iid2pid)
        predict = [i[:self.TopK] for i in predict]
        from collections import defaultdict
        cates_count = defaultdict(int)
        for p in predict:
            for prediction in p:
                c = iid2pid.get(prediction, -1)
                cates_count[c] += 1  # not score-based scores[idx][k]
        values = [cates_count[i] for i in cates_name]
        return values

    def cal_acc_score(self, label_lists, score_lists):
        topk = self.TopK
        score = {}
        ndcgs = []
        hrs = []
        mrrs = []
        aucs = []
        for lab, sco in zip(label_lists, score_lists):
            ndcg = NDCG(lab, lab, topk)
            hr = HR(lab, lab, topk)
            mrr = MRR(lab, topk)
            auc = AUC_score(sco, lab)
            ndcgs.append(ndcg)
            hrs.append(hr)
            mrrs.append(mrr)
            aucs.append(auc)
        # 计算accuracy指标
        score[f'NDCG@{topk}'] = np.round(np.mean(ndcgs), 4)
        score[f'HR@{topk}'] = np.round(np.mean(hrs), 4)
        score[f'MRR@{topk}'] = np.round(np.mean(mrrs), 4)
        score[f'AUC@{topk}'] = np.round(np.mean(aucs), 4)
        return score

    def get_categories(self, iid2pid):
        return list(set(iid2pid.values()))

    def cal_fair_score(self, iid2pid, predict):
        # provider fairness的评估指标 exposure
        topk = self.TopK
        score = {}
        cates_value = self.get_cates_value(iid2pid, predict)
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

    def evaluate(self, grounding_result, iid2pid):
        predict_lists, label_lists, score_lists = self.get_data(grounding_result)  # 返回二维列表，所有用户的predict 和scores
        acc_score = self.cal_acc_score(label_lists, score_lists)  # sens_feat 代表每一个prompt的敏感属性list
        fair_score = self.cal_fair_score(iid2pid, predict_lists)
        acc_score.update(fair_score)
        print(f'Evaluate_result:{acc_score}')
        return acc_score