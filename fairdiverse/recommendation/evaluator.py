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
        """
            Evaluates the model on the provided dataloader and calculates performance metrics.


            :param dataloader: The data loader that provides batches of user-item interactions and corresponding labels.
            :param model: The model to evaluate.
            :param store_scores: Whether to return the predicted scores as a sparse matrix. Defaults to `False`.
            :return: A dictionary containing the evaluation metric(s) (e.g., AUC score).
        """
        pass


class CTR_Evaluator(Abstract_Evaluator):
    def __int__(self,config):
        super().__init__(config=config)

    def eval(self, dataloader, model, store_scores = False):
        """
            Evaluates the model on the provided dataloader and calculates performance metrics.

            This function runs the evaluation on a dataset using the provided model. It calculates the AUC score based on
            the predicted scores and ground truth labels. If `store_scores` is set to `True`, it also returns the evaluation
            results as a sparse matrix of predicted scores.

            :param dataloader: The data loader that provides batches of user-item interactions and corresponding labels.
            :param model: The model to evaluate.
            :param store_scores: Whether to return the predicted scores as a sparse matrix. Defaults to `False`.
            :return: A dictionary containing the evaluation metric(s) (e.g., AUC score), and optionally, a sparse matrix
                     of predicted scores.
        """
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
        """
            Evaluates the model on the provided dataloader and calculates performance metrics.

            This function runs the evaluation on a dataset using the provided model. It calculates the Ranking metrics based on
            the predicted scores and ground truth labels. If `store_scores` is set to `True`, it also returns the evaluation
            results as a sparse matrix of predicted scores.

            :param dataloader: The data loader that provides batches of user-item interactions and corresponding labels.
            :param model: The model to evaluate.
            :param store_scores: Whether to return the predicted scores as a sparse matrix. Defaults to `False`.
            :return: A dictionary containing the evaluation metric(s), and optionally, a sparse matrix
                     of predicted scores.
        """
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
                    # print(row)
                    # print(col)
                    # has_duplicates = len(col) != len(set(col))
                    # print("是否有重复元素:", has_duplicates)
                    # exit(0)
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
                    score = score + 1.0
                    # print(score)
                    # print(score.shape)
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
                        for i, iid in enumerate(rank_items):
                            group_id = self.iid2pid[iid]
                            if self.config['fairness_type'] == "Exposure":
                                exposure_dict[f"top@{k}"][group_id] += 1
                            else:
                                exposure_dict[f"top@{k}"][group_id] += np.round(score[ids[i]], self.config['decimals'])
                            #exposure_dict[f"top@{k}"][group_id] += 1




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
        """
        This method processes the input data to extract prediction lists, label lists, and score lists for each user.

        :param 'predict_list': A list of predicted items.
        :param 'positive_items': A list of items that are considered positive (e.g., liked or preferred by the user).
        :param 'scores': A list of scores corresponding to the predicted items, indicating the confidence of the prediction.

        :return:
        - `predict_lists`: A list of predict lists for all users.
        - `label_lists`: For each user, a list of binary labels indicating whether each predicted item is positive (1) or not (0).
        - `score_lists`: A list of score lists corresponding to the predicted items for all users.
        """
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
        """
        Get the category values based on predicted indices and their corresponding categories.

        This method processes the predicted indices along with their mapping to category IDs
        and returns a list of counts for each category, representing the frequency of occurrence
        in the top-k predictions.

        :param iid2pid : dict
            A dictionary mapping item indices (int) to their respective category IDs (int).
            If an item index is not found in the dictionary, it defaults to -1.

        :param predict : List[List[int]]
            A 2D list where each sublist contains the predicted indices (top-k predictions)
            for corresponding input data points.

        :param topk : int
            The number of top predictions considered for each data point. This determines
            how many elements from the beginning of each sublist in `predict` are processed.

        :return: List[int]
            A list of integers where each value corresponds to the count of occurrences for
            a specific category across all top-k predictions. The order of these counts matches
            the sorted order of category IDs as returned by `get_categories(iid2pid)`.
        """
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
        """
        Calculate accuracy scores for recommendation system evaluation.

        This method computes the average NDCG (Normalized Discounted Cumulative Gain), HR (Hit Ratio), and MRR (Mean Reciprocal Rank)
        at a specified `topk` cutoff for a list of ground-truth labels and corresponding prediction scores.

        :param label_lists : List[List[int]]
            A list of lists containing ground-truth labels. Each sublist represents the relevant items for a user or query.
        :param score_lists : List[List[float]]
            A list of lists containing predicted scores. Each sublist corresponds to the relevance scores for items
            matching the order in `label_lists`.
        :param topk : int
            The number of top predictions to consider when calculating the metrics.

        :return: Dict[str, float]
            A dictionary containing the average NDCG, HR, and MRR scores at the given `topk`, with keys formatted as 'NDCG@{topk}',
            'HR@{topk}', and 'MRR@{topk}' respectively. Scores are rounded to 4 decimal places.
        """
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
        """
        Calculate fairness scores for a recommendation system's evaluation.

        This method computes various fairness metrics at a specified top-k cutoff to evaluate
        the diversity and inclusiveness of the predicted items. It utilizes different fairness
        measures like MMF (Max-Min Fairness), Gini coefficient, Min-Max Ratio, and Entropy to
        quantify the balance across different categories or groups within the predictions.

        :param iid2pid (Dict[int, int]): A mapping where keys are item IDs and values are their respective group/category IDs.
        :param predict (List[Tuple[int, float]]): A list of tuples, each containing an item ID and its predicted score/score.
        :param topk (int): The top-k count used to consider the highest scored items for fairness evaluation.

        :return:
        A dictionary with keys as the metric names prefixed with the top-k cutoff (e.g., 'MMF@5') and values as the
        corresponding calculated scores, rounded to 4 decimal places.
        """
        #
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
        """
        Evaluate the performance of a language model based on grounding results and item-pid mappings.

        This method assesses the accuracy and fairness of the model's predictions at different top-K thresholds.
        It computes both accuracy scores and fairness scores,汇总 these into a comprehensive evaluation result.

        :param grounding_result (Dict[str, Any]): The output from the model grounding process, containing necessary information for evaluation.
        :param iid2pid (Dict[str, str]): A mapping from item IDs to product IDs, used in calculating fairness metrics.

        :return:
        - eval_result (Dict[str, float]): A dictionary summarizing the evaluation outcomes, including accuracy and fairness scores for each specified top-K value.
        """
        predict_lists, label_lists, score_lists = self.get_data(grounding_result)
        eval_result = {}
        for topk in self.topk_list:
            acc_score = self.cal_acc_score(label_lists, score_lists, topk)
            fair_score = self.cal_fair_score(iid2pid, predict_lists, topk)
            eval_result.update(acc_score)
            eval_result.update(fair_score)
            # eval_result.update({f'Top{topk}': acc_score})
        print(f'Evaluate_result:{eval_result}')
        return eval_result