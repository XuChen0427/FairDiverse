import os
import pickle
import gzip
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

pd.set_option('display.max_rows', 2000)


def pkl_load(filename):
    if not os.path.exists(filename):
        print('filename={} not exists!')
        return
    with gzip.open(filename, 'rb') as f:
        data_dict = pickle.load(f)
    return data_dict


def pkl_save(data_dict, filename):
    with gzip.open(filename, 'wb') as f:
        pickle.dump(data_dict, f)


class subtopic:
    def __init__(self, subtopic_id, subtopic):
        self.subtopic_id = subtopic_id
        self.subtopic = subtopic


class div_query:
    def __init__(self, qid, query, subtopic_id_list, subtopic_list):
        '''
        object for diversity query
        alpha = 0.5 by default
        doc_list: the inital document ranking derived from indri
        doc_score_list: the normalized relevance score list of documents
        best_metric: the best metric of the query
        stand_alpha_DCG: stand alpha-DCG (from DSSA) used for normalization
        '''
        self.qid = qid
        self.query = query
        self.subtopic_id_list = subtopic_id_list
        self.subtopic_list = []
        self.doc_list = []
        self.doc_score_list = []
        self.best_metric = 0
        self.stand_alpha_DCG = 0

        for index in range(len(subtopic_id_list)):
            t = subtopic(subtopic_id_list[index], subtopic_list[index])
            self.subtopic_list.append(t)

    def set_std_metric(self, m):
        self.stand_alpha_DCG = m

    def add_docs(self, doc_list):
        self.doc_list = doc_list
        self.DOC_NUM = len(self.doc_list)
        init_data = np.zeros((len(doc_list), len(self.subtopic_list)), dtype=int)
        self.subtopic_df = pd.DataFrame(init_data, columns=self.subtopic_id_list, index=doc_list)

    def add_query_suggestion(self, query_suggestion):
        self.query_suggestion = query_suggestion

    def add_docs_rel_score(self, doc_score_list):
        self.doc_score_list = doc_score_list

    def get_test_alpha_nDCG(self, docs_rank):
        '''
        get the alpha_nDCG@20 for the input document list (for testing).
        '''
        temp_data = np.zeros((len(docs_rank), len(self.subtopic_list)), dtype=int)
        temp_array = np.array(self.best_subtopic_df)
        metrics = []
        p = 0.5
        real_num = min(20, len(docs_rank))
        best_docs_index = []
        for index in range(real_num):
            result_index = self.best_docs_rank.index(docs_rank[index])
            best_docs_index.append(result_index)
            temp_data[index, :] = temp_array[result_index, :]
            if index == 0:
                score = np.sum(temp_data[index, :])
                metrics.append(score)
            else:
                r_ik = np.array([np.sum(temp_data[:index, s]) for s in range(temp_data.shape[1])], dtype=np.int64)
                t = np.power(p, r_ik)
                score = np.dot(temp_data[index, :], t) / np.log2(2 + index)
                metrics.append(score)
        ''' normalized by the stand alpha DCG '''
        if hasattr(self, 'stand_alpha_DCG') and self.stand_alpha_DCG > 0:
            try:
                alpha_nDCG = np.sum(metrics) / self.stand_alpha_DCG
            except:
                print('except np.sum =', np.sum(metrics), 'self.global_best_metric = ', self.global_best_metric)
        else:
            print('error! qid =', self.qid)
            alpha_nDCG = 0
        return alpha_nDCG
    

    def get_alpha_DCG(self, docs_rank, print_flag=False):
        '''
        get the alpha-DCG for the input document list (for generating training samples)
        '''
        temp_data = np.zeros((len(docs_rank), len(self.subtopic_list)), dtype=int)
        temp_array = np.array(self.best_subtopic_df)
        metrics = []
        p = 0.5
        for index in range(len(docs_rank)):
            result_index = self.best_docs_rank.index(docs_rank[index])
            temp_data[index, :] = temp_array[result_index, :]
            if index == 0:
                score = np.sum(temp_data[index, :])
                metrics.append(score)
            else:
                r_ik = np.array([np.sum(temp_data[:index, s]) for s in range(temp_data.shape[1])], dtype=np.int64)
                t = np.power(p, r_ik)
                score = np.dot(temp_data[index, :], t) / np.log2(2 + index)
                metrics.append(score)
        if print_flag:
            print('self.best_gain = ', self.best_gain, 'sum(best_gain) = ', np.sum(self.best_gain), 'best_metric = ',
                  self.best_metric)
            print('test metrics = ', metrics, 'sum(metrics) = ', np.sum(metrics))
        '''get the total gain for the input document list'''
        alpha_nDCG = np.sum(metrics)
        return alpha_nDCG

    def get_best_rank(self, top_n=None, alpha=0.5):
        '''
        get the best diversity document ranking based on greedy strategy
        '''
        p = 1.0 - alpha
        if top_n == None:
            top_n = self.DOC_NUM
        real_num = int(min(top_n, self.DOC_NUM))
        temp_data = np.zeros((real_num, len(self.subtopic_list)), dtype=int)
        temp_array = np.array(self.subtopic_df)
        best_docs_rank = []
        best_docs_rank_rel_score = []
        best_gain = []
        ''' greedy document selection '''
        for step in range(real_num):
            scores = []
            if step == 0:
                for index in range(real_num):
                    temp_score = np.sum(temp_array[index, :])
                    scores.append(temp_score)
                result_index = np.argsort(scores)[-1]
                gain = scores[result_index]
                docid = self.doc_list[result_index]
                doc_rel_score = self.doc_score_list[result_index]
                best_docs_rank.append(docid)
                best_docs_rank_rel_score.append(doc_rel_score)
                best_gain.append(scores[result_index])
                temp_data[0, :] = temp_array[result_index, :]
            else:
                for index in range(real_num):
                    if self.doc_list[index] not in best_docs_rank:
                        r_ik = np.array([np.sum(temp_data[:step, s]) for s in range(temp_array.shape[1])],
                                        dtype=np.int64)
                        t = np.power(p, r_ik)
                        temp_score = np.dot(temp_array[index, :], t)
                        scores.append(temp_score)
                    else:
                        scores.append(-1.0)
                result_index = np.argsort(scores)[-1]
                gain = scores[result_index]
                docid = self.doc_list[result_index]
                doc_rel_score = self.doc_score_list[result_index]
                if docid not in best_docs_rank:
                    best_docs_rank.append(docid)
                    best_docs_rank_rel_score.append(doc_rel_score)
                else:
                    print('document already added!')
                best_gain.append(scores[result_index] / np.log2(2 + step))
                temp_data[step, :] = temp_array[result_index, :]
        self.best_docs_rank = best_docs_rank
        self.best_docs_rank_rel_score = best_docs_rank_rel_score
        self.best_gain = best_gain
        self.best_subtopic_df = pd.DataFrame(temp_data, columns=self.subtopic_id_list, index=self.best_docs_rank)
        self.best_metric = np.sum(self.best_gain)


class div_dataset:
    def __init__(self, config):
        self.Best_File = os.path.join(config['data_dir'], 'div_query.data')
        self.Train_File = os.path.join(config['data_dir'], config['model']+'listpair_train.data')
        self.config = config

    ''' generate list-pair training samples '''

    def get_listpairs(self, div_query, context, top_n):
        best_rank = div_query.best_docs_rank
        metrics = []
        samples = []
        for index in range(len(best_rank)):
            if best_rank[index] not in context:
                metric = div_query.get_alpha_DCG(context + [best_rank[index]])
            else:
                metric = -1.0
            metrics.append(metric)
        ''' padding the metrics '''
        if len(metrics) < top_n:
            metrics.extend([0] * (top_n - len(metrics)))
        total_count = 0
        for i in range(len(best_rank)):
            ''' set a limit to the total sample number '''
            if total_count > 20:
                break
            count = 0
            for j in range(i + 1, len(best_rank)):
                ''' set a limit to sample number on the same context'''
                if count > 5:
                    break
                if metrics[i] < 0 or metrics[j] < 0 or metrics[i] == metrics[j]:
                    pass
                elif metrics[i] > metrics[j]:
                    count += 1
                    total_count += 1
                    positive_mask = torch.zeros(top_n)
                    negative_mask = torch.zeros(top_n)
                    weight = metrics[i] - metrics[j]
                    positive_mask[i] = 1
                    negative_mask[j] = 1
                    samples.append((metrics, positive_mask, negative_mask, weight))
                elif metrics[i] < metrics[j]:
                    count += 1
                    total_count += 1
                    positive_mask = torch.zeros(top_n)
                    negative_mask = torch.zeros(top_n)
                    weight = metrics[j] - metrics[i]
                    positive_mask[j] = 1
                    negative_mask[i] = 1
                    samples.append((metrics, positive_mask, negative_mask, weight))
        return samples

    def get_listpair_train_data(self, top_n=50):
        '''
        generate list-pair trianing samples using top 50 relevant documents
        save as a data file: listpair_train.data
        data_dict[qid] = [(metrics, positive_mask, negative_mask, weight),...]
        metrics, positive_mask and negative_mask are padding as tensors with length of top_n
        '''
        qd = pickle.load(open(self.Best_File, 'rb'))
        train_dict = {}
        for qid in tqdm(qd, desc="Gen Train Data"):
            temp_q = qd[qid]
            result_list = []
            real_num = int(min(top_n, temp_q.DOC_NUM))
            for i in range(real_num):
                listpair_data = self.get_listpairs(temp_q, temp_q.best_docs_rank[:i], top_n)
                if len(listpair_data) > 0:
                    result_list.extend(listpair_data)
            train_dict[str(qid)] = result_list
        pickle.dump(train_dict, open(self.Train_File, 'wb'), True)
