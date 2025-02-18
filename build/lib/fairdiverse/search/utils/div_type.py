import os
import pickle
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

pd.set_option('display.max_rows', 2000)


class subtopic:
    def __init__(self, subtopic_id, subtopic):
        """
        Represents a subtopic of a query.

        :param subtopic_id: Unique identifier for the subtopic.
        :param subtopic: Text representation of the subtopic.
        """
        self.subtopic_id = subtopic_id
        self.subtopic = subtopic


class div_query:
    def __init__(self, qid, query, subtopic_id_list, subtopic_list):
        """
        Represents a diversity query for re-ranking search results.
        
        :param qid: Unique query identifier.
        :param query: Text of the query.
        :param subtopic_id_list: List of subtopic IDs associated with the query.
        :param subtopic_list: List of subtopic texts corresponding to the subtopic IDs.
        """
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
        """
        Sets the standard alpha-DCG metric for normalization.
        
        :param m: Standard alpha-DCG metric value.
        """
        self.stand_alpha_DCG = m

    def add_docs(self, doc_list):
        """
        Adds a list of documents to the query and initializes subtopic relevance tracking.
        
        :param doc_list: List of document identifiers.
        """
        self.doc_list = doc_list
        self.DOC_NUM = len(self.doc_list)
        init_data = np.zeros((len(doc_list), len(self.subtopic_list)), dtype=int)
        self.subtopic_df = pd.DataFrame(init_data, columns=self.subtopic_id_list, index=doc_list)

    def add_query_suggestion(self, query_suggestion):
        """
        Adds query suggestions related to the main query.
        
        :param query_suggestion: Suggested query string.
        """
        self.query_suggestion = query_suggestion

    def add_docs_rel_score(self, doc_score_list):
        """
        Adds relevance scores for the documents associated with the query.
        
        :param doc_score_list: List of relevance scores for documents.
        """
        self.doc_score_list = doc_score_list

    def get_test_alpha_nDCG(self, docs_rank):
        """
        Get the alpha_nDCG@20 for the input document list (for testing).
        
        :param docs_rank: Ordered list of document identifiers.
        :return: Alpha-nDCG score for the given ranking.
        """
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
        """
        Computes the alpha-DCG for the input document list (for generating training samples)
        
        :param docs_rank: A list of document IDs representing the ranking order.
        :param print_flag: A boolean flag indicating whether to print intermediate computation results.
        :return: The computed alpha-DCG score for the given document ranking.
        """

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
        """
        Generates the best document ranking using a greedy selection strategy.
        
        :param top_n: The number of top documents to be selected (default: all available documents).
        :param alpha: A parameter controlling redundancy reduction (default: 0.5).
        :return: Updates class attributes with the best document ranking and associated gains.
        """

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
        """
        Initializes the dataset object with file paths and configuration. 

        :param config: A dictionary containing configuration settings.
        """
        self.Best_File = os.path.join(config['data_dir'], 'div_query.data')
        self.Train_File = os.path.join(config['data_dir'], config['model'], 'listpair_train.data')
        if not os.path.exists(os.path.join(config['data_dir'], config['model'])):
            os.makedirs(os.path.join(config['data_dir'], config['model']))
        self.config = config

    def get_listpairs(self, div_query, context, top_n):
        """
        Generates list-pair samples
        
        :param div_query: The query object that contains the list of ranked documents.
        :param context: A list of previously considered documents in the context.
        :param top_n: The number of top-ranked documents to consider.
        :return: A list of generated samples, each containing metrics, positive/negative masks, and weights. 
        """
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
        """
        Generates list-pair training samples using the top N relevant documents. 
        This function processes the best document ranks for each query, generates list-pair samples, and saves them to a file: listpair_train.data. 
        data_dict[qid] = [(metrics, positive_mask, negative_mask, weight),...]
        metrics, positive_mask and negative_mask are padding as tensors with length of top_n

        :param top_n: The number of top-ranked documents to use for generating the list-pairs.
        :return: Saves the generated list-pair training data into a file. 
        """
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
