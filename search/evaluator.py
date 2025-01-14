import os
import numpy as np
import pandas as pd
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import pickle
import Graph4DIV
import torch
from data_process import *

MAXDOC = 50
REL_LEN = 18


def adjust_graph(A, rel_score_list, degree_tensor, selected_doc_id):
    '''
    adjust adjancent matrix A during the testing process, set the selected doc degree = 0
    :param rel_score_list: initial relevance of the document
    :param degree_tensor: degree tensor of each document
    :return: adjacent matrix A, degree tensor
    '''
    ''' connect selected document to the query node '''
    A[0, selected_doc_id+1, 0] = rel_score_list[selected_doc_id]
    A[0, 0, selected_doc_id+1] = rel_score_list[selected_doc_id]
    ''' remove edges between selected document and candidates '''
    A[0, selected_doc_id+1, 1:] = torch.tensor([0.0]*50).float()
    A[0, 1:, selected_doc_id+1] = torch.tensor([0.0]*50).float()
    ''' set the degree of selected document '''
    degree_tensor[0, selected_doc_id] = torch.tensor(0.0)
    return A, degree_tensor

        
def evaluate_accuracy(y_pred, y_label):
    num = len(y_pred)
    all_acc = 0.0
    count = 0
    for i in range(num):
        pred = (y_pred[i] > 0.5).astype(int)
        label = y_label[i]
        acc = 1 if pred == label else 0
        all_acc += acc
        count += 1
    return all_acc / count


def get_metric_nDCG_random_graph4div(model, test_tuple, div_q, qid):
    '''
    get the alpha-nDCG for the input query, the input document list are randomly shuffled. 
    :param test_tuple: the features of the test query qid, test_turple = (feature, index, rel_feat, rel_score, A, degree)
    :param div_q: the div_query object of the test query qid
    :param qid: the id for the test query
    :return: the alpha-nDCG for the test query
    '''
    metric = 0
    end = Max_doc_num = len(div_q.best_docs_rank)
    current_docs_rank = []
    if not test_tuple:
        return 0 
    else:
        feature = test_tuple[0]
        index = test_tuple[1]
        rel_feat_tensor = torch.tensor(test_tuple[2]).float()
        rel_score_list = test_tuple[3]
        A = test_tuple[4]
        degree_tensor = test_tuple[5]

        A.requires_grad = False
        degree_tensor.requires_grad = False
        rel_feat_tensor.requires_grad = False
        lt = len(rel_score_list)
        if lt < MAXDOC:
            rel_score_list.extend([0.0]*(MAXDOC-lt))
        rel_score = torch.tensor(rel_score_list).float()
        
        A = A.reshape(1, A.shape[0], A.shape[1])
        feature = feature.reshape(1, feature.shape[0], feature.shape[1])
        rel_feat_tensor = rel_feat_tensor.reshape(1, rel_feat_tensor.shape[0], rel_feat_tensor.shape[1])
        degree_tensor = degree_tensor.reshape(1, degree_tensor.shape[0], degree_tensor.shape[1])
        
        if th.cuda.is_available():
            A = A.cuda()
            feature = feature.cuda()
            rel_feat_tensor = rel_feat_tensor.cuda()
            degree_tensor = degree_tensor.cuda()
        
        while len(current_docs_rank)<Max_doc_num:
            outputs = model(A, feature, rel_feat_tensor, degree_tensor)
            out = outputs.cpu().detach().numpy()
            result = np.argsort(-out[:end])

            for i in range(len(result)):
                if result[i] < Max_doc_num and index[result[i]] not in current_docs_rank:
                    current_docs_rank.append(index[result[i]])
                    adjust_index = result[i]
                    break
            A, degree_tensor = adjust_graph(A, rel_score, degree_tensor, adjust_index)

        if len(current_docs_rank)>0:
            new_docs_rank = [div_q.doc_list[i] for i in current_docs_rank]
            metric = div_q.get_test_alpha_nDCG(new_docs_rank)
    return metric


def get_metric_nDCG_random_DESA(model, test_tuple, div_q, qid):
    '''
    get the alpha-nDCG for the input query, the input document list are randomly shuffled.
    :param test_tuple: the features of the test query qid, test_turple = {}
    :param div_q: the div_query object of the test query qid
    :param qid: the id for the test query
    :return: the alpha-nDCG for the test query
    '''
    metric = 0
    end = Max_doc_num = len(div_q.best_docs_rank)
    current_docs_rank = []
    if not test_tuple:
        return 0
    else:
        doc_mask = test_tuple['doc2vec_mask'].unsqueeze(0) # [1,50]
        sub_mask = test_tuple['sub2vec_mask'].unsqueeze(0) # [1,10]
        doc_emb = test_tuple['doc2vec'].unsqueeze(0).float() # [1, 50, 100]
        sub_emb = test_tuple['sub2vec'].unsqueeze(0).float() # [1,10,100]
        pos_qrel_feat = test_tuple['pos_qrel_feat'].unsqueeze(0).float() # [1,50,18]
        subrel_feat_mask = test_tuple['subrel_feat_mask'].unsqueeze(0)
        pos_subrel_feat = test_tuple['pos_subrel_feat'].unsqueeze(0).float() # [1,50,10,18]

        doc_mask.requires_grad = False
        sub_mask.requires_grad = False
        doc_emb.requires_grad = False
        sub_emb.requires_grad = False
        pos_qrel_feat.requires_grad = False
        subrel_feat_mask.requires_grad = False
        pos_subrel_feat.requires_grad = False

        if th.cuda.is_available():
            doc_mask, sub_mask, doc_emb, sub_emb, pos_qrel_feat, subrel_feat_mask, pos_subrel_feat =\
                doc_mask.cuda(), sub_mask.cuda(), doc_emb.cuda(), sub_emb.cuda(), pos_qrel_feat.cuda(), \
                subrel_feat_mask.cuda(), pos_subrel_feat.cuda()
        #print(doc_emb.shape, sub_emb.shape, doc_mask.shape, sub_mask.shape, pos_qrel_feat.shape, pos_subrel_feat.shape)
        score = model(doc_emb, sub_emb, doc_mask, sub_mask, pos_qrel_feat, pos_subrel_feat, mode='Test')
        result = list(np.argsort(score[:len(test_tuple['doclist'])].cpu().detach().numpy()))
        if len(result) > 0:
            new_docs_rank = []
            for i in range(len(result)-1, -1, -1):
                new_docs_rank.append(test_tuple['doclist'][result[i]])
            #new_docs_rank = [test_tuple['doclist'][result[i]] for i in range(len(result)-1, len(result)-len(test_tuple['doclist'])-1, -1)]
            metric = div_q.get_test_alpha_nDCG(new_docs_rank)
    return metric


def evaluate_test_qids(model, test_tuple, div_q, qid, mode='metric'):
    metric = 0
    end = Max_doc_num = len(div_q.best_docs_rank)
    current_docs_rank = []
    if test_tuple[0].shape[0] == 0:
        if mode == 'metric':
            return 0 
        else:
            return []
    else:
        X = test_tuple[0]
        rel_feat = test_tuple[1]
        X.requires_grad = False
        rel_feat.requires_grad = False
        X = X.reshape(1, X.shape[0], X.shape[1])
        rel_feat = rel_feat.reshape(1, rel_feat.shape[0], rel_feat.shape[1])
        
        if th.cuda.is_available():
            X = X.cuda()
            rel_feat = rel_feat.cuda()
        
        outputs = model(X, rel_feat, False)
        out = outputs.cpu().detach().numpy().reshape(MAXDOC)
        # print('out.shape = ',out.shape)
        # print('out = ', out)
        result = np.argsort(-out[:end])
        # print('result =', result)

        for i in range(len(result)):
            if result[i] < Max_doc_num and result[i] not in current_docs_rank:
                current_docs_rank.append(result[i])

        if len(current_docs_rank)>0:
            new_docs_rank = [div_q.doc_list[i] for i in current_docs_rank]
            metric = div_q.get_test_alpha_nDCG(new_docs_rank)
            # print('qid = {}, metric = {}, mode = {}'.format(qid, metric, mode))
            if mode == 'metric':
                return metric
            elif mode == 'both':
                return metric, new_docs_rank


def get_metrics_20(csv_file_path):
    all_qids=range(1,201)
    del_index=[94,99]
    all_qids=np.delete(all_qids,del_index)
    qids=[str(i) for i in all_qids]

    df=pd.read_csv(csv_file_path)

    alpha_nDCG_20=df.loc[df['topic'].isin(qids)]['alpha-nDCG@20'].mean()
    NRBP_20=df.loc[df['topic'].isin(qids)]['NRBP'].mean()
    ERR_IA_20=df.loc[df['topic'].isin(qids)]['ERR-IA@20'].mean()
    # Pre_IA_20=df.loc[df['topic'].isin(qids)]['P-IA@20'].mean()
    S_rec_20=df.loc[df['topic'].isin(qids)]['strec@20'].mean()
    
    
    return alpha_nDCG_20, NRBP_20, ERR_IA_20, S_rec_20


def get_global_fullset_metric(logger, best_model_list, test_qids_list, dump_dir):
    '''
    get the final metrics for the five fold best models.
    :param best_model_list: the best models for the five corresponding folds.
    :param test_qids_list: the corresponding test qids for five folds.
    :param dump_dir: the document ranking output dir.
    '''
    output_file = dump_dir + 'run'
    fout = open(output_file, 'w')
    all_models = best_model_list

    
    qd = pickle.load(open('../data/div_query.data', 'rb'))
    std_metric = []

    ''' get the metrics for five folds '''
    for i in range(len(all_models)):
        test_qids = test_qids_list[i]
        test_dataset_dict = get_test_dataset(logger, i+1, test_qids)
        model_file = all_models[i]
        model = DALETOR(0.0)
        model.load_state_dict(th.load(model_file))

        model.eval()
        if th.cuda.is_available():
            model = model.cuda()

        ''' ndeval test '''
        for qid in test_qids:
            metric, docs_rank = evaluate_test_qids(model, test_dataset_dict[str(qid)], qd[str(qid)], str(qid), 'both')
            if len(docs_rank)>0:
                for index in range(len(docs_rank)):
                    content = str(qid) + ' Q0 ' + str(docs_rank[index]) + ' ' + str(index+1) + ' -4.04239 indri\n'
                    fout.write(content)
    fout.close()
    csv_path = dump_dir+'result.csv'
    command = '../eval/ndeval ../eval/2009-2012.diversity.ndeval.qrels ' + output_file + ' >' + str(csv_path)
    os.system(command)
    
    alpha_nDCG_20, NRBP_20, ERR_IA_20, S_rec_20 = get_metrics_20(csv_path)
    logger.info('alpha_nDCG@20_std = {}, NRBP_20 = {}, ERR_IA_20 = {}, S_rec_20 = {}'.format(alpha_nDCG_20, NRBP_20, ERR_IA_20, S_rec_20))

