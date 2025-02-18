import os
import torch
import pickle
import random
import numpy as np
import multiprocessing

from tqdm import tqdm

from .utils import load_embedding, get_rel_feat, pkl_save, pkl_load, split_list


MAXDOC=50
MAX_DIV_DIM = 8
REL_LEN=18

def build_each_train_dataset(qid_list, qd, train_dict, rel_feat_dict, res_dir, query_emb, doc_emb):
    """
    Generates and saves the training dataset for each query in qid_list. 
    
    :param qid_list: A list of query IDs to process.
    :param qd: A dictionary containing the query data (query, document list, subtopic info).
    :param train_dict: A dictionary of pre-generated training samples.
    :param rel_feat_dict: A dictionary of relevance features for query-document pairs.
    :param res_dir: The directory where the processed data will be saved.
    :param query_emb: A dictionary of query embeddings.
    :param doc_emb: A dictionary of document embeddings.
    :return: Saves the processed data for each query in a .pkl.gz file. 
    """

    for qid in tqdm(qid_list, desc="GenTrainData", ncols = 90):
        sample_dict={}
        sample_path = res_dir + str(qid) + '.pkl.gz'

        sample_feat_list = []
        sample_rel_list = []
        sample_div_list = []
        query = qd[str(qid)].query
        Doc_list = qd[str(qid)].doc_list
        df = np.array(qd[qid].subtopic_df)
        sample_list = train_dict[str(qid)]
        count = -1
        while count < len(sample_list):
            if count == -1:
                doc_list = qd[str(qid)].best_docs_rank
            else:
                doc_list = sample_list[count]
            
            rel_feat_list = []
            div_labels = []
            feat_list = []
            feat_list.append(torch.tensor(query_emb[query]).float())

            for i in range(len(doc_list)):
                rel_feat = rel_feat_dict[(query, doc_list[i])]
                rel_feat_list.append(torch.tensor(rel_feat).float())
                doc_feat = doc_emb[doc_list[i]]
                feat_list.append(torch.tensor(doc_feat).float())
                index = Doc_list.index(doc_list[i])
                div_feat = list(df[index, :])
                if len(div_feat) < MAX_DIV_DIM:
                    div_feat.extend([0]*(MAX_DIV_DIM - len(div_feat)))
                div_labels.append(torch.tensor(div_feat).float())
            
            if len(feat_list) < (MAXDOC+1):
                feat_list.extend([torch.tensor([0]*100).float()]*(MAXDOC+1-len(feat_list)))
            if len(div_labels) < MAXDOC:
                div_labels.extend([torch.tensor([0]*MAX_DIV_DIM).float()]*(MAXDOC-len(div_labels)))
            if len(rel_feat_list) < MAXDOC:
                rel_feat_list.extend([torch.tensor([0]*REL_LEN).float()]*(MAXDOC-len(rel_feat_list)))
            
            feat_tensor = torch.stack(feat_list, dim=0).float()
            rel_feat = torch.stack(rel_feat_list, dim=0).float()
            div_tensor = torch.stack(div_labels, dim=0).float()

            assert feat_tensor.shape[0] == (MAXDOC+1)
            assert rel_feat.shape[0] == (MAXDOC)
            assert div_tensor.shape[0] == MAXDOC
            if div_tensor.shape[1] != MAX_DIV_DIM:
                print('qid = {}, len={}'.format(qid, div_tensor.shape[1]))
            assert div_tensor.shape[1] == MAX_DIV_DIM
            sample_feat_list.append(feat_tensor)
            sample_rel_list.append(rel_feat)
            sample_div_list.append(div_tensor)
            count += 1
        
        assert len(sample_feat_list) == len(sample_div_list)
        assert len(sample_rel_list) == len(sample_div_list)
        sample_dict[qid]=[
            (sample_feat_list[i],
            sample_rel_list[i],
            sample_div_list[i])
            for i in range(len(sample_feat_list))
        ]
        pkl_save(sample_dict, sample_path)


def build_train_dataset(config, worker_num=20):
    """
    Builds the training dataset by distributing the workload across multiple workers. 
    
    :param config: A dictionary containing configuration settings.
    :param worker_num: The number of workers to use for parallel processing.
    :return: Generates the training dataset and saves it in the result directory.
    """
    res_dir=os.path.join(config['data_dir'], config['model'], 'train/')
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    all_qids=np.load(os.path.join(config['data_dir'], 'all_qids.npy'))
    qd = pickle.load(open(os.path.join(config['data_dir'], 'div_query.data'), 'rb'))
    doc_emb = load_embedding(os.path.join(config['data_dir'], config['embedding_dir'], config['embedding_type']+'_doc.emb'))
    query_emb = load_embedding(os.path.join(config['data_dir'], config['embedding_dir'], config['embedding_type']+'_query.emb'))
    train_dict = pkl_load(os.path.join(config['data_dir'], config['model'], 'list_train_samples.pkl.gz'))
    rel_feat_dict = get_rel_feat(os.path.join(config['data_dir'], 'rel_feat.csv'))

    task_list = split_list(all_qids, worker_num)
    jobs=[]
    for task in task_list:
        p = multiprocessing.Process(target=build_each_train_dataset, args=(task, qd, train_dict, rel_feat_dict, res_dir, query_emb, doc_emb))
        jobs.append(p)
        p.start()
    for job in jobs:
        job.join()


def build_test_dataset(config):
    """
    Builds the test dataset for evaluation. 
    
    :param config: A dictionary containing configuration settings.
    :return: Generates the test dataset and saves it in the result directory.
    """
    qd = pickle.load(open(os.path.join(config['data_dir'], 'div_query.data'), 'rb'))
    doc_emb = load_embedding(os.path.join(config['data_dir'], config['embedding_dir'], config['embedding_type']+'_doc.emb'))
    query_emb = load_embedding(os.path.join(config['data_dir'], config['embedding_dir'], config['embedding_type']+'_query.emb'))
    output_dir = os.path.join(config['data_dir'], config['model'], 'test/')
    rel_feat_dict = get_rel_feat(os.path.join(config['data_dir'], 'rel_feat.csv'))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    all_qids=range(1,201)
    del_index=[94,99]
    all_qids=np.delete(all_qids,del_index)
    qids=[str(i) for i in all_qids]
    for qid in tqdm(qids,desc="gen Test", ncols=80):
        print('qid=',qid)
        test_dict = {}
        query = qd[str(qid)].query
        output_file_path = output_dir + str(qid) + '.pkl.gz'
        doc_list = qd[str(qid)].doc_list[:50]
        real_num = len(doc_list)
        
        feat_list = []
        rel_feat_list = []

        feat_list.append(torch.tensor(query_emb[query]).float())

        for i in range(len(doc_list)):
            doc_feat = doc_emb[doc_list[i]]
            feat_list.append(torch.tensor(doc_feat).float())
            rel_feat = torch.tensor(rel_feat_dict[(query, doc_list[i])]).float()
            rel_feat_list.append(rel_feat)

        if len(feat_list) < (MAXDOC+1):
            feat_list.extend([torch.tensor([0]*100).float()]*(MAXDOC+1-len(feat_list)))
        if len(rel_feat_list) < MAXDOC:
            rel_feat_list.extend([torch.tensor([0]*REL_LEN).float()]*(MAXDOC-len(rel_feat_list)))
        
        feat_tensor = torch.stack(feat_list, dim=0).float()
        rel_feat_tensor = torch.stack(rel_feat_list, dim=0).float()
        assert feat_tensor.shape[0] == (MAXDOC+1)
        assert rel_feat_tensor.shape[0] == MAXDOC

        test_dict[qid]=(
            feat_tensor,
            rel_feat_tensor
        )
        pkl_save(test_dict, output_file_path)


def gen_list_training_sample(config, top_n = 50, sample_num = 200):
    """
    Generates list training samples by selecting top-ranked documents for each query. 

    :param config: A dictionary containing configuration settings.
    :param top_n: The number of top-ranked documents to consider for each sample.
    :param sample_num: The number of samples to generate for each query.
    :return: Saves the generated training samples in a file for later use.
    """
    qd = pickle.load(open(os.path.join(config['data_dir'], 'div_query.data'),'rb'))
    doc_emb = load_embedding(os.path.join(config['data_dir'], config['embedding_dir'], config['embedding_type']+'_doc.emb'))
    rel_feat_dict = get_rel_feat(os.path.join(config['data_dir'], 'rel_feat.csv'))
    train_dict={}
    for qid in tqdm(qd, desc="Gen Train"):
        temp_q=qd[qid]
        temp_doc_list = temp_q.doc_list[:100]
        result_list=[]
        real_num=int(min(top_n, temp_q.DOC_NUM))
        for i in range(sample_num):
            random.shuffle(temp_doc_list)
            top_docs = temp_doc_list[:real_num]
            flag = 0
            for j in range(len(top_docs)):
                if (qid, top_docs[j]) not in rel_feat_dict:
                    flag = 1
                    break
            if flag == 0 and top_docs not in result_list:
                result_list.append(top_docs)
        print('qid={}, len={}'.format(qid, len(result_list)))
        train_dict[str(qid)]=result_list
    if not os.path.exists(os.path.join(config['data_dir'], config['model'])):
            os.makedirs(os.path.join(config['data_dir'], config['model']))
    pkl_save(train_dict, os.path.join(config['data_dir'], config['model'], 'list_train_samples.pkl.gz'))


def Process(config):
    """ Main function for DALETOR data processing.
    
    :param config: A dictionary containing configuration settings.
    :return: Generates and saves the training and test datasets, as well as list training samples.
    """
    gen_list_training_sample(config)
    build_train_dataset(config)
    build_test_dataset(config)