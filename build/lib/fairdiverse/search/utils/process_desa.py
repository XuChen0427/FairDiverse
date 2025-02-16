import os
import pickle
import torch
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import KFold
from ..utils.utils import load_embedding, read_rel_feat
from ..utils.div_type import div_dataset


def gen_data_file_train(train_qids, qd, train_data, doc_emb, query_emb, rel_feat, save_path):
    """
    Generates and saves training data files with document embeddings, query suggestions, and relevance features.
    This function processes training query IDs to create structured data samples containing document vectors,
    query suggestion vectors, relevance features, and various masking tensors for handling variable-length inputs.
    
    :param train_qids: List of training query IDs to process
    :param qd: Dictionary containing query-document information
    :param train_data: Dictionary containing training data pairs
    :param doc_emb: Dictionary of document embeddings
    :param query_emb: Dictionary of query embeddings
    :param rel_feat: Dictionary of relevance features
    :param save_path: Path where the processed training data will be saved
    """
    data_list = []  # {qid:, query:, doclist:, doc2vec:, sub2vec:, rel_feat:, sub_rel_feat:, list_pair:}
    #max_d, max_s, max_ps, max_ns = 0, 0, 0, 0
    for qid in tqdm(train_qids):
        doc2vec = [doc_emb[docid] for docid in qd[qid].best_docs_rank]
        sub2vec = [query_emb[query_sugg] for query_sugg in qd[qid].query_suggestion]

        for i in range(len(train_data[qid])):
            '''
            max_d, max_s, max_ps, max_ns = max(max_d, len(doc2vec)), max(max_s, len(sub2vec)), \
                                           max(max_ps, len(temp['pos_subrel_feat'])), \
                                           max(max_ns, len(temp['neg_subrel_feat']))
            '''
            temp = {}
            temp['qid'] = qid
            temp['query'] = qd[qid].query
            temp['doclist'] = qd[qid].best_docs_rank
            temp['doc2vec_mask'] = torch.tensor([1]*len(doc2vec)+[0]*(50-len(doc2vec)))
            temp['sub2vec_mask'] = torch.tensor([1]*len(sub2vec)+[0]*(10-len(sub2vec)))
            temp['doc2vec'] = torch.tensor(doc2vec+[[0]*100]*(50-len(doc2vec)))
            temp['sub2vec'] = torch.tensor(sub2vec+[[0]*100]*(10-len(sub2vec)))
            temp['positive_mask'] = train_data[qid][i][1]
            temp['negative_mask'] = train_data[qid][i][2]
            temp['weight'] = train_data[qid][i][3]
            pos_id = qd[qid].best_docs_rank[int(torch.argmax(train_data[qid][i][1]))]
            neg_id = qd[qid].best_docs_rank[int(torch.argmax(train_data[qid][i][2]))]
            temp['pos_qrel_feat'] = torch.Tensor(rel_feat[qd[qid].query][pos_id])
            temp['neg_qrel_feat'] = torch.Tensor(rel_feat[qd[qid].query][neg_id])
            temp['subrel_feat_mask'] = torch.tensor([1]*len(qd[qid].query_suggestion)+[0]*(10-len(qd[qid].query_suggestion)))
            temp['pos_subrel_feat'] = torch.tensor([rel_feat[query_sugg][pos_id] for query_sugg in
                                                qd[qid].query_suggestion]+[[0]*18]*(10-len(qd[qid].query_suggestion)))
            temp['neg_subrel_feat'] = torch.tensor([rel_feat[query_sugg][neg_id] for query_sugg in
                                       qd[qid].query_suggestion]+[[0]*18]*(10-len(qd[qid].query_suggestion)))
            data_list.append(temp)
    torch.save(data_list, save_path)
    #return max_d, max_s, max_ps, max_ns


def gen_data_file_test(test_qids, qd, test_data, doc_emb, query_emb, rel_feat, save_path):
    """
    Generates and saves test data files with document embeddings, query suggestions, and relevance features.
    This function processes test query IDs to create a structured dictionary of test samples, including
    document vectors, query suggestion vectors, and relevance features with appropriate padding and masking.
    
    :param test_qids: List of test query IDs to process
    :param qd: Dictionary containing query-document information
    :param test_data: Dictionary containing test data pairs
    :param doc_emb: Dictionary of document embeddings
    :param query_emb: Dictionary of query embeddings
    :param rel_feat: Dictionary of relevance features
    :param save_path: Path where the processed test data will be saved
    """
    data_list = {}  # {qid:, query:, doclist:, doc2vec:, sub2vec:, rel_feat:, sub_rel_feat:, list_pair:}
    for qid in tqdm(test_qids):
        data_list[qid] = {}
        doc2vec = [doc_emb[docid] for docid in qd[qid].best_docs_rank]
        sub2vec = [query_emb[query_sugg] for query_sugg in qd[qid].query_suggestion]
        data_list[qid]['qid'] = qid
        data_list[qid]['query'] = qd[qid].query
        data_list[qid]['doclist'] = qd[qid].best_docs_rank
        data_list[qid]['doc2vec_mask'] = torch.tensor([1] * len(doc2vec) + [0] * (50 - len(doc2vec)))
        data_list[qid]['sub2vec_mask'] = torch.tensor([1] * len(sub2vec) + [0] * (10 - len(sub2vec)))
        data_list[qid]['doc2vec'] = torch.tensor(doc2vec + [[0] * 100] * (50 - len(doc2vec)))
        data_list[qid]['sub2vec'] = torch.tensor(sub2vec + [[0] * 100] * (10 - len(sub2vec)))
        data_list[qid]['pos_qrel_feat'] = torch.Tensor([rel_feat[qd[qid].query][pos_id]
                                    for pos_id in qd[qid].best_docs_rank]+[[0]*18]*(50-len(qd[qid].best_docs_rank)))
        pos_subrel_feat = [] # 50*10*18
        subrel_mask = []
        for pos_id in qd[qid].best_docs_rank:
            temp1 = []
            for query_sugg in qd[qid].query_suggestion:
                temp1.append(rel_feat[query_sugg][pos_id])
            temp2 = [0]*len(temp1)+[1]*(10-len(temp1))
            temp1.extend([[0]*18]*(10-len(temp1)))
            pos_subrel_feat.append(temp1)
            subrel_mask.append(temp2)
        subrel_mask.extend([[0] * 10] * (50 - len(pos_subrel_feat)))
        pos_subrel_feat.extend([[[0]*18]*10]*(50-len(pos_subrel_feat)))
        data_list[qid]['subrel_feat_mask'] = torch.tensor(subrel_mask)
        data_list[qid]['pos_subrel_feat'] = torch.tensor(pos_subrel_feat)
    torch.save(data_list, save_path)


def divide_five_fold_train_test(config):
    """
    Performs 5-fold cross-validation data splitting and preprocessing for model training and testing.
    This function loads necessary embeddings and features, splits the data into 5 folds, and generates
    processed training and test data files for each fold using the specified configuration settings.
    
    :param config: Dictionary containing configuration parameters including data directories, 
                  embedding types, and model settings
    """
    all_qids = np.load(os.path.join(config['data_dir'], 'all_qids.npy'))
    qd = pickle.load(open(os.path.join(config['data_dir'], 'div_query.data'), 'rb'))
    train_data = pickle.load(open(os.path.join(config['data_dir'], config['model'], 'listpair_train.data'), 'rb'))
    doc_emb = load_embedding(os.path.join(config['data_dir'], config['embedding_dir'], config['embedding_type']+'_doc.emb'))
    query_emb = load_embedding(os.path.join(config['data_dir'], config['embedding_dir'], config['embedding_type']+'_query.emb'))
    rel_feat = read_rel_feat(os.path.join(config['data_dir'], 'rel_feat.csv'))

    data_dir = os.path.join(config['data_dir'], config['model'], 'fold/')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    fold = 0
    for train_ids, test_ids in KFold(5).split(all_qids):
        fold += 1
        res_dir = os.path.join(data_dir, 'fold'+str(fold))
        if not os.path.exists(res_dir):
            os.makedirs(res_dir)
        train_ids.sort()
        test_ids.sort()
        train_qids = [str(all_qids[i]) for i in train_ids]
        test_qids = [str(all_qids[i]) for i in test_ids]
        '''{qid:, query:, doc2vec:, sub2vec:, rel_feat:, sub_rel_feat:, list_pair:} '''
        gen_data_file_train(train_qids, qd, train_data, doc_emb, query_emb, rel_feat, os.path.join(res_dir, 'train_data.pkl'))
        gen_data_file_test(test_qids, qd, train_data, doc_emb, query_emb, rel_feat, os.path.join(res_dir, 'test_data.pkl'))


def Process(config):
    """
    Initializes and executes the complete data processing pipeline for the model.
    This function creates a dataset object, generates list-pair training data, and performs
    5-fold cross-validation data splitting and preprocessing.
    
    :param config: Dictionary containing configuration parameters for data processing
    """
    D = div_dataset(config)
    D.get_listpair_train_data()
    divide_five_fold_train_test(config)