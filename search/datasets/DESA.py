import os
import time
import pickle
import torch
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_
from ..utils.utils import load_embedding, read_rel_feat
from ..rerank_model.DESA import DESA
from ..utils.loss import list_pairwise_loss
from ..evaluator import evaluate_test_qids_DESA


class DESADataset(Dataset):
    def __init__(self,  input_list):
        self.data = input_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self,  idx):
        doc_emb = self.data[idx][0].float()
        sub_emb = self.data[idx][1].float()
        doc_mask = 1-self.data[idx][2]
        sub_mask = 1-self.data[idx][3]
        weight = torch.tensor(self.data[idx][4]).float()
        index_i = self.data[idx][5]
        index_j = self.data[idx][6]
        pos_qrel_feat = self.data[idx][7].float()
        neg_qrel_feat = self.data[idx][8].float()
        pos_subrel_feat = self.data[idx][9].float()
        neg_subrel_feat = self.data[idx][10].float()
        subrel_mask = 1-self.data[idx][11]
        doc_emb.requires_grad = False
        sub_emb.requires_grad = False
        doc_mask.requires_grad = False
        sub_mask.requires_grad = False
        weight.requires_grad = False
        index_i.requires_grad = False
        index_j.requires_grad = False
        pos_qrel_feat.requires_grad = False
        neg_qrel_feat.requires_grad = False
        pos_subrel_feat.requires_grad = False
        neg_subrel_feat.requires_grad = False
        subrel_mask.requires_grad = False
        return doc_emb, sub_emb, doc_mask, sub_mask, weight, index_i, index_j, pos_qrel_feat, neg_qrel_feat, \
               pos_subrel_feat, neg_subrel_feat, subrel_mask


def gen_data_file_train(train_qids, qd, train_data, doc_emb, query_emb, rel_feat, save_path):
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


def get_fold_loader(fold, train_data, BATCH_SIZE):
    input_list = []
    starttime = time.time()
    print('Begin loading fold {} training data'.format(fold))
    for i in range(len(train_data)):
        input_list.append([train_data[i]['doc2vec'],
                           train_data[i]['sub2vec'],
                           train_data[i]['doc2vec_mask'],
                           train_data[i]['sub2vec_mask'],
                           train_data[i]['weight'],
                           torch.argmax(train_data[i]['positive_mask']),
                           torch.argmax(train_data[i]['negative_mask']),
                           train_data[i]['pos_qrel_feat'],
                           train_data[i]['neg_qrel_feat'],
                           train_data[i]['pos_subrel_feat'],
                           train_data[i]['neg_subrel_feat'],
                           train_data[i]['subrel_feat_mask']])
    desa_dataset = DESADataset(input_list)
    loader = DataLoader(
        dataset=desa_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    print('Training data loaded!')
    endtime = time.time()
    print('Total time  = ', round(endtime - starttime, 2), 'secs')
    return loader


def DESA_run(config):
    ''' load randomly shuffled queries '''
    if not os.path.exists(os.path.join(config['model_save_dir'], config['model'])):
        os.makedirs(os.path.join(config['model_save_dir'], config['model']))
    qd = pickle.load(open(os.path.join(config['data_dir'], 'div_query.data'), 'rb'))
    fold_p = os.path.join(config['data_dir'], config['model'], 'fold/')
    final_metrics = []
    best_model_list = []
    fold_time = 0
    for _fold in os.listdir(fold_p):
        train_data = torch.load(os.path.join(fold_p, _fold, 'train_data.pkl'))
        test_data = torch.load(os.path.join(fold_p, _fold, 'test_data.pkl'))
        desa_data_loader = get_fold_loader(fold_time, train_data, config['batch_size'])
        fold_time += 1
        print('Fold = ', fold_time)
        model = DESA(config['embedding_length'], 8, 2,config['embedding_length'], 8, 2, 8, config['dropout'])
        # doc_d_model, doc_nhead, doc_nlayers, sub_d_model, sub_nhead, sub_nlayers, d_model, nhead
        if torch.cuda.is_available():
            model = model.cuda()
        opt = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], betas=(0.9, 0.999), eps=1e-08, weight_decay=5e-4)
        params = list(model.parameters())
        if fold_time == 1:
            print('model = ', model)
            print(len(params))
            for param in params:
                print(param.size())
            n_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
            print('* number of parameters: %d' % n_params)

        max_metric = 0
        patience = 0
        best_model = ""
        for epoch in range(config['epoch']):
            print('Start Training...')
            model.train()
            for step, train_data in enumerate(tqdm(desa_data_loader, desc='BATCH', ncols=80)):
                doc_emb, sub_emb, doc_mask, sub_mask, weight, index_i, index_j, \
                pos_qrel_feat, neg_qrel_feat, \
                pos_subrel_feat, neg_subrel_feat, subrel_mask = train_data
                if torch.cuda.is_available():
                    doc_emb = doc_emb.cuda()
                    sub_emb = sub_emb.cuda()
                    doc_mask = doc_mask.cuda()
                    sub_mask = sub_mask.cuda()
                    weight = weight.cuda()
                    index_i = index_i.cuda()
                    index_j = index_j.cuda()
                    pos_qrel_feat = pos_qrel_feat.cuda()
                    neg_qrel_feat = neg_qrel_feat.cuda()
                    pos_subrel_feat = pos_subrel_feat.cuda()
                    neg_subrel_feat = neg_subrel_feat.cuda()
                    subrel_mask = subrel_mask.cuda()
                score_1, score_2 = model(doc_emb, sub_emb, doc_mask, sub_mask, pos_qrel_feat, pos_subrel_feat,
                                         index_i, index_j, neg_qrel_feat, neg_subrel_feat, subrel_mask)
                acc, loss = list_pairwise_loss(score_1, score_2, weight)
                opt.zero_grad()
                loss.backward()
                clip_grad_norm_(model.parameters(), max_norm=1)
                opt.step()
                if (step + 1) % config['eval_steps'] == 0:
                    model.eval()
                    metrics = []
                    for qid in test_data:
                        metric = evaluate_test_qids_DESA(model, test_data[str(qid)], qd[str(qid)])
                        metrics.append(metric)
                    avg_alpha_NDCG = np.mean(metrics)
                    if max_metric < avg_alpha_NDCG:
                        max_metric = avg_alpha_NDCG
                        tqdm.write('max avg_alpha_NDCG updated: {}'.format(max_metric))
                        model_filename = os.path.join(config['model_save_dir'], config['model'], 'TOTAL_EPOCH_' + str(config['epoch']) + '_FOLD_' + str(fold_time) + '_EPOCH_' + str(epoch) + '_LR_' + str(config['learning_rate']) + '_BATCHSIZE_' + str(
                        config['batch_size']) + '_DROPOUT_' + str(config['dropout']) + '_' + str(config['embedding_type']) + '.pickle')
                        torch.save(model.state_dict(), model_filename)
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
                        tqdm.write("decay lr: {}, load model: {}".format(new_lr, best_model))
            model.eval()
            metrics = []
            for qid in test_data:
                metric = evaluate_test_qids_DESA(model, test_data[str(qid)], qd[str(qid)])
                metrics.append(metric)
            avg_alpha_NDCG = np.mean(metrics)
            if max_metric < avg_alpha_NDCG:
                max_metric = avg_alpha_NDCG
                tqdm.write('max avg_alpha_NDCG updated: {}'.format(max_metric))
                model_filename = os.path.join(config['model_save_dir'], config['model'], 'TOTAL_EPOCH_' + str(config['epoch']) + '_FOLD_' + str(fold_time) + '_EPOCH_' + str(epoch) + '_LR_' + str(config['learning_rate']) + '_BATCHSIZE_' + str(
                config['batch_size']) + '_DROPOUT_' + str(config['dropout']) + '_' + str(config['embedding_type']) + '.pickle')
                torch.save(model.state_dict(), model_filename)
                best_model = model_filename
            if epoch == (config['epoch'] - 1):
                final_metrics.append(max_metric)
                best_model_list.append(best_model)

    print('alpha-nDCG = {}, best model = {}'.format(sum(final_metrics)/len(final_metrics), best_model_list))


