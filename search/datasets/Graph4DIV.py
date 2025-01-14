class GraphDataset(Dataset):
    def __init__(self,  graph_list):
        self.data = graph_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self,  idx):
        feat = self.data[idx][0]
        w = self.data[idx][1]
        rel_feat_tensor = torch.tensor(self.data[idx][2]).float()
        pos_mask = self.data[idx][3].bool()
        neg_mask = self.data[idx][4].bool()
        A = self.data[idx][5].float()
        degree_tensor = self.data[idx][6].float()
        feat.requires_grad = False
        rel_feat_tensor.requires_grad = False
        pos_mask.requires_grad = False
        neg_mask.requires_grad = False
        A.requires_grad = False
        degree_tensor.requires_grad = False
        return A, feat, rel_feat_tensor, degree_tensor, pos_mask, neg_mask, w


class Dataset(TensorDataset):
    def __init__(self, X_input_ids1, X_attention_mask1, X_token_type_ids1, X_input_ids2, X_attention_mask2, X_token_type_ids2, y_labels=None):
        super(Dataset, self).__init__()
        X_input_ids1 = torch.LongTensor(X_input_ids1)
        X_attention_mask1 = torch.LongTensor(X_attention_mask1)
        X_token_type_ids1 = torch.LongTensor(X_token_type_ids1)
        X_input_ids2 = torch.LongTensor(X_input_ids2)
        X_attention_mask2 = torch.LongTensor(X_attention_mask2)
        X_token_type_ids2 = torch.LongTensor(X_token_type_ids2)
        if y_labels is not None:
            y_labels = torch.FloatTensor(y_labels)
            self.tensors = [X_input_ids1, X_attention_mask1, X_token_type_ids1, X_input_ids2, X_attention_mask2, X_token_type_ids2, y_labels]
        else:
            self.tensors = [X_input_ids1, X_attention_mask1, X_token_type_ids1, X_input_ids2, X_attention_mask2, X_token_type_ids2]

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)

    def __len__(self):
        return len(self.tensors[0])
    

def get_clf_train_coverage(fold):
    fold_judge_coverage = './data/gcn_dataset/clf_cover_result/fold'+str(fold)+'_train_clf.csv'
    f = open(fold_judge_coverage, 'r')
    cover_dict = {}
    count = 0
    for line in f:
        if count == 0:
            count += 1
            continue
        line = line.strip('\n')
        qid, docx, docy, c = line.split(',')
        cover_dict[(qid, docx, docy)] = int(c)
    return cover_dict


def get_clf_test_coverage():
    cover_dict = {}
    for fold in range(1,6):
        filename = './data/gcn_dataset/clf_cover_result/fold' + str(fold) + '_test_clf.csv'
        f = open(filename, 'r')
        count = 0
        for line in f:
            if count == 0:
                count += 1
                continue
            line = line.strip('\n')
            qid, docx, docy, c = line.split(',')
            if (qid, docx, docy) in cover_dict:
                print('error:', (qid, docx, docy))
            cover_dict[(qid, docx, docy)] = int(c)
    return cover_dict


def build_training_graph(qid_list, res_dir, qd, rel_feat_dict, train_data, doc_emb, query_emb, cover_feat_dict, EMB_LEN):
    for qid in tqdm(qid_list, desc = 'GenTrainGraph'):
        output_file_path = res_dir+str(qid)+'.pickle'
        graph_dict = {}
        graph_dict[qid] = []
        if qid not in train_data:
            pickle.dump(graph_dict, open(output_file_path, 'wb'), True)
            continue
        for item in train_data[qid]:
            label = list([float(i) for i in item[0]])
            if len(label) == 0:
                print('No training data for query #{}!'.format(qid))
                continue
            temp_q = qd[str(qid)]
            query = temp_q.query
            doc_list = temp_q.best_docs_rank#to check
            doc_rel_score_list = temp_q.best_docs_rank_rel_score
            lt = len(doc_rel_score_list)
            if lt < MAXDOC:
                doc_rel_score_list.extend([0.0]*(MAXDOC-lt))
            X_q = query_emb[query]
            node_feat = []
            node_feat.append(X_q)
            rel_feat_list = []
            ''' Build Adjacent Matrix for the Query '''
            A = np.zeros((MAXDOC+1, MAXDOC+1), dtype = float)
            for i in range(len(doc_list)):
                for j in range(i+1, len(doc_list)):
                    docx_id = doc_list[i]
                    docy_id = doc_list[j]
                    ''' selected document will not be connected to the candidate documents '''
                    if label[i] < 0 or label[j] < 0:
                        continue
                    try:
                        c = cover_feat_dict[(qid, str(docx_id), str(docy_id))]
                        A[i+1][j+1] = c
                        A[j+1][i+1] = c
                    except:
                        try:
                            c = cover_feat_dict[(qid, str(docy_id), str(docx_id))]
                            A[i+1][j+1] = c
                            A[j+1][i+1] = c
                        except:
                            print('No coverage data! qid = {}, i = {}, j = {}, docx = {}, docy = {}'.format(qid, i, j, docx_id, docy_id))
            ''' get degree feature of each document '''
            sub_A = A[1:, 1:]
            degree_list = []
            for i in range(sub_A.shape[0]):
                degree = np.sum(sub_A[i, :])
                degree_list.append(degree)
            degree_tensor = torch.tensor(degree_list).float().reshape(sub_A.shape[0], 1)

            ''' get node features '''
            for i in range(MAXDOC):
                if i < len(doc_list):
                    doc_label = 1 if label[i] < 0 else 0
                    if doc_label == 1:
                        ''' for the selected document '''
                        A[0][i+1] = doc_rel_score_list[i]
                        A[i+1][0] = doc_rel_score_list[i]
                        X_doc = doc_emb[doc_list[i]]
                        rel_feat = rel_feat_dict[(query, doc_list[i])]
                    else:
                        ''' for the candidate document '''
                        X_doc = doc_emb[doc_list[i]]
                        rel_feat = rel_feat_dict[(query, doc_list[i])]
                else:
                    ''' for the padding document '''
                    X_doc = [0]*EMB_LEN
                    rel_feat = [0]*REL_LEN
                node_feat.append(X_doc)
                rel_feat_list.append(rel_feat)
            A = torch.tensor(A).float()
            ''' Graph_dict[qid]=[(feature_tensor,weight,relevant_features,pos_mask,neg_mask,Adj_Matrix,degree_tensor)]'''
            feature_tensor = th.tensor(node_feat).float()
            weight = item[3]
            pos_mask = item[1].clone().detach()
            neg_mask = item[2].clone().detach()
            graph_dict[qid].append((feature_tensor, weight, rel_feat_list, pos_mask, neg_mask, A, degree_tensor))
        pickle.dump(graph_dict, open(output_file_path, 'wb'), True)


def multiprocessing_build_fold_training_graph(EMB_LEN, EMB_TYPE, worker_num = 4):
    ''' generate 5 fold training graph '''
    res_dir = './data/' + str(EMB_TYPE) + '_fold_training_graph/'
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    rel_feat_dict = get_rel_feat()
    all_qids = np.load('./data/gcn_dataset/all_qids.npy')
    qd = pickle.load(open('./data/gcn_dataset/div_query.data', 'rb'))
    train_data = pickle.load(open('./data/gcn_dataset/listpair_train.data', 'rb'))
    if EMB_TYPE == 'doc2vec':
        doc_emb = load_emb('./data/gcn_dataset/doc2vec_doc.emb')
        query_emb = load_emb('./data/gcn_dataset/doc2vec_query.emb')
    elif EMB_TYPE == 'bert':
        doc_emb = load_emb('./data/gcn_dataset/bert_doc.emb')
        query_emb = load_emb('./data/gcn_dataset/bert_query.emb')
    fold = 0
    for train_ids, test_ids in KFold(5).split(all_qids):
        fold += 1
        cover_feat_dict = get_clf_train_coverage(fold)
        if not cover_feat_dict:
            print('Error! Intent coverage fold {} file not found'.format(fold))
            continue
        fold_dir = res_dir + 'fold_' + str(fold) + '/'
        if not os.path.exists(fold_dir):
            os.makedirs(fold_dir)
        train_ids.sort()
        train_qids = [str(all_qids[i]) for i in train_ids]
        task_list = split_list(train_qids, worker_num)
        jobs = []
        for task in task_list:
            p = multiprocessing.Process(target = build_training_graph, args = (task, fold_dir, qd, rel_feat_dict, train_data, doc_emb, query_emb, cover_feat_dict, EMB_LEN))
            jobs.append(p)
            p.start()
    print('training graph generation done!')


def build_test_graph(EMB_LEN, EMB_TYPE):
    ''' Build the test Graph for testing '''
    graph_dict = {}
    qid_list = []
    rel_feat_dict = get_rel_feat()
    cover_feat_dict = get_clf_test_coverage()
    qd = pickle.load(open('./data/gcn_dataset/div_query.data', 'rb'))
    output_path = './data/gcn_dataset/' + str(EMB_TYPE) + '_test_graph.data'
    if EMB_TYPE == "doc2vec":
        doc_emb = load_emb('./data/gcn_dataset/doc2vec_doc.emb')
        query_emb = load_emb('./data/gcn_dataset/doc2vec_query.emb')
    elif EMB_TYPE == 'bert':
        doc_emb = load_emb('./data/gcn_dataset/bert_doc.emb')
        query_emb = load_emb('./data/gcn_dataset/bert_query.emb')
    ''' remove query #95 and #100 '''
    all_qids = range(1, 201)
    del_index = [94, 99]
    all_qids = np.delete(all_qids, del_index)
    qids = [str(i) for i in all_qids]

    for qid in tqdm(qids, desc = "Gen Test Graph"):
        temp_q = qd[str(qid)]
        ''' use the top 50 relevant document for diversity ranking '''
        doc_list = temp_q.doc_list[:MAXDOC]
        rel_score_list = temp_q.doc_score_list[:MAXDOC]
        real_num = len(doc_list)
        label = list([0] * real_num)
        index = [i for i in range(real_num)]
        ''' The document list for testing is randomly shuffled'''
        c = list(zip(doc_list, rel_score_list, index))
        random.shuffle(c)
        doc_list[:], rel_score_list[:], index[:] = zip(*c)
        ''' padding '''
        if len(label) == 0:
            continue
        elif len(label)<MAXDOC:
            label.extend([0] * (MAXDOC-len(label)))
        ''' get the node features '''
        query = temp_q.query
        X_q = query_emb[query]
        node_feat = []
        node_feat.append(X_q)
        rel_feat_list = []
        for i in range(MAXDOC):
            if i < real_num:
                X_doc = doc_emb[doc_list[i]]
                rel_feat = rel_feat_dict[(query, doc_list[i])]
                rel_feat_list.append(rel_feat)
            else:
                X_doc = [0]*EMB_LEN
                rel_feat = [0]*REL_LEN
                rel_feat_list.append(rel_feat)
            node_feat.append(X_doc)

        ''' Build Adjcent Matrix for the shuffled document list '''
        A = np.zeros((MAXDOC+1, MAXDOC+1), dtype = float)
        for i in range(len(doc_list)):
            for j in range(i+1, len(doc_list)):
                docx_id = doc_list[i]
                docy_id = doc_list[j]
                try:
                    c = cover_feat_dict[(str(qid), str(docx_id), str(docy_id))]
                    A[i+1][j+1] = c
                    A[j+1][i+1] = c
                except:
                    try:
                        c = cover_feat_dict[(str(qid), str(docy_id), str(docx_id))]
                        A[i+1][j+1] = c
                        A[j+1][i+1] = c
                    except:
                        print('ERROR: qid = {}, i = {}, j = {}, docx = {}, docy = {}'.format(qid, i, j, docx_id, docy_id))
        ''' get the degree features '''
        sub_A = A[1:, 1:]
        degree_list = []
        for i in range(sub_A.shape[0]):
            degree = np.sum(sub_A[i, :])
            degree_list.append(degree)
        degree_tensor = torch.tensor(degree_list).float().reshape(sub_A.shape[0], 1)

        node_feat = torch.tensor(node_feat).float()
        A = torch.tensor(A).float()
        ''' Graph_dict[qid] = (node_features, relevance_feature_list, relevance_score_list, Adj_Matrix, degree_tensor) '''
        graph_dict[qid] = (node_feat, index, rel_feat_list, rel_score_list, A, degree_tensor)
    pickle.dump(graph_dict, open(output_path, 'wb'), True)
