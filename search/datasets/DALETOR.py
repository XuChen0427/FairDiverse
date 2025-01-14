from torch.utils.data import Dataset, DataLoader


class TrainDataset(Dataset):
    # 初始化
    def __init__(self, train_list):
        # 读入数据
        self.data = train_list

    # 返回df的长度
    def __len__(self):
        # print('data len=',len(self.data))
        return len(self.data)

    # 获取第idx+1列的数据
    def __getitem__(self, idx):
        X = self.data[idx][0].clone().detach().float()
        rel_feat = self.data[idx][1].clone().detach().float()
        div_feat = self.data[idx][2].clone().detach().float()
        X.requires_grad=False
        rel_feat.requires_grad=False
        div_feat.requires_grad=False
        return X, rel_feat, div_feat


class TestDataset(Dataset):
    # 初始化
    def __init__(self, test_list):
        # 读入数据
        self.data = test_list

    # 返回df的长度
    def __len__(self):
        return len(self.data)

    # 获取第idx+1列的数据
    def __getitem__(self, idx):
        X=self.data[idx][0].clone().detach().float()
        rel_feat = self.data[idx][1].clone().detach().float()
        X.requires_grad=False
        rel_feat.requires_grad=False
        return X, rel_feat
    

def build_each_train_dataset(qid_list, qd, train_dict, rel_feat_dict, res_dir, query_emb, doc_emb):
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
            
            # print('qid = {}, div = {}'.format(qid, df.shape))
            
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
        
        # 保存文档向量到文件中, 只保存一次, 以query为单位, 保存该query下best_rank的前50个文档
        assert len(sample_feat_list) == len(sample_div_list)
        assert len(sample_rel_list) == len(sample_div_list)
        sample_dict[qid]=[
            (sample_feat_list[i],
            sample_rel_list[i],
            sample_div_list[i])
            for i in range(len(sample_feat_list))
        ]
        pkl_save(sample_dict, sample_path)


def build_train_dataset(worker_num=20):
    res_dir='../data/train/'
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    all_qids=np.load('../data/all_qids.npy')
    qd = pickle.load(open('../data/div_query.data', 'rb'))
    # train_data=pickle.load(open('../data/listpair_train.data','rb'))
    doc_emb = load_emb('../data/doc2vec_doc.emb')
    query_emb = load_emb('../data/doc2vec_query.emb')
    train_dict = pkl_load('../data/list_train_samples.pkl.gz')
    rel_feat_dict = get_feature()

    task_list = split_list_n_list(all_qids, worker_num)
    jobs=[]
    for task in task_list:
        p = multiprocessing.Process(target=build_each_train_dataset, args=(task, qd, train_dict, rel_feat_dict, res_dir, query_emb, doc_emb))
        jobs.append(p)
        p.start()
    print('training dataset generation done!')


def build_test_dataset():
    qd = pickle.load(open('../data/div_query.data', 'rb'))
    doc_emb = load_emb('../data/doc2vec_doc.emb')
    query_emb = load_emb('../data/doc2vec_query.emb')
    output_dir = '../data/test/'
    rel_feat_dict = get_feature()
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
        #训练数据，(emb,query_mask, suggestion_mask, doc_mask, rel_feat)
        pkl_save(test_dict, output_file_path)


def get_train_loader(logger, fold, train_ids, BATCH_SIZE):
    data_list = []
    t=time.time()
    starttime=time.time()
    logger.info('Begin loading fold {} training data'.format(fold))
    logger.info('trian_ids = {}'.format(train_ids))
    data_dir='../data/train/'
    doc_tensor_dict={}
    for qid in tqdm(train_ids,desc='load train data', ncols=80):
        # 加载query和doc的embedding
        file_path = os.path.join(data_dir,str(qid)+'.pkl.gz')
        with gzip.open(file_path,'rb') as f:
            try:
                sample_dict = pickle.load(f)
            except EOFError:
                continue
        data_list.extend(sample_dict[str(qid)])
    train_dataset=TrainDataset(data_list)
    loader=DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    logger.info('Training data loaded!')
    endtime=time.time()
    logger.info('Total time = {} secs'.format(round(endtime - starttime, 2)))
    return loader


def get_test_loader(logger, fold, test_qids, BATCH_SIZE):
    t=time.time()
    starttime=time.time()
    data_dir = '../data/test/'
    test_dataset = {}
    test_data_list = []
    for qid in tqdm(test_qids, desc="load test data",ncols=80):
        # 加载query和doc的embedding
        file_path = os.path.join(data_dir,str(qid)+'.pkl.gz')
        with gzip.open(file_path,'rb') as f:
            try:
                temp_test_dict=pickle.load(f)
            except EOFError:
                continue
        test_dataset[str(qid)]=temp_test_dict[str(qid)]
        test_data_list.append(test_dataset[qid])
    
    evaluate_dataset=TestDataset(test_data_list)
    loader = DataLoader(
        dataset=evaluate_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    endtime=time.time()
    return loader


def get_test_dataset(logger, fold, test_qids):
    data_dir = '../data/test/'
    test_dataset = {}
    for qid in tqdm(test_qids, desc="load test data",ncols=80):
        # 加载query和doc的embedding
        file_path = os.path.join(data_dir,str(qid)+'.pkl.gz')
        with gzip.open(file_path,'rb') as f:
            try:
                temp_test_dict=pickle.load(f)
            except EOFError:
                continue
        test_dataset[str(qid)]=temp_test_dict[str(qid)]
    return test_dataset


def gen_list_training_sample(top_n = 50, sample_num = 200):
    qd = pickle.load(open('../data/div_query.data','rb'))
    doc_emb = load_emb('../data/doc2vec_doc.emb')
    rel_feat_dict = get_feature()
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
            # for j in range(len(top_docs)):
            #     if top_docs[j] not in doc_emb:
            #         flag = 1
            #         break
            for j in range(len(top_docs)):
                if (qid, top_docs[j]) not in rel_feat_dict:
                    flag = 1
                    break
            if flag == 0 and top_docs not in result_list:
                result_list.append(top_docs)
        print('qid={}, len={}'.format(qid, len(result_list)))
        train_dict[str(qid)]=result_list
    pkl_save(train_dict, '../data/list_train_samples.pkl.gz')

if __name__ == '__main__':
    # gen_list_training_sample()
    # build_train_dataset()
    # build_test_dataset()
    print('dataset gen!')