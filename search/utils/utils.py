import math
def split_list(origin_list, n):
    res_list = []
    L = len(origin_list)
    N = int(math.ceil(L / float(n)))
    begin = 0
    end = begin + N
    while begin < L:
        if end < L:
            temp_list = origin_list[begin:end]
            res_list.append(temp_list)
            begin = end
            end += N
        else:
            temp_list = origin_list[begin:]
            res_list.append(temp_list)
            break
    return res_list


def load_emb(filename, sep = '\t'):
    '''
    load embedding from file
    :param filename: embedding file name
    :param sep: the char used as separation symbol
    :return: a dict with item name as key and embedding vector as value
    '''
    with open(filename, 'r') as fp:
        result = {}
        for l in fp:
            l = l.strip()
            if l == '':
                continue
            sp = l.split(sep)
            vals = [float(sp[i]) for i in range(1, len(sp))]
            result[sp[0]] = vals
        return result
    

def get_rel_feat(path):
    rel_feat = pd.read_csv(path)
    rel_feat_names = list(sorted(set(rel_feat.columns) - {'query', 'doc'}))
    rel_feat[rel_feat_names] = StandardScaler().fit_transform(rel_feat[rel_feat_names])
    rel_feat = dict(zip(map(lambda x: tuple(x), rel_feat[['query', 'doc']].values),
            rel_feat[rel_feat_names].values.tolist()))
    return rel_feat


def read_rel_feat(path):
    rel_feat = {}
    f = csv.reader(open(path, 'r'), delimiter = ',')
    next(f)
    for line in f:
        if line[0] not in rel_feat:
            rel_feat[line[0]] = {}
        if line[1] not in rel_feat[line[0]]:
            rel_feat[line[0]][line[1]] = np.array([float(val) for val in line[2:]])
    return rel_feat


def get_fold_loader(fold, train_ids, BATCH_SIZE, EMB_TYPE):
    graph_list = []
    t = time.time()
    starttime = time.time()
    print('Begin loading fold {} training data'.format(fold))
    print('trian_ids = ', train_ids)
    data_dir = './data/'+str(EMB_TYPE)+'_fold_training_graph/fold_'+str(fold)+'/'
    for qid in tqdm(train_ids, desc = 'loading training data', ncols=80):
        file_path = os.path.join(data_dir, str(qid)+'.pickle')
        temp_dict = pickle.load(open(file_path, 'rb'))
        graph_list.extend(temp_dict[str(qid)])
    graph_dataset = GraphDataset(graph_list)
    loader = DataLoader(
        dataset = graph_dataset, 
        batch_size = BATCH_SIZE, 
        shuffle = True, 
        num_workers = 4, 
        pin_memory = True, 
    )
    print('Training data loaded!')
    endtime = time.time()
    print('Total time  = ', round(endtime - starttime, 2), 'secs')
    return loader


def get_intent_coverage():
    ''' get the intent coverage of all document pairs '''
    qd = pickle.load(open('./data/gcn_dataset/div_query.data', 'rb'))
    output_file = './data/gcn_dataset/intent_coverage.csv'
    fout = open(output_file, 'w')
    fout.write('qid,docx,docy,label\n')
    for qid in qd:
        dq = qd[qid]
        doc_list = dq.doc_list
        subtopic_df = np.array(dq.subtopic_df)
        real_num = min(len(doc_list), MAXDOC)
        for i in range(real_num):
            for j in range(i+1, real_num):
                doc_x_id = doc_list[i]
                doc_y_id = doc_list[j]
                coverage = 0
                for k in range(subtopic_df.shape[1]):
                    if subtopic_df[i][k] == 1 and subtopic_df[j][k] == 1:
                        coverage = 1
                        break
                fout.write(str(qid)+','+str(doc_x_id)+','+str(doc_y_id)+','+str(coverage)+'\n')
    fout.close()


def document_process():
    ''' tokenize the document for relation classifier training '''
    input_file_path = './data/gcn_dataset/full_content.pkl.gz'
    output_file_path = './data/gcn_dataset/token_content.pkl.gz'
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    with gzip.open(input_file_path, 'rb') as f:
        doc_dict = pickle.load(f)
    token_dict = {}

    qd = pickle.load(open('./data/gcn_dataset/div_query.data', 'rb'))
    all_docs_id_list = []
    for qid in qd:
        dq = qd[qid]
        doc_list = dq.doc_list
        real_num = min(len(doc_list), MAXDOC)
        all_docs_id_list.extend(doc_list[:real_num])
    all_docs_id_list = list(set(all_docs_id_list))

    for key in tqdm(doc_dict, desc = "Docs process"):
        if key in all_docs_id_list:
            content = doc_dict[key]
            if type(content) == list:
                content = '. '.join(content)
            content = content.lower()
            sent = tokenizer.tokenize(content)
            token_dict[key] = sent[:511]
    with gzip.open(output_file_path, 'wb') as f:
        pickle.dump(token_dict, f)


def gen_data_file(data_list, file_full, file_equal):
    ''' Generate full dataset '''
    full_output = open(file_full, 'w')
    full_output.write('qid,docx,docy,label\n')
    for t in tqdm(data_list, desc = "Gen full data"):
        full_output.write(str(t[0])+','+str(t[1])+','+str(t[2])+','+str(t[3])+'\n')
    full_output.close()
    ''' Generate proportionate dataset '''
    equal_output = open(file_equal, 'w')
    equal_output.write('qid,docx,docy,label\n')
    pos_data = []
    neg_data = []
    for t in tqdm(data_list, desc = "Split data"):
        if t[3] == '1':
            pos_data.append(t)
        elif t[3] == '0':
            neg_data.append(t)
        else:
            print('error', t)
    Lp = len(pos_data)
    Ln = len(neg_data)
    real_num = min(Lp, Ln)
    for i in tqdm(range(real_num), desc = "Gen pos"):
        t = pos_data[i]
        equal_output.write(str(t[0])+','+str(t[1])+','+str(t[2])+','+str(t[3])+'\n')
    ''' Randomly sample the negative data '''
    output_neg_index = list(set([random.randint(0, Ln - 1) for _ in range(int(1.2 * real_num))]))[:real_num]
    output_neg_index.sort()
    for i in tqdm(output_neg_index, desc = 'Gen neg'):
        u = neg_data[i]
        equal_output.write(str(u[0])+','+str(u[1])+','+str(u[2])+','+str(u[3])+'\n')
    equal_output.close()
