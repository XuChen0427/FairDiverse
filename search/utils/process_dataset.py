import os
import copy
import pickle
import multiprocessing
import xml.dom.minidom
from div_type import *

from .utils import split_list

MAXDOC = 50
REL_LEN = 18


def get_query_dict(config):
    dq_dict = {}
    topics_list = []
    for year in ['2009','2010','2011','2012']:
        filename = os.path.join(config['data_dir'], config['ground_truth'], 'wt_topics/wt' + year + '.topics.xml')
        DOMTree = xml.dom.minidom.parse(filename)
        collection = DOMTree.documentElement
        topics = collection.getElementsByTagName("topic")
        topics_list.extend(topics)
    ''' load subtopics for each query '''
    for topic in topics_list:
        if topic.hasAttribute("number"):
            qid = topic.getAttribute("number")
        query = topic.getElementsByTagName('query')[0].childNodes[0].data
        subtopics = topic.getElementsByTagName('subtopic')
        subtopic_id_list = []
        subtopic_list = []
        for subtopic in subtopics:
            if subtopic.hasAttribute('number'):
                subtopic_id = subtopic.getAttribute('number')
                subtopic_id_list.append(subtopic_id)
            sub_query = subtopic.childNodes[0].data
            subtopic_list.append(sub_query)
        dq = div_query(qid, query, subtopic_id_list, subtopic_list)
        dq_dict[str(qid)] = dq
    return dq_dict


def get_query_suggestion(dq, config):
    dq_dict = {}
    filename = os.path.join(config['data_dir'], 'query_suggestion.xml')
    DOMTree = xml.dom.minidom.parse(filename)
    collection = DOMTree.documentElement
    topics = collection.getElementsByTagName("topic")
    ''' load subtopics for each query '''
    for topic in topics:
        if topic.hasAttribute("number"):
            qid = topic.getAttribute("number")
        query = topic.getElementsByTagName('query')[0].childNodes[0].data
        subtopics = topic.getElementsByTagName('subtopic1')
        subtopic_id_list = []
        subtopic_list = []
        for subtopic in subtopics:
            if subtopic.hasAttribute('number'):
                subtopic_id = subtopic.getAttribute('number')
                subtopic_id_list.append(subtopic_id)
            suggestion = subtopic.getElementsByTagName('suggestion')[0].childNodes[0].data
            subtopic_list.append(suggestion)
        dq[str(qid)].add_query_suggestion(subtopic_list)
    return dq_dict


def get_docs_dict(config):
    ''' 
    get the relevance score of the documents 
    docs_dict[qid] = [doc_id, ...]
    docs_rel_score_dict[qid] = [score, ...]
    '''
    docs_dict = {}
    docs_rel_score_dict = {}
    for year in ['2009','2010','2011','2012']:
        filename = os.path.join(config['data_dir'], config['ground_truth'], 'wt' + year + '.txt')
        f = open(filename)
        for line in f:
            qid, _, docid, _, score, _ = line.split(' ')
            if str(qid) not in docs_dict:
                docs_dict[str(qid)] = []
                docs_rel_score_dict[str(qid)] = []
            docs_dict[str(qid)].append(str(docid))
            docs_rel_score_dict[str(qid)].append(float(score))
    ''' Normalize the relevance score of the documents '''
    for qid in docs_rel_score_dict:
        temp_score_list = copy.deepcopy(docs_rel_score_dict[qid])
        for i in range(len(temp_score_list)):
            temp_score_list[i] = docs_rel_score_dict[qid][0]/docs_rel_score_dict[qid][i]
        docs_rel_score_dict[qid] = temp_score_list
    return docs_dict, docs_rel_score_dict


def get_doc_judge(qd, dd, ds, config):
    ''' 
    load document list and relevance socre list for the corresponding query 
    qd : query dictionary
    dd : document dictionary
    ds : document relevance score dictionary
    '''
    get_query_suggestion(qd, config)
    for key in qd:
        qd[key].add_docs(dd[key])
        qd[key].add_docs_rel_score(ds[key])
    for year in ['2009','2010','2011','2012']:
        filename = os.path.join(config['data_dir'], config['ground_truth'], 'wt_judge/' + year + '.diversity.qrels')
        f = open(filename, 'r')
        for line in f:
            qid, subtopic, docid, judge = line.split(' ')
            judge = int(judge)
            if judge > 0:
                if str(docid) in qd[str(qid)].subtopic_df.index.values:
                    qd[str(qid)].subtopic_df[str(subtopic)][str(docid)] = 1
    return qd


def data_process_worker(task, data_dir):
    for item in task:
        qid = item[0]
        dq = item[1]
        ''' get the best ranking for the top 50 relevant documents '''
        dq.get_best_rank(MAXDOC)
        pickle.dump(dq, open(data_dir+str(qid)+'.data', 'wb'), True)


def calculate_best_rank(qd, config):
    data_dir = os.path.join(config['data_dir'], 'best_rank/')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    q_list = []
    for key in qd:
        x = copy.deepcopy(qd[key])
        q_list.append((str(key), x))
    jobs = []
    task_list = split_list(q_list, 8)
    for task in task_list:
        p = multiprocessing.Process(target = data_process_worker, args = (task, data_dir))
        jobs.append(p)
        p.start()
    for job in jobs:
        job.join()


def get_stand_best_metric(qd, config):
    ''' load best alpha-nDCG from DSSA '''
    std_dict = pickle.load(open(os.path.join(config['data_dir'], 'stand_metrics.data'), 'rb'))
    for qid in std_dict:
        m = std_dict[qid]
        target_q = qd[str(qid)]
        target_q.set_std_metric(m)


def data_process(config):
    ''' get subtopics for each query '''
    qd = get_query_dict(config)
    ''' get documents dictionary '''
    dd, ds = get_docs_dict(config)
    ''' get diversity judge for documents '''
    qd = get_doc_judge(qd, dd, ds, config)
    ''' get the stand best alpha-nDCG from DSSA '''
    get_stand_best_metric(qd, config)
    ''' get the best ranking for top n relevant documents and save as files'''
    calculate_best_rank(qd, config)
    ''' combine the best ranking into a single file '''
    generate_qd(config)


def generate_qd(config):
    ''' generate diversity_query file from data_dir '''
    data_dir = os.path.join(config['data_dir'], 'best_rank/')
    files = os.listdir(data_dir)
    files.sort(key = lambda x:int(x[:-5]))
    query_dict = {}
    for f in files:
        file_path = os.path.join(data_dir, f)
        temp_q = pickle.load(open(file_path, 'rb'))
        query_dict[str(f[:-5])] = temp_q
    pickle.dump(query_dict, open(os.path.join(config['data_dir'], 'div_query.data'), 'wb'), True)
    return query_dict