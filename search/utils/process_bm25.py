import os
import math
import json
import pickle
import numpy as np
from collections import Counter
from tqdm import tqdm

from .utils import pkl_save

MAXDOC = 50


def generate_bm25_scores_for_query(config):
    qd = pickle.load(open(os.path.join(config['data_dir'], 'div_query.data'), 'rb'))
    
    data_content_dir = config['data_content_dir']
    doc_content_dict = {}
    for fname in os.listdir(data_content_dir):
        with open(os.path.join(data_content_dir, fname), 'r') as fr:
            data_dict = json.load(fr)
            doc_content_dict[fname[:-5]] = data_dict

    bm25_dict = {}
    for qid in tqdm(qd):
        bm25_dict[qid] = [{}]
        query = qd[qid].query
        doc_list = qd[qid].doc_list[:MAXDOC]
        query_suggestion = qd[qid].query_suggestion
        doc_content_list = [doc_content_dict[doc_id]['body_text'] for doc_id in doc_list]
        bm25_scores = calculate_bm25(query, doc_content_list)
        for i, doc_id in enumerate(doc_list):
            bm25_dict[qid][0][doc_id] = bm25_scores[i]
        sub_query_bm25 = []
        for sub_query in query_suggestion:
            bm25_scores = calculate_bm25(sub_query, doc_content_list)
            sub_query_bm25.append(bm25_scores)
        bm25_dict[qid].append(np.array(sub_query_bm25))
    
    if not os.path.exists(os.path.join(config['data_dir'], config['model'])):
        os.makedirs(os.path.join(config['data_dir'], config['model']))
    pkl_save(bm25_dict, os.path.join(config['data_dir'], config['model'], 'bm25_scores.pkl'))


def calculate_bm25(query, documents, k1=1.5, b=0.75):
    N = len(documents)
    avgdl = sum(len(doc.split()) for doc in documents) / N

    bm25_scores = []

    def idf(term):
        df = sum(1 for doc in documents if term in doc.split())
        return math.log((N - df + 0.5) / (df + 0.5) + 1.0)

    for doc in documents:
        score = 0.0
        doc_terms = Counter(doc.split())
        doc_length = len(doc.split())

        for term in query.split():
            if term in doc_terms:
                f = doc_terms[term]
                term_idf = idf(term)
                score += term_idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * (doc_length / avgdl)))

        bm25_scores.append(score)
    
    return bm25_scores
