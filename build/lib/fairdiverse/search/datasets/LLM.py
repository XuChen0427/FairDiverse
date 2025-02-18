import os
import time
import json
import pickle
import logging
import argparse
import json_repair

from tqdm import tqdm

from ..llm_model.api_llm import LMAgent
from ..utils.utils import restore_doc_ids, remove_duplicate, get_metrics_20


MAXDOC = 50

adhoc_rerank_prompt_input = """## Input Data
<Query>: {query}
<Document List>: {document_list}

## Output Data
<Output>: 
"""


def llm_run(config):
    """
    Executes a large language model-based document reranking pipeline for search result diversification.
    This function processes queries and their candidate documents through a language model to generate diversified document rankings. 
    
    :param config: Dictionary containing configuration parameters
    """

    best_rank_dir = os.path.join(config['data_dir'], 'best_rank/')
    data_content_dir = os.path.join(config['data_dir'], config['data_content_dir'])

    output_dir = os.path.join(config['tmp_dir'], config['model'], config['model_name'])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_txt = os.path.join(output_dir, 'output_best.txt')
    final_output_path = os.path.join(output_dir, 'output_best_new.txt')
    csv_path = os.path.join(output_dir, 'result.csv')
    
    qid_need_process = sorted(os.listdir(best_rank_dir), key=lambda x: int(x.split('.')[0]))
    doc_content_dict = {}
    print("LOAD Document Content ...")
    for fname in os.listdir(data_content_dir):
        with open(data_content_dir+fname, 'r') as fr:
            data_dict = json.load(fr)
            doc_content_dict[fname[:-5]] = data_dict

    adhoc_rerank_prompt_fr = open(os.path.join(config['data_dir'], config['prompts_dir']), 'r')
    adhoc_rerank_prompt = adhoc_rerank_prompt_fr.read()
    adhoc_rerank_prompt_fr.close()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    llm = LMAgent(config)

    for file_name in tqdm(qid_need_process):
        
        qd = pickle.load(open(best_rank_dir+file_name, 'rb'))
        qid = qd.qid
        query = qd.query
        doc_list = qd.doc_list[:MAXDOC]

        print('=====> Query {}. Processing Begin'.format(qid))
        
        docid2num = {doc_id: idx + 1 for idx, doc_id in enumerate(doc_list)}
        document_content_list = '\n'.join(['['+str(docid2num[doc_id])+'] '+' '.join(doc_content_dict[doc_id]['body_text'].split(' ')[:100]) for doc_id in doc_list])
        _adhoc_rerank_prompt = adhoc_rerank_prompt+adhoc_rerank_prompt_input.format(query=query, document_list=document_content_list)
        output = llm('', _adhoc_rerank_prompt, max_new_tokens=config['max_new_tokens'])
        output = output[:output.find('}')+1]
        output = json_repair.loads(output)["rerank_list"]
        doc_ranking = restore_doc_ids(output, docid2num)

        if len(doc_ranking) < 20:
            remain_doc = [item for item in doc_list if item not in doc_ranking]
            doc_ranking = doc_ranking + remain_doc[:20-len(doc_ranking)]
        print('=====> Document Ranking {}. '.format(doc_ranking))

        judge_f = open(os.path.join(output_dir, file_name[:-5]+'.txt'), 'w')
        for i in range(len(doc_ranking)):
            doc_ranking_score = len(doc_ranking) - i
            judge_f.write(str(qid) + ' Q0 ' + doc_ranking[i] + ' ' + str(i + 1) + ' ' + str(doc_ranking_score) + ' indri\n')
        judge_f.close()

    command = 'cat '+output_dir+'/* > '+output_txt
    os.system(command)
    remove_duplicate(output_txt, final_output_path)
    command = './search/eval/clueweb09/ndeval ./search/eval/clueweb09/2009-2012.diversity.ndeval.qrels '+final_output_path+' >' + str(csv_path)
    os.system(command)

    alpha_nDCG_20, NRBP_20, ERR_IA_20, S_rec_20 = get_metrics_20(csv_path)
    print('alpha_nDCG@20_std = {}, NRBP_20 = {}, ERR_IA_20 = {}, S_rec_20 = {}'.format(alpha_nDCG_20, NRBP_20, ERR_IA_20, S_rec_20))

    


