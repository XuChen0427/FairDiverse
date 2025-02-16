import os
import json
import pickle

from ..utils.utils import pkl_load
from ..evaluator import get_metrics_20
from .base import BasePostProcessUnsupervisedModel

'''
@inproceedings{santos2010exploiting,
  title={Exploiting query reformulations for web search result diversification},
  author={Santos, Rodrygo LT and Macdonald, Craig and Ounis, Iadh},
  booktitle={Proceedings of the 19th international conference on World wide web},
  pages={881--890},
  year={2010}
}
'''

class xQuAD(BasePostProcessUnsupervisedModel):
    def __init__(self, top_k=20):
        super().__init__(top_k)
    
    def rerank(self, config):
        qd = pickle.load(open(os.path.join(config['data_dir'], 'div_query.data'), 'rb'))
        bm25_dict = pkl_load(os.path.join(config['data_dir'], config['model'], 'bm25_scores.pkl'))

        output_dir = os.path.join(config['tmp_dir'], config['model'])
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        output_txt = os.path.join(output_dir, 'output_best.txt')
        csv_path = os.path.join(output_dir, 'result.csv')

        for qid in qd:
            doc_ranking = self.calculate_xquad_score(qid, bm25_dict, config['lambda'])

            judge_f = open(os.path.join(output_dir, str(qid)+'.txt'), 'w')
            for i in range(len(doc_ranking)):
                doc_ranking_score = len(doc_ranking) - i
                judge_f.write(str(qid) + ' Q0 ' + doc_ranking[i] + ' ' + str(i + 1) + ' ' + str(doc_ranking_score) + ' indri\n')
            judge_f.close()

        command = 'cat '+output_dir+'/* > '+output_txt
        os.system(command)
        command = './search/eval/clueweb09/ndeval ./search/eval/clueweb09/2009-2012.diversity.ndeval.qrels '+output_txt+' >' + str(csv_path)
        os.system(command)

        alpha_nDCG_20, NRBP_20, ERR_IA_20, S_rec_20 = get_metrics_20(csv_path)
        print('alpha_nDCG@20_std = {}, NRBP_20 = {}, ERR_IA_20 = {}, S_rec_20 = {}'.format(alpha_nDCG_20, NRBP_20, ERR_IA_20, S_rec_20))


    def calculate_xquad_score(self, qid, bm25_dict, _lambda=0.5):
        """
        Calculate expanded ranking score using original query and suggestions
        Args:
            qid: Query ID
            bm25_dict: Dictionary containing BM25 scores for original and suggestion queries
            _lambda: Weight parameter between original and expanded scores (default: 0.5)
            top_k: Number of documents to return (default: 10)
        
        Returns:
            List of selected document IDs in ranked order
        """
        original_scores = bm25_dict[qid][0]  # Original query scores
        suggestion_scores = bm25_dict[qid][1]  # Suggestion query scores (n_suggestions x n_docs)
        
        selected_docs = []
        available_docs = set(original_scores.keys())
        
        # Calculate suggestion probabilities (assuming normalized BM25 scores)
        total_suggestion_score = suggestion_scores.sum(axis=0)
        suggestion_probs = suggestion_scores / (total_suggestion_score + 1e-10)  # Add epsilon to avoid division by zero
        
        while len(selected_docs) < self.top_k and available_docs:
            best_score = float('-inf')
            best_doc = None
            
            for doc_id in available_docs:
                # Get original query score component
                original_component = (1 - _lambda) * original_scores[doc_id]
                
                # Calculate expanded component
                expanded_component = 0
                doc_idx = list(original_scores.keys()).index(doc_id)
                
                for qi_idx in range(len(suggestion_scores)):
                    qi_score = suggestion_probs[qi_idx][doc_idx]
                    
                    # Calculate product term for already selected documents
                    product_term = 1
                    for selected_doc in selected_docs:
                        selected_idx = list(original_scores.keys()).index(selected_doc)
                        product_term *= (1 - suggestion_scores[qi_idx][selected_idx])
                    
                    expanded_component += qi_score * product_term
                
                expanded_component *= _lambda
                
                # Combined score
                total_score = original_component + expanded_component
                
                if total_score > best_score:
                    best_score = total_score
                    best_doc = doc_id
            
            if best_doc is None:
                break
                
            selected_docs.append(best_doc)
            available_docs.remove(best_doc)
        
        return selected_docs

