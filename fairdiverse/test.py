# from recommendation.trainer import RecTrainer
#
# config = {'model': 'BPR', 'data_type': 'pair', 'fair-rank': True, 'rank_model': 'APR', 'use_llm': False, 'log_name': "test", 'dataset': 'steam'}
# trainer = RecTrainer(train_config=config)
# trainer.train()

# from recommendation.reranker import RecReRanker

# config = {'ranking_store_path': 'steam-base-mf', 'model': 'CPFair', 'fair-rank': True, 'log_name': 'test',
#           'fairness_metrics': ["MMF", "GINI"], 'dataset': 'steam'}

# reranker = RecReRanker(train_config=config)
# reranker.rerank()

from search.trainer import SRDTrainer