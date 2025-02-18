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

config = {'model': 'xquad', 'dataset': 'clueweb09', 'task': 'search', 'mode': 'train', "log_name": "test", "model_save_dir": "model/", "tmp_dir": "tmp/", "best_model_list": [], "device": "cpu"}
trainer = SRDTrainer(train_config=config)
trainer.train()
