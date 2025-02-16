import numpy as np
import os
import yaml
from scipy.sparse import save_npz, load_npz
from .rerank_model import CPFair, FairRec, FairRecPlus, k_neighbor, min_regularizer, PMMF, Welf, TaxRank, FairSync, RAIF
from .metric import dcg, MMF, Gini, Entropy, MinMaxRatio
from datetime import datetime
import json


class RecReRanker(object):
    def __init__(self, train_config):
        """Initialize In-processing and base models.

        :param train_config: Your custom config files.
        """


        self.dataset = train_config['dataset']
        #self.stage = stage
        self.train_config = train_config


    def load_configs(self, dir):
        """
           Loads and merges configuration files for the model, dataset, and evaluation.

           This function loads multiple YAML configuration files, including the process configuration,
           dataset-specific settings, model configurations, and evaluation parameters. All configurations
           are merged, with the highest priority given to the class's own `config` attribute.

           :param dir: The directory where the main process configuration file is located.
           :return: A dictionary containing the merged configuration from all files.
        """
        print("start to load config...")
        with open(os.path.join(dir, "process_config.yaml"), 'r') as f:
            config = yaml.safe_load(f)
        # print(train_data_df.head())

        print("start to load model...")
        with open(os.path.join("recommendation", "properties", "models.yaml"), 'r') as f:
            model_config = yaml.safe_load(f)

        model_path = os.path.join("recommendation", "properties", "models", self.train_config['model'] + ".yaml")
        # if not os.path.exists(model_path):
        #     raise NotImplementedError("we do not support such model type!")
        with open(model_path, 'r') as f:
            model_config.update(yaml.safe_load(f))
        config.update(model_config)

        with open(os.path.join("recommendation", "properties", "evaluation.yaml"), 'r') as f:
            config.update(yaml.safe_load(f))

        config.update(self.train_config)  ###train_config has highest rights
        print("your loading config is:")
        print(config)

        return config

    def rerank(self):
        """
            Training post-processing main workflow.
        """

        dir = os.path.join("recommendation", "processed_dataset", self.dataset)
        config = self.load_configs(dir)

        ranking_score_path = os.path.join("recommendation", "log", config['ranking_store_path'])
        if not os.path.exists(ranking_score_path):
            raise ValueError(f"do not exist the path {ranking_score_path}, please check the path or run the ranking phase to generate scores for re-ranking !")
        print("loading ranking scores....")
        file = os.path.join(ranking_score_path, "ranking_scores.npz")
        ranking_scores = load_npz(file).toarray() #[user_num, item_num]
        ###we need to remove the group do not appear after the ranking phase, for evaluate full group, please evaluate in retrieval stage



        if config['model'] == "CPFair":
            Reranker = CPFair(config)
        elif config['model'] == "FairRec":
            Reranker = FairRec(config)
        elif config['model'] == "FairRecPlus":
            Reranker = FairRecPlus(config)
        elif config['model'] == 'k_neighbor':
            Reranker = k_neighbor(config)
        elif config['model'] == 'min_regularizer':
            Reranker = min_regularizer(config)
        elif config['model'] == 'PMMF':
            Reranker = PMMF(config)
        elif config['model'] == 'Welf':
            Reranker = Welf(config)
        elif config['model'] == 'TaxRank':
            Reranker = TaxRank(config)
        elif config['model'] == 'FairSync':
            Reranker = FairSync(config)
        elif config['model'] == 'RAIF':
            Reranker = RAIF(config)
        else:
            raise NotImplementedError(f"We do not support the model type {self.train_config['model']}")

        #item_scores = np.sum(ranking_scores, axis=0, keepdims=False)

        metrics = ["ndcg", "u_loss"]
        rerank_result = {}
        exposure_result = {}
        for k in config['topk']:
            rerank_result.update({f"{m}@{k}":0 for m in metrics})

            rerank_list = Reranker.rerank(ranking_scores, k)
            exposure_list = np.zeros(config['group_num'])
            for u in range(len(rerank_list)):
                sorted_result_score = np.sort(ranking_scores[u])[::-1]
                true_dcg = dcg(sorted_result_score, k)
                rerank_items = rerank_list[u]

                for i in rerank_items:
                    if i not in Reranker.iid2pid.keys():
                        gid = 0
                    else:
                        gid = Reranker.iid2pid[i]
                    if config['fairness_type'] == "Exposure":
                        exposure_list[gid] += 1
                    else:
                        exposure_list[gid] += np.round(ranking_scores[u][i], config['decimals'])
                reranked_score = ranking_scores[u][rerank_items]
                pre_dcg = dcg(np.sort(reranked_score)[::-1], k)
                rerank_result[f"ndcg@{k}"] += pre_dcg/true_dcg
                rerank_result[f"u_loss@{k}"] += (np.sum(sorted_result_score[:k]) - np.sum(reranked_score[:k]))/k

            rerank_result[f"ndcg@{k}"] /= len(rerank_list)
            rerank_result[f"u_loss@{k}"] /= len(rerank_list)
            for fairness_metric in self.train_config['fairness_metrics']:
                if fairness_metric == 'MinMaxRatio':
                    rerank_result[f"MinMaxRatio@{k}"] = MinMaxRatio(exposure_list)
                elif fairness_metric == 'MMF':
                    rerank_result[f"MMF@{k}"] = MMF(exposure_list)
                elif fairness_metric == 'Entropy':
                    rerank_result[f"Entropy@{k}"] = Entropy(exposure_list)
                elif fairness_metric == 'GINI':
                    rerank_result[f"GINI@{k}"] = Gini(exposure_list)

            #rerank_result[f"mmf@{k}"] = MMF(exposure_list)
            #rerank_result[f"gini@{k}"] = Gini(exposure_list)
            #print(exposure_list)
            exposure_result[f"top@{k}"] = str(list(exposure_list))


        for k in rerank_result.keys():
            rerank_result[k] = np.round(rerank_result[k], config['decimals'])


        today = datetime.today()
        today_str = f"{today.year}-{today.month}-{today.day}"
        log_dir = os.path.join("recommendation", "log", f"{today_str}_{config['log_name']}")

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        with open(os.path.join(log_dir, 'test_result.json'), 'w') as file:
            json.dump(rerank_result, file)
        with open(os.path.join(log_dir, 'exposure_result.json'), 'w') as file:
            json.dump(exposure_result, file)
            #file.write(str(exposure_list))
        #print("exposure list:")
        #print(exposure_list)
        print(rerank_result)

        with open(os.path.join(log_dir, "config.yaml"), 'w') as f:
            yaml.dump(config, f)

        print(f"result and config dump in {log_dir}")






