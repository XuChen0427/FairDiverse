import os

import pandas as pd
import yaml

from datasets.COMPAS import Compas
from preprocessing_model import fairness_method_mapping
from ranker_model import ranker_mapping


class RankerTrainer(object):
    def __init__(self, train_config):
        self.train_config = train_config
        self.load_configs()

    def load_configs(self):
        # Load the configs for the dataset and for the pre-processing method
        with open(os.path.join(self.train_config['task'], "properties", "dataset",
                               self.train_config['dataset'].lower() + ".yaml"), 'r') as f:
            data_config = yaml.safe_load(f)
        self.train_config["data_reader_class"] = data_config["data_reader_class"]

        print(self.train_config['train_ranker_config']['preprocessing_model'])
        with open(os.path.join(self.train_config['task'], "properties", "models",
                               self.train_config['train_ranker_config']['preprocessing_model'] + ".yaml"),
                  'r') as f:
            model_config = yaml.safe_load(f)
        self.train_config["pre_processing_config"] = model_config["pre_processing_config"]

        print(self.train_config)

    def train(self):
        # train the ranker with the pre-processing method
        if self.train_config["data_reader_class"]["name"] == "COMPAS":
            dataset = Compas(self.train_config["data_reader_class"])

        if "pre_processing_config" in self.train_config:
            fairness_method = fairness_method_mapping[self.train_config["pre_processing_config"]["name"]](
                self.train_config["pre_processing_config"], dataset)

        ranker = ranker_mapping[self.train_config['train_ranker_config']['name']](
            self.train_config['train_ranker_config'], dataset)

        for run in range(dataset.k_fold):
            data_train, data_test = dataset.read_data(run)

            if "pre_processing_config" in self.train_config:
                if self.train_config["pre_processing_config"]["qid_process"]:
                    data_train_fair = []
                    data_test_fair = []
                    qids = data_train[dataset["query_col"]].unique()
                    for qid in qids:
                        root_model_path = fairness_method.model_path.copy()
                        fairness_method.model_path = os.path.join(root_model_path, str(run), str(qid))

                        data_qid_train = data_train[dataset["query_col"] == qid]
                        data_qid_test = data_test[dataset["query_col"] == qid]

                        fairness_method.fit(data_qid_train, str(run))

                        data_qid_train_fair = fairness_method.transform(data_qid_train, str(run))
                        data_qid_test_fair = fairness_method.transform(data_qid_test, str(run))

                        data_train_fair.append(data_qid_train_fair)
                        data_test_fair.append(data_qid_test_fair)

                    data_train_fair = pd.concat(data_train_fair)
                    data_train_fair.to_csv(os.path.join(fairness_method.fair_data_path, 'fair_train_data.csv'))

                    data_test_fair = pd.concat(data_test_fair)
                    data_test_fair.to_csv(os.path.join(fairness_method.fair_data_path, 'fair_test_data.csv'))
                else:

                    fairness_method.fit(data_train, str(run))
                    data_train_fair = fairness_method.transform(data_train, str(run), "train")
                    data_test_fair = fairness_method.transform(data_test, str(run), "test")

                ranker_train_data = data_train_fair
                ranker_test_data = data_test_fair
            else:
                ranker_train_data = data_train
                ranker_test_data = data_test
            ranker.train(ranker_train_data, ranker_test_data, run)
            prediction = ranker.predict(data_test_fair, run, "test")

            ranker.evaluate_run(prediction, run, "test")

        ranker.evaluate("test")  # computes mean over runs
