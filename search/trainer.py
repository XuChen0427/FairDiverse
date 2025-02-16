import os
import yaml


class SRDTrainer(object):
    def __init__(self, train_config):
        """
        Initialize post-processing and base models.
        :param train_config: Your custom config files.
        """

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

        print("start to load dataset config...")
        with open(os.path.join(self.train_config['task'], "properties", "dataset", self.train_config['dataset'].lower() + ".yaml"), 'r') as f:
            config = yaml.safe_load(f)
        config.update({'data_dir': dir})

        print("start to load model config...")

        with open(os.path.join(self.train_config['task'], "properties", "models", self.train_config['model'].lower() + ".yaml"),
                  'r') as f:
            model_config = yaml.safe_load(f)

        config.update(model_config)

        with open(os.path.join("recommendation", "properties", "evaluation.yaml"), 'r') as f:
            config.update(yaml.safe_load(f))

        config.update(self.train_config)  ###train_config has highest rights
        print("your loading config is:")
        print(config)

        return config


    def train(self):
        """
        Training post-processing search model main workflow.
        """

        dir = os.path.join(self.train_config['task'], "processed_dataset", self.train_config['dataset'])
        config = self.load_configs(dir)

        if os.path.exists(os.path.join(config['task'], "processed_dataset", config['dataset'], config['model'])) and config['reprocess'] == False:
            print("Data has been processed, start to load the dataset...")
        else:
            print("start to process data...")
            if os.path.join(config['data_dir'], 'div_query.data') not in config['data_dir']:
                from .utils.process_dataset import data_process
                data_process(config)
            if config['model'].lower() == 'desa':
                from .utils.process_desa import Process
                Process(config)
            elif config['model'].lower() == 'daletor':
                from .utils.process_daletor import Process
                Process(config)
            elif config['model'].lower() == 'xquad':
                from .utils.process_bm25 import generate_bm25_scores_for_query
                generate_bm25_scores_for_query(config)
            elif config['model'].lower() == 'pm2':
                from .utils.process_bm25 import generate_bm25_scores_for_query
                generate_bm25_scores_for_query(config)
            elif config['model'].lower() == 'llm':
                pass
            else:
                raise NotImplementedError(f"Not supported model type: {config['model']}")

        print("start to load dataset......")
        self.device = config['device']
        if config['mode'] == 'test' and config['best_model_list'] != []:
            print("start to test the model...")
            from .evaluator import get_global_fullset_metric
            get_global_fullset_metric(config)
        elif config['mode'] == 'train':
            """
            For implementing your own supervised methods, you need to first re-write xxx_run and then implement your own model.
            You can also just copy the two example xxx_run implementation for quick starting.
            For the unsupervised methods, just implment the model function.
            """

            if config['model'].lower() == 'desa':
                from .datasets.DESA import DESA_run
                DESA_run(config)
            elif config['model'].lower() == 'daletor':
                from .datasets.DALETOR import DALETOR_run
                DALETOR_run(config)
            elif config['model'].lower() == 'xquad':
                from .postprocessing_model.xQuAD import xQuAD
                xquad = xQuAD()
                xquad.rerank(config)
            elif config['model'].lower() == 'pm2':
                from .postprocessing_model.PM2 import PM2
                pm2 = PM2()
                pm2.rerank(config)
            elif config['model'].lower() == 'llm':
                from .datasets.LLM import llm_run
                llm_run(config)
            else:
                raise NotImplementedError(f"Not supported model type: {config['model']}")



