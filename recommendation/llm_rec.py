import os.path
from .utils.group_utils import Init_Group_AdjcentMatrix, get_iid2text
import pandas as pd
from .llm import Grounder, Prompt_Constructer, LLM_caller
from .evaluator import LLM_Evaluator
import os
import yaml
import json
from datetime import datetime

class LLMRecommender(object):
    """

    """
    def __init__(self, dataset, stage, train_config):
        """Initialize In-processing and base models.

            :param dataset: utilized dataset.
            :param stage: In-processing or Post-processing stage.
            :param train_config: Your custom config files.
        """
        self.dataset = dataset
        self.stage = stage
        self.llm_type = train_config['llm_type']
        self.config = train_config

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

        with open(os.path.join("recommendation", "properties", "dataset", f"{self.dataset}.yaml"),
                  'r') as f:
            config.update(yaml.safe_load(f))

        print("start to load model...")
        with open(os.path.join("recommendation", "properties", "models.yaml"), 'r') as f:
            model_config = yaml.safe_load(f)

        with open(os.path.join("recommendation", "properties", "models", f"LLM.yaml"),
                  'r') as f:
            model_config.update(yaml.safe_load(f))

        config.update(model_config)

        with open(os.path.join("recommendation", "properties", "evaluation.yaml"), 'r') as f:
            config.update(yaml.safe_load(f))

        config.update(self.config)  ###train_config has highest rights
        print("your loading config is:")
        print(config)

        return config

    def recommend(self):
        """
            Training LLMs-based in-processing and base model main workflow.
        """

        dataset_dir = os.path.join("recommendation", "processed_dataset", self.dataset)
        dataset_file_name = self.dataset + '.test.ranking'
        input_file = pd.read_csv(os.path.join(dataset_dir, dataset_file_name), delimiter='\t')
        iid2text, iid2pid = get_iid2text(self.dataset), Init_Group_AdjcentMatrix(self.dataset)
        config = self.load_configs(dataset_dir)
        prompt_constructer = Prompt_Constructer(config)
        prompt_dataset = prompt_constructer.construct_prompt(input_file, iid2text, iid2pid)

        LLM = LLM_caller(config)

        results_list = LLM.get_response(prompt_dataset)
        LLM.clear()
        grounder = Grounder(config)

        grounding_result = grounder.grounding(results_list, id2title=iid2text)

        evaluator = LLM_Evaluator(config)

        eval_result = evaluator.llm_eval(grounding_result, iid2pid)

        # dump the logs and eval result
        today = datetime.today()
        today_str = f"{today.year}-{today.month}-{today.day}"
        log_dir = os.path.join("recommendation", "log", f"{today_str}_{config['log_name']}")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)


        print(f"training complete! start to save the config and model...")
        print(f" config files are dump in {log_dir}")
        with open(os.path.join(log_dir, "config.yaml"), 'w') as f:
            yaml.dump(config, f)


        with open(os.path.join(log_dir, 'test_result.json'), 'w') as f:
            json.dump(eval_result, f)
        print(f"dump in {log_dir}")


