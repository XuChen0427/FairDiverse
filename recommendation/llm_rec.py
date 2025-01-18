import os.path
from .utils.group_utils import Init_Group_AdjcentMatrix, get_iid2text
import pandas as pd
from .llm import Evaluator, Grounder, Prompt_Constructer, LLM_caller
import os

class LLMRecommender(object):
    """

    """
    def __init__(self, dataset, stage, train_config):
        self.dataset = dataset
        self.stage = stage
        self.config = train_config



    def recommend(self):
        dataset_dir = os.path.join("recommendation", "processed_dataset", self.dataset)
        dataset_file_name = self.dataset + '.test.ranking'
        input_file = pd.read_csv(os.path.join(dataset_dir, dataset_file_name), delimiter='\t')
        iid2text, iid2pid = get_iid2text(self.dataset), Init_Group_AdjcentMatrix(self.dataset)

        prompt_constructer = Prompt_Constructer(item_feature_list=self.config['item_feature_list'],
                                                item_domain=self.config['item_domain'],
                                                history_behavior_field=self.config['history_behavior_field'],
                                                item_candidate_field=self.config['item_candidate_field'],
                                                pos_length_field=self.config['pos_length'])
        prompt_dataset = prompt_constructer.construct_prompt(input_file, iid2text, iid2pid)

        LLM = LLM_caller(llm=self.config['llm'], llm_path=self.config['llm_path'],
                         device=self.config['llm_path'], api_key=self.config['api_key'],
                         api_base=self.config['api_key'], max_tokens=self.config['max_tokens'],
                         temperature=self.config['temperature'])

        results_list = LLM.get_response(prompt_dataset)

        grounder = Grounder(grounding_model=self.config['grounding_model'],
                            grounding_model_path=self.config['grounding_model_path'],
                            grounding_batch_size=self.config['grounding_batch_size'],
                            use_8bit=self.config['use_8bit'],
                            saved_embs_filename=self.config['saved_embs_filename'],
                            device=self.config['grounding_device'],
                            device_map=self.config['grounding_device_map'])

        grounding_result = grounder.grounding(results_list, id2title=iid2text)

        evaluator = Evaluator(self.config['metric_list'], self.config['TopK'])

        evaluator.evaluate(grounding_result, iid2pid)






