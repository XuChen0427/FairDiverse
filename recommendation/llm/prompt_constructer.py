
from tqdm import tqdm



class Prompt_Constructer(object):
    def __init__(self, config):
        self.item_feature_list = ['id','title','publisher']
        self.item_domain = config['item_domain']
        self.fair_prompt = config['fair_prompt']
        self.history_behavior_field = 'history_behaviors'
        self.item_candidate_field = 'items'
        self.pos_length_field = 'pos_length'

    def construct_inter_dict(self, input_file, iid2title, iid2cate):
        """
        Constructs an interaction dictionary from the given input data.

        This method processes an input file, typically a DataFrame, to create a dictionary representation of user interactions.
        It extracts user history, candidate items, positive and negative samples for recommendation tasks. The method also
        enriches item information using provided mappings for item titles and categories.

        Parameters:
        :param input_file (pandas.DataFrame): The DataFrame containing user interaction data with necessary fields.
        :param iid2title (dict): A mapping of item IDs to their respective titles.
        :param iid2cate (dict): A mapping of item IDs to their respective categories.

        Returns:
        :returns data_dict (dict): A dictionary where each key is a user ID and the value is another dictionary containing:
            - 'history_items' (list[dict]): List of dictionaries representing historical items interacted by the user,
              each with 'id', 'title', and 'publisher' keys.
            - 'item_candidates' (list): List of candidate items for the user.
            - 'positive_items' (list): Subset of 'item_candidates' up to the positive sample length, indicating observed interactions.
            - 'negative_items' (list): Remaining items in 'item_candidates', considered as negative samples.
        """
        data_dict = {}
        print(input_file.head)
        for index, row in input_file.iterrows():
            userid = row['user_id']
            history_itemid_list = eval(row[self.history_behavior_field])
            item_candidate = eval(row[self.item_candidate_field])
            pos_len = int(row[self.pos_length_field])
            data_dict[userid] = {
                'history_items': [],
                'item_candidates': item_candidate,
                'positive_items': item_candidate[:pos_len],
                'negative_items': item_candidate[pos_len:]
            }
            for item_id in history_itemid_list:
                item_feat_dict = {'id': item_id, # item id
                                  'title': iid2title.get(item_id, 'unknown'),
                                  'publisher': iid2cate.get(item_id, 'unknown')
                                  }
                data_dict[userid]['history_items'].append(item_feat_dict)
        return data_dict

    def construct_prompt(self, input_file, iid2title, iid2pid):
        """
        Constructs a prompt dataset from the given input.

        Parameters:
        - input_file (str): The path to the input file containing necessary data for constructing the prompts.
        - iid2title (dict): A dictionary mapping identifiers to their respective titles, enhancing context in prompts.
        - iid2pid (dict): A dictionary associating identifiers with parent identifiers, adding hierarchical information.

        Returns:
        - str: A JSON string representing the constructed dataset, formatted suitably for use as prompts.

        """
        # print(cates_name)
        data_dict = self.construct_inter_dict(input_file, iid2title, iid2pid)
        prompt_dataset_json = self.data_to_json(data_dict)
        return prompt_dataset_json

    def data_to_json(self, data_dict):
        """
        Converts a dictionary of user data into a JSON formatted list for recommendation tasks.

        Parameters:
        - data_dict (Dict[str, Dict]): A dictionary where keys are user identifiers and values are dictionaries containing
          'history_items' (a list of items viewed by the user), 'item_candidates' (a list of candidate items for recommendation),
          'positive_items' (items liked by the user, if any), and 'negative_items' (items not liked by the user, if any).

        Returns:
        - List[Dict]: A list of dictionaries, each representing a recommendation task with structured information including
          instruction, input context based on user history, and candidate items for generating recommendations.
        """
        json_list = []
        for user, feats in tqdm(data_dict.items()):
            history = f"The user has viewed the following {self.item_domain}s before, with features as: "
            for item in feats['history_items']:
                # history += f"Item id: {item['item_id']}:"
                for feat in self.item_feature_list:
                    history += f"Item {feat}: {item[feat]},"
                history += '\n'
            # target_item = feats['positive_items']
            # target_movie_str = "" + str(target_item) + ""
            instruction = f"You are a {self.item_domain} recommender. Given a list of {self.item_domain} the user has clicked before, please recommend a item that the user may like."
            instruction += self.fair_prompt if self.fair_prompt else f""
            json_list.append({
                "instruction": instruction,
                "input": f"{history}",
                # "output": target_movie_str,
                'item_candidates': feats['item_candidates'],
                'positive_items': feats['positive_items'],
                'negative_items': feats['negative_items'],
                # "sensitiveAttribute": iid2pid[target_item]
            })
        return json_list