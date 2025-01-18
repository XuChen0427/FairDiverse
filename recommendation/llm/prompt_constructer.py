
from tqdm import tqdm



class Prompt_Constructer(object):
    def __init__(self, item_feature_list=['id','name','publisher'], item_domain='game',
                 history_behavior_field='history_behaviors',
                 item_candidate_field='items',
                 pos_length_field='pos_length'):
        self.item_feature_list = item_feature_list
        self.item_domain = item_domain
        self.history_behavior_field = history_behavior_field
        self.item_candidate_field = item_candidate_field
        self.pos_length_field = pos_length_field
        # history_behavior_field: history_behaviors,
        # item_candidate_field: items,
        # pos_length_field: pos_length,

    def construct_inter_dict(self, input_file, iid2title, iid2cate):
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

        # print(cates_name)
        data_dict = self.construct_inter_dict(input_file, iid2title, iid2pid)
        prompt_dataset_json = self.data_to_json(data_dict, iid2pid)
        return prompt_dataset_json

    def data_to_json(self, data_dict, iid2pid):
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
            json_list.append({
                "instruction": f"You are a {self.item_domain} recommender. Given a list of {self.item_domain} the user has clicked before, please recommend a item that the user may like.",
                "input": f"{history}",
                # "output": target_movie_str,
                'item_candidates': feats['item_candidates'],
                'positive_items': feats['positive_items'],
                'negative_items': feats['negative_items'],
                # "sensitiveAttribute": iid2pid[target_item]
            })
        return json_list