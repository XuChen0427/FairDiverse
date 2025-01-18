import os.path
import numpy as np
from tqdm import tqdm, trange
from ..utils.group_utils import get_cos_similar_torch
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class Grounder(object):
    def __init__(self, grounding_model, grounding_model_path,grounding_batch_size=32,
                 use_8bit=False, saved_embs_filename=None, device='cuda', device_map='auto'):
        # self.config = config
        self.grounding_model = grounding_model
        self.grounding_model_path = grounding_model_path
        self.ground_in_8bit = use_8bit
        self.saved_embs_filename = saved_embs_filename
        self.grounding_batch_size = grounding_batch_size
        self.device_map = device_map
        self.device = device

    def load_model_tokenizer(self):
        if self.grounding_model in ['Llama3-8B-Instruct', 'bert', 'gpt2']:
            self.tokenizer = AutoTokenizer.from_pretrained(self.grounding_model_path)
            self.tokenizer.padding_side = "left"
            self.model = AutoModelForCausalLM.from_pretrained(
                self.grounding_model_path,
                load_in_8bit=self.ground_in_8bit,
                torch_dtype=torch.float16,
                device_map=self.device_map,
            )
        else:
            raise Exception('Illegal Grouding Model Defined.')
        self.model.config.pad_token_id = self.tokenizer.pad_token_id = 0  # unk
        self.model.config.bos_token_id = 1
        self.model.config.eos_token_id = 2
        self.model.config.output_hidden_states = True
        # self.model.half()

    def get_embedding(self, text, index):
        tokens = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(self.device)
        input_ids = tokens["input_ids"]
        attention_mask = tokens['attention_mask']
        with torch.no_grad():
            output = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            embedding = output['hidden_states'][index]
            embedding = torch.mean(embedding, dim=1, keepdim=False)
        return embedding

    def get_title_embedding(self, titles, index):
        if os.path.exists(self.saved_embs_filename):
            embs = torch.load(self.saved_embs_filename)
        else:
            embs = []
            batch_size = self.grounding_batch_size
            for b in trange(int(np.ceil(len(titles) / batch_size)), desc='Get Title Embedding'):
                min_id = b * batch_size
                max_id = min((b + 1) * batch_size, len(titles))
                text = titles[min_id:max_id]
                emb = self.get_embedding(text, index)
                embs.append(emb)
            embs = torch.cat(embs, dim=0)
            torch.save(embs, self.saved_embs_filename)
        return embs

    def map_titles(self, titles, id2title, o_emb, index, candidates):
        result = []
        batch_size = self.grounding_batch_size
        t_embs = []
        for b in trange(int(np.ceil(len(titles) / batch_size)), desc='Item Embedding'):
            min_id = b * batch_size
            max_id = min((b + 1) * batch_size, len(titles))
            text = titles[min_id:max_id]
            emb = self.get_embedding(text, index)
            t_embs.append(emb)
        t_embs = torch.cat(t_embs, dim=0)
        scores = []
        for t_emb, cand in tqdm(zip(t_embs, candidates), desc='map titles'):  # 遍历每一个用户的predict
            cos_sim = get_cos_similar_torch(t_emb, o_emb, device=self.device)  # 计算每一个predict和 所有title的相似度
            # print(cos_sim.shape)
            cos_sim = cos_sim[cand]
            # print(cand)
            sorted_elements_with_indices = sorted(zip(cos_sim, cand))[::-1]
            score = [element for element, index in sorted_elements_with_indices]
            topk_list = [index for element, index in sorted_elements_with_indices]
            # print(topk_list)
            # topk_list = np.argsort(cos_sim)[::-1]  # 前100个time
            # score = cos_sim[topk_list]
            scores.append(score)
            result.append(topk_list)
        return result, scores

    def get_ranking_itemlist(self, response_result, o_emb, id2title, index):
        # output = [response_result[i]['output'] for i in range(len(response_result))]  # 每个用户u的ground truth
        # title2id = {v: k for k, v in id2title.items()}
        # ground_truth_ids = [user['positive_items'] for user in response_result] # [title2id[ti] for ti in output]
        candidates = [user['item_candidates'] for user in response_result]
        predict = [user['predict'] for user in response_result]
        predict, scores = self.map_titles(predict, id2title, o_emb, index, candidates)
        return predict, scores

    def grounding(self, response_results, id2title):

        title2id = {v: k for k, v in id2title.items()}
        title_name = list(id2title.values())
        self.load_model_tokenizer()

        title_emb = self.get_title_embedding(title_name, -1)  #取所有title的最后一层的embedding
        predict, scores = self.get_ranking_itemlist(response_results, title_emb, id2title, -1)
        for idx, res in enumerate(response_results):
            res['predict_list'] = predict[idx]
            # res['prediction_list'] = [title2id[i] for i in predict[idx]]
            res['scores'] = scores[idx]
        # print(f'response_result:{response_results}')
        return response_results