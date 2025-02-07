import os.path
import numpy as np
from tqdm import tqdm, trange
from ..utils.group_utils import get_cos_similar_torch
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class Grounder(object):
    def __init__(self, config):
        # self.config
        self.grounding_model = config['grounding_model']
        self.llm_path_dict = config['llm_path_dict']
        print(type(self.grounding_model))
        assert self.grounding_model in self.llm_path_dict.keys(), f"Grounding Model {self.grounding_model} Not Found."
        self.grounding_model_path = self.llm_path_dict[self.grounding_model]
        self.ground_in_8bit = config['use_8bit']
        self.saved_embs_filename = os.path.join("recommendation", "llm", "stored_embs", config['saved_embs_filename']) if config['saved_embs_filename'] else None
        self.grounding_batch_size = config['batch_size']
        self.device_map = config['device_map']
        self.device = config['device']

    def load_model_tokenizer(self):
        """
        Loads the tokenizer and model for text generation based on the specified grounding model.

        This method initializes the tokenizer and model required for text generation tasks. It checks if the provided
        grounding model exists within the pre-defined dictionary of model paths. If found, it proceeds to load the tokenizer
        and model using the Hugging Face Transformers library. The tokenizer's padding side is set to 'left', and the model
        is configured with options such as load_in_8bit, torch_dtype, and device_map for efficient memory usage and deployment.
        Additional model configurations are set, including token ID assignments for pad, bos, eos tokens, and enabling
        output of hidden states. If the specified grounding model is not recognized, an exception is raised.

        Parameters:
        - self.grounding_model (str): The name or identifier of the grounding model to be loaded.
        - self.llm_path_dict (dict): A dictionary mapping model names to their respective local paths.
        - self.grounding_model_path (str): The path where the model is located, derived from `self.llm_path_dict`.
        - self.ground_in_8bit (bool): A flag indicating whether to load the model in 8-bit precision for memory efficiency.
        - self.device_map (str or dict): Specifies how to map model’s tensors to devices; can be a string or a dictionary.

        Raises:
        - Exception: If the `grounding_model` is not a key in `self.llm_path_dict`, indicating an unsupported model.

        Note:
        - Ensure that the `self.llm_path_dict` contains the correct paths for each model before calling this method.
        - Model loading can be resource-intensive; ensure sufficient system resources are available.
        """
        if self.grounding_model in self.llm_path_dict.keys():
            self.tokenizer = AutoTokenizer.from_pretrained(self.grounding_model_path)
            self.tokenizer.padding_side = "left"
            self.model = AutoModelForCausalLM.from_pretrained(
                self.grounding_model_path,
                load_in_8bit=self.ground_in_8bit,
                torch_dtype='auto',
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
        """
        Get the embedding of the input text using the model at a specified hidden layer index.

        Parameters
        - text : str
            The input text to be embedded.
        - index : int
            The index of the hidden layer from which to extract the embeddings.

        Returns
        - embedding : torch.Tensor
            The averaged embedding vector for the input text.
        """
        tokens = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(self.device)
        input_ids = tokens["input_ids"]
        attention_mask = tokens['attention_mask']
        with torch.no_grad():
            output = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            embedding = output['hidden_states'][index]
            embedding = torch.mean(embedding, dim=1, keepdim=False)
        return embedding

    def get_title_embedding(self, titles, index):
        """
        Get title embeddings using a model's embedding generation capability.

        This method retrieves embeddings for a list of titles. It first checks if a saved embeddings file exists,
        and if so, loads the embeddings from it. If not, it generates the embeddings in batches using the
        `get_embedding` method. The generated or loaded embeddings are then returned as a PyTorch tensor.

        Parameters
        ----------
        titles : List[str]
            A list of strings representing the titles for which embeddings are to be obtained.
        index : int
            An index indicating a specific configuration or identifier used in the embedding process.

        Returns
        -------
        torch.Tensor
            A tensor containing the embeddings for the provided titles. Each row corresponds to the embedding of a title.

        Notes
        -----
        - If `saved_embs_filename` is set and the file exists, embeddings are directly loaded from it, bypassing computation.
        - Embeddings are computed in batches to manage memory efficiently, with the batch size defined by `grounding_batch_size`.
        - If embeddings are computed, they can optionally be saved to `saved_embs_filename` for future reuse.
        """
        if self.saved_embs_filename and os.path.exists(self.saved_embs_filename):
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
            if self.saved_embs_filename:
                torch.save(embs, self.saved_embs_filename)
        return embs

    def map_titles(self, titles, o_emb, index, candidates):
        """
        Map titles to their similarity scores with a given object embedding.

        This method computes the cosine similarity between the embeddings of a list of titles and a given object embedding.
        It processes the titles in batches, calculates their embeddings using `get_embedding` method, and then assesses
        the similarity with the object embedding. The output is a list of top-k similar titles' indices for each set of candidate
        titles provided, along with the corresponding similarity scores.

        Parameters
        ----------
        titles : List[str]
            A list of titles whose embeddings are to be computed and compared.

        o_emb : torch.Tensor
            The embedding vector of the object against which the title embeddings will be compared.

        index : Any
            The index or data structure used by `get_embedding` to fetch or compute embeddings.

        candidates : List[List[int]]
            A list of lists, where each sublist contains indices representing candidate titles for a user.

        Returns
        -------
        Tuple[List[List[int]], List[List[float]]]
            A tuple containing two elements:
            - The first element is a list of lists, where each sublist contains the indices of top-k most similar titles
              for each set of candidates.
            - The second element is a list of lists, where each sublist contains the similarity scores corresponding to the
              top-k titles' indices.

        """
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
        for t_emb, cand in tqdm(zip(t_embs, candidates), desc='map titles'):  # Iterate through each user’s predictions.
            cos_sim = get_cos_similar_torch(t_emb, o_emb, device=self.device)  # Calculate the similarity between each prediction and all the titles.
            # print(cos_sim.shape)
            cos_sim = cos_sim[cand]
            # print(cand)
            sorted_elements_with_indices = sorted(zip(cos_sim, cand))[::-1]
            score = [element for element, index in sorted_elements_with_indices]
            topk_list = [index for element, index in sorted_elements_with_indices]
            scores.append(score)
            result.append(topk_list)
        return result, scores

    def get_ranking_itemlist(self, response_result, o_emb, index):
        """
        Processes the response data to extract item candidates and their predicted scores. It utilizes the provided embeddings and index to map and score the candidates, returning a list of ranked items along with their respective scores.

        Parameters:
        - response_result (List[Dict[str, Any]]): A list of dictionaries containing user-specific 'item_candidates' and 'predict' values.
        - o_emb (np.ndarray): The embedding matrix used for scoring the item candidates.
        - index (Any): The index object that aids in mapping and retrieving candidate information efficiently.

        Returns:
        - Tuple[List[str], List[float]]: A tuple containing two lists:
            - The first list contains the top-ranked item identifiers based on the prediction scores.
            - The second list holds the corresponding prediction scores for each item in the ranked list.
        """
        candidates = [user['item_candidates'] for user in response_result]
        predict = [user['predict'] for user in response_result]
        predict, scores = self.map_titles(predict, o_emb, index, candidates)
        return predict, scores

    def grounding(self, response_results, id2title):
        """
        Grounding method to enhance response results with predicted item rankings and scores.

        This method enhances a list of response results by grounding them with relevant
        item predictions and associated scores. It utilizes a pre-trained model to generate
        embeddings for a provided list of titles and then ranks items within each response
        result based on their similarity to these embeddings.

        Parameters:
        - response_results (List[Dict]): A list of dictionaries, each representing a response result.
        - id2title (Dict[int, str]): A dictionary mapping IDs to their corresponding title names.

        Returns:
        - List[Dict]: The updated list of response results, where each result now includes
          a 'predict_list' key holding the predicted item IDs and a 'scores' key with their
          respective similarity scores.
        """
        # title2id = {v: k for k, v in id2title.items()}
        title_name = list(id2title.values())
        self.load_model_tokenizer()

        title_emb = self.get_title_embedding(title_name, -1)  #Get the embeddings of the last layer for all titles.
        predict, scores = self.get_ranking_itemlist(response_results, title_emb, -1)
        for idx, res in enumerate(response_results):
            res['predict_list'] = predict[idx]
            # res['prediction_list'] = [title2id[i] for i in predict[idx]]
            res['scores'] = scores[idx]
        # print(f'response_result:{response_results}')
        return response_results