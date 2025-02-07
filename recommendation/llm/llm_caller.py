from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


class LLM_caller(object):
    def __init__(self, config):
        self.config = config
        self.llm_type = config['llm_type']
        self.device = config['device']
        if self.llm_type == 'api':
            self.init_llm_api(config['llm_name'], config['api_key'], config['api_base'],
                              config['max_tokens'], config['temperature'])
        elif self.llm_type == 'local':
            self.init_llm_local(config['llm_name'], config['llm_path_dict'],
                                config['max_tokens'], config['use_8bit'], config['device_map'])
        elif self.llm_type == 'vllm':
            self.init_llm_vllm(config['llm_name'], config['llm_path_dict'],
                                config['temperature'], config['max_tokens'])
        else:
            raise ValueError('llm type not found.')

    def clear(self):
        if self.llm_type == 'vllm':
            del self.vllm
            del self.tokenizer
        if self.llm_type =='local':
            del self.model
            del self.tokenizer


    def init_llm_api(self, llm_name, api_key, api_base='EMPTY', max_tokens=256, temperature=0.8):
        print(f'use api llm for generating...')
        self.llm_name = llm_name
        self.api_key = api_key
        self.api_base = api_base
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.llm_func = self.GetResultFromGPT


    def init_llm_local(self, llm_name, llm_path_dict, max_tokens=512, use_8bit=False, device_map='auto'):
        """
        Initialize a local language model (LLM) for text generation.

        This method sets up a local instance of a causal language model using Hugging Face's transformers library. It loads the model and tokenizer from the provided path, with options for controlling memory usage and device allocation.

        Parameters:
        :param llm_name: `str`
            The name of the language model to initialize. Must correspond to a key in the `llm_path_dict`.
        :param llm_path_dict: `Dict[str, str]`
            A dictionary mapping model names to their local directory paths.
        :param max_tokens: `int`, optional
            The maximum number of tokens to generate. Defaults to 512.
        :param use_8bit: `bool`, optional
            Whether to use 8-bit quantization for loading the model, reducing memory usage at the cost of speed. Defaults to `False`.
        :param device_map: `str`, optional
            Specifies how to allocate the model across devices. 'auto' automatically balances the model on available devices. Custom mappings can also be defined. Defaults to 'auto'.

        Raises:
            AssertionError: If the specified `llm_name` is not found in `llm_path_dict`.

        Attributes Set:
            :self.llm_name: `str`
                Name of the initialized LLM.
            :self.llm_path: `str`
                Local path to the initialized LLM.
            :self.max_tokens: `int`
                Maximum token generation limit.
            :self.model: `transformers.AutoModelForCausalLM`
                The loaded causal language model.
            :self.tokenizer: `transformers.AutoTokenizer`
                The tokenizer paired with the loaded model.
            :self.llm_func: `Callable`
                A reference to the method for generating results using the initialized LLM.
        """
        print(f'use local llm for generating...')
        assert llm_name in llm_path_dict.keys(), f"LLM {llm_name} Path Not Found"
        self.llm_name = llm_name
        self.llm_path = llm_path_dict[llm_name]
        self.max_tokens = max_tokens
        self.model = AutoModelForCausalLM.from_pretrained(
            self.llm_path,
            load_in_8bit=use_8bit,
            torch_dtype="auto",
            device_map=device_map,
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_path, trust_remote_code=True)
        self.llm_func = self.GetResultFromHuggingFace

    def init_llm_vllm(self, llm_name, llm_path_dict, temperature=0.8, max_tokens=256):
        """
        Initialize the Language Model using vLLM for text generation.

        This method sets up the vLLM framework to utilize a specified language model for generating text. It configures
        sampling parameters like temperature and maximum tokens, selects the tokenizer corresponding to the language model,
        and determines whether to use batch processing based on the provided configuration.

        Parameters:
        :param llm_name: (str) The name of the language model to be initialized.
        :param llm_path_dict: (dict) A dictionary mapping language model names to their respective local paths.
        :param temperature: (float, optional) The sampling temperature for generated text. Defaults to 0.8.
        :param max_tokens: (int, optional) The maximum number of tokens to generate. Defaults to 256.

        Attributes Set:
        :self.use_batch: (bool) Indicates if batch processing should be used, based on the configuration.
        :self.llm_path: (str) The path to the language model files.
        :self.tokenizer: (PreTrainedTokenizer) The tokenizer associated with the selected language model.
        :self.llm_name: (str) The name of the initialized language model.
        :self.vllm: (LLM) An instance of vLLM's LLM class configured with the model details.
        :self.sampling_params: (SamplingParams) Sampling parameters for text generation.
        :self.llm_func: (function) Determines the function for getting results, either batch or non-batch, based on :self.use_batch:.

        Raises:
        :FileNotFoundError: If the specified language model path does not exist.
        :ValueError: If the tokenizer fails to initialize due to incorrect configuration or missing remote code trust.
        """
        print(f'use vllm for generating...')
        from vllm import LLM, SamplingParams
        self.use_batch = self.config['use_batch']
        self.llm_path = llm_path_dict[llm_name]
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_path, trust_remote_code=True)
        self.llm_name = llm_name
        self.vllm = LLM(model=self.llm_path, max_model_len=8192)
        self.sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
        self.llm_func = self.GetBatchResultFromVllm if self.use_batch else self.GetResultFromVllm

    def GetBatchResultFromVllm(self, data):
        """
        This method processes a batch of data to generate text using the VLLM engine, with customization based on the LLM name.

        Parameters
        - data : List[Dict[str, str]]
            A list of dictionaries, each containing 'input' and 'instruction' keys for generating text.

        Returns
        - List[str]
            A list of generated texts corresponding to the input data.
        """
        if 'Mistral' in self.llm_name:
            messages_list = [[
                {"role": "user", "content": prompt_dict['input'] + '\n'+ prompt_dict['instruction']},
            ] for prompt_dict in data]
        else:
            messages_list = [[
                {"role": "system", "content": prompt_dict['input']},
                {"role": "user", "content": prompt_dict['instruction']},
            ] for prompt_dict in data]
        prompt_ids = [self.tokenizer.apply_chat_template(messages, add_generation_prompt=True) for messages in messages_list ]
        outputs = self.vllm.generate(prompt_token_ids=prompt_ids,
                           sampling_params=self.sampling_params,
                            use_tqdm= False
                           )
        generated_text = [output.outputs[0].text for output in outputs]
        return generated_text


    def GetResultFromVllm(self, prompt_dict):
        """
        Obtains a generated text result from the VLLM model given a dictionary containing input and instruction data.

        Parameters:
        - prompt_dict (dict): A dictionary carrying two keys:
            - 'input' (str): The context or initial information for the conversation.
            - 'instruction' (str): The specific directive or query for the model to act upon.

        Returns:
        - str: The generated text output from the VLLM model corresponding to the provided inputs.

        Note:
        - Ensure `self.tokenizer` and `self.vllm` are properly initialized and configured before calling this method.
        - The `sampling_params` should be set to appropriate values within the class instance for desired generation behavior.
        """
        messages = [
            {"role": "system", "content": prompt_dict['input']},
            {"role": "user", "content": prompt_dict['instruction']},
        ]
        prompt_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        outputs = self.vllm.generate(prompt_token_ids=[prompt_ids],
                           sampling_params=self.sampling_params,
                            use_tqdm= False
                           )
        print(outputs[0].outputs[0].text)
        return outputs[0].outputs[0].text

    def GetResultFromHuggingFace(self, prompt_dict):
        """
        GetResultFromHuggingFace(self, prompt_dict: dict) -> str

        Generates a response from a Hugging Face language model based on the provided prompt dictionary.

        This method constructs a message context using the 'input' and 'instruction' keys from the `prompt_dict`.
        It applies a chat template to format the messages, tokenizes them, and prepares the input for the model.
        The model's generate function is then used to produce new text, which is decoded and returned as the final response.

        Parameters:
        - **prompt_dict** (dict): A dictionary containing two keys:
          - 'input' (str): The initial context or scenario for the conversation.
          - 'instruction' (str): The user's instruction or query within the given context.

        Returns:
        - **response** (str): The generated text output by the language model in response to the prompt.
        """
        messages = [
            {"role": "system", "content": prompt_dict['input']},
            {"role": "user", "content": prompt_dict['instruction']},
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            return_tensors="pt",
            add_generation_prompt=True,
            # return_dict=True
        )

        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)  # if args.model != 'glm' else text.to('cuda')
        # else:
        # model_inputs = tokenizer.build_inputs_for_generation(text, allowed_special="all", return_tensors="pt",padding=True).to('cuda')
        generated_ids = self.model.generate(
            model_inputs.input_ids,
            max_new_tokens=self.max_tokens,
            pad_token_id= self.tokenizer.eos_token_id
        )

        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        # print(f'response:{response}')
        return response

    def GetResultFromGPT(self, prompt_dict):
        """
        GetResultFromGPT(self, prompt_dict: dict) -> str

        Sends a request to an OpenAI-compatible API server, specifically vLLM, using the provided prompt dictionary to generate a text completion based on a pre-configured language model.

        This method constructs a chat completion request with a system message and a user message extracted from the `prompt_dict`. It then sends the request to the API endpoint defined by `openai_api_base`, utilizing the API key `openai_api_key`. The response content from the API is returned as a string.

        Parameters:
            prompt_dict (dict): A dictionary containing two keys:
                - 'input': A string representing the system's input or context for the conversation.
                - 'instruction': A string representing the user's message or instruction for generating a response.

        Returns:
            str: The generated text content from the API's response.

        Note:
            - Ensure that `self.api_key` and `self.api_base` are set appropriately before calling this method.
            - The model name (`self.llm_name`), maximum number of tokens to generate (`self.max_tokens`), and temperature setting (`self.temperature`) are used to configure the generation parameters and must be predefined within the instance.

        """
        from openai import OpenAI
        # Set OpenAI's API key and API base to use vLLM's API server.
        openai_api_key = self.api_key
        openai_api_base = self.api_base
        client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )

        chat_response = client.chat.completions.create(
            model=self.llm_name,
            messages=[
                {"role": "system", "content": prompt_dict['input']},
                {"role": "user", "content": prompt_dict['instruction']},
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,

        )
        response = chat_response.choices[0].message.content
        # print(response)
        return response

    def batch_list(self, input_list, batch_size):
        """
        Yields batches of a list in specified size.

        This method takes an input list and divides it into smaller sublists (batches),
        each of which has a maximum length defined by the `batch_size` parameter. It yields
        these batches one at a time, allowing for efficient processing of large lists in chunks.

        Parameters
        ----------
        input_list : list
            The list to be divided into batches.
        batch_size : int
            The maximum size of each batch. Must be a positive integer.

        Yields
        ------
        list
            Sublists (batches) of the input list, each with up to `batch_size` elements.

        Raises
        ------
        ValueError
            If `batch_size` is not a positive integer.
        """
        for i in range(0, len(input_list), batch_size):
            yield input_list[i:i + batch_size]

    def get_response(self, data):
        """
        Get the response from the language model based on the input data.

        This method processes the input data and retrieves responses from the
        configured language model. It supports both single-input and batched
        processing modes, depending on the configuration of the object.

        In the case of a 'vllm' type model with batch processing enabled,
        the input data is divided into batches according to the specified batch size.
        Each batch is processed sequentially, with the results being collected
        and merged back with the original input data structure. For other models or
        when batch processing is not used, each input item is processed individually.

        Parameters:
            data (List[Dict]): A list of dictionaries, where each dictionary contains
                               the necessary input for the language model's `llm_func`.

        Returns:
            List[Dict]: A list of dictionaries, where each dictionary from the input `data`
                        is extended with a new key `'predict'` holding the respective response
                        from the language model.

        Note:
            The method currently prints progress bars using `tqdm` for visual feedback
            during batch processing. The commented out code at the end suggests a file
            saving operation which is not executed in this function's scope.
        """
        json_list = []
        print(f'-------- Get Model {self.llm_name} Response-------')

        if self.llm_type == 'vllm' and self.use_batch:
            batched_data = list(self.batch_list(data, self.config['batch_size']))
            all_result = []
            for batch in tqdm(batched_data):
                batched_result = self.llm_func(batch)
                all_result.extend(batched_result)
            for prompt_dict, r in zip(data, all_result):
                prompt_dict['predict'] = r
                json_list.append(prompt_dict)
        else:
            for prompt_dict in tqdm(data):
                response = self.llm_func(prompt_dict)
                # print(f'response:{response}')
                prompt_dict['predict'] = response
                json_list.append(prompt_dict)

        return json_list
