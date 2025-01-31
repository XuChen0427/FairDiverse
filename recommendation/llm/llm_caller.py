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
        for i in range(0, len(input_list), batch_size):
            yield input_list[i:i + batch_size]

    def get_response(self, data):
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
        # outpath = args.output_file_path + f'{args.fairness_type}fair_{args.model}_response.json'
        # with open(outpath, 'w') as f:
        #     json.dump(json_list, f, indent=4)