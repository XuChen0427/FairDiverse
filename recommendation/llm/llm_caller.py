from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


class LLM_caller(object):
    def __init__(self, llm, llm_path, device='cuda', api_key=None, api_base='EMPTY', max_tokens=256, temperature=0.8):
        self.llm = llm
        self.llm_path = llm_path
        self.device = device
        if self.llm == 'api':
            self.init_llm_api(api_key, api_base, max_tokens, temperature)
        else:
            self.init_llm_model()


    def init_llm_api(self, api_key, api_base, max_tokens, temperature):
        self.api_key = api_key
        self.api_base = api_base
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.llm_func = self.GetResultFromGPT


    def init_llm_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            self.llm_path,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_path, trust_remote_code=True)
        self.llm_func = self.GetResultFromHuggingFace

    def GetResultFromHuggingFace(self, prompt_dict):

        messages = [
            {"role": "system",
             "content": prompt_dict['input']},
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
            max_new_tokens=512,
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
            model=self.llm_path,
            messages=[
                {"role": "system", "content": prompt_dict['input']},
                {"role": "user", "content": prompt_dict['instruction']},
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,  # 温度

        )
        response = chat_response.choices[0].message.content
        print(response)
        return response

    def get_response(self, data):
        json_list = []
        print(f'--------Model {self.llm}-------')

        for prompt_dict in tqdm(data):
            response = self.llm_func(prompt_dict)

            # print(f'response:{response}')
            prompt_dict['predict'] = response
            json_list.append(prompt_dict)

        return json_list
        # outpath = args.output_file_path + f'{args.fairness_type}fair_{args.model}_response.json'
        # with open(outpath, 'w') as f:
        #     json.dump(json_list, f, indent=4)