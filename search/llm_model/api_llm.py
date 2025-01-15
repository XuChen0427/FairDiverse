import backoff
import json
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
import random
import time
import re


class OpenAIApiException(Exception):
    def __init__(self, msg, error_code):
        self.msg = msg
        self.error_code = error_code


class OpenAIApiProxy():
    def __init__(self, url, api_key=None):
        retry_strategy = Retry(
            total=1,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST"],
        )
        adapter = HTTPAdapter()
        self.session = requests.Session()
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        self.api_key = api_key
        self.url = url

    def __call__(self, params_gpt, headers={}):
        headers['Content-Type'] = headers['Content-Type'] if 'Content-Type' in headers else 'application/json'
        if self.api_key:
            headers['Authorization'] = "Bearer " + self.api_key
        try:
            response = self.session.post(self.url, headers=headers, data=json.dumps(params_gpt))
        except Exception as e:
            time.sleep(10)
            response = self.session.post(self.url, headers=headers, data=json.dumps(params_gpt))

        if response.status_code in (429, 404, 500, 502, 503, 504, 104):
            for idx in range(10000000000000000):
                interval = min(1, 10+idx*5)
                if idx % 10 == 0:
                    print("meet error 429 or 404, code: {}, msg: {},run {}".format(
                        response.status_code, response.text, interval, idx
                    ), flush=True)

                response = self.session.post(self.url, headers=headers, data=json.dumps(params_gpt))
                if response.status_code not in (429, 404, 500, 502, 503, 504, 104):
                    break
        if response.status_code != 200:
            err_msg = f"access openai error, status code: {response.status_code}, errmsg: {response.text}"
            raise OpenAIApiException(err_msg, response.status_code)
        data = json.loads(response.text)
        return data


class OpenAILMAgent():

    def __init__(self, config):
        self.model = config["model_name"]
        self.apikey = config["api_key"]
        self.temperature = config["temperature"]
        self.max_new_tokens = config["max_new_tokens"]
        self.top_p = config["top_p"]
        self._proxy = OpenAIApiProxy(url=config['api_url']+'/v1/chat/completions', api_key=self.apikey)

    def __call__(self, system_prompt, input_prompt, max_new_tokens):
        if isinstance(input_prompt, list):
            system_message, input_message = input_prompt[0].split("## Input Data")
        else:
            system_message, input_message = input_prompt.split("## Input Data")
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": "## Input Data"+input_message}
        ]

        prompt_dict = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_new_tokens,
            "top_p": self.top_p,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
        }
        response = self._proxy(prompt_dict)
        content = response["choices"][0]["message"]["content"]

        pattern = r'\{"rerank_list": "[^"]+"\}'

        match = re.search(pattern, content, re.DOTALL)
        if match:
            # Convert the string to a dictionary
            result_dict = match.group()
            return [result_dict] if isinstance(input_prompt, list) else result_dict
        else:
            return [content] if isinstance(input_prompt, list) else content
    
