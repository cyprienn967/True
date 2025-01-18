import datetime
import json
import re
import logging
import os
from openai import OpenAI
import time
from pathlib import Path

from .gpt_output_utils import certified_gpt_output_prefix
from ...configs.openai_config import OpenaiConfig

class KeyError(Exception):
    """OpenAIKey not provided in environment variable."""
    pass

class AOAIUtil:

    def __init__(self, key=None) -> None:
        self.auth_token = None
        self.default_credential = None
        self.default_model = "gpt-4o-mini"
        self.client = OpenAI(api_key=key)

    def get_model(self, model: str = None) -> str:
        if model:
            return model
        else:
            return self.default_model

    def get_completion(
            self,
            prompt: str,
            model: str = "gpt-4o-mini",
            temperature: float = 1.0,
            top_p: float = 0.0,
            max_completion_tokens: int = 100,
            frequency_penalty: float = 0.0,
            presence_penalty: float = 0.0,
            logprobs: bool = False,
            stop: list() = ["<|im_end|>"],
            n: int = 1):
        response = self.client.completions.create(
            model=model,
            messages=prompt,
            temperature=temperature,
            top_p=top_p,
            max_completion_tokens=max_completion_tokens,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            logprobs=logprobs,
            stop=stop,
            n=n
        )
        if not certified_gpt_output_prefix(response.choices[0].message.content):
            raise Exception('GPT undesired output due to rpm limit reached, resending current request')
        return response

    def get_chat_completion(
            self,
            prompt: str,
            model: str = "gpt-4o-mini",
            temperature: float = 1.0,
            top_p: float = 0.0,
            max_completion_tokens: int = 100,
            frequency_penalty: float = 0.0,
            presence_penalty: float = 0.0,
            logprobs: bool = False,
            stop: list() = ["<|im_end|>"],
            n: int = 1,
            ):
        if isinstance(prompt, str):
            messages = [
                {'role': 'user', 'content': prompt},
            ]
        else:
            messages = prompt
        
        response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_completion_tokens=max_completion_tokens,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stop=stop,
                n=n)
        # if not certified_gpt_output_prefix(response.choices[0].message.content):
        #     raise Exception(f'GPT undesired output due to rpm limit reached, resending current request. \n<GPT_OUTPUT>\n{response.choices[0].message.content}\n</GPT_OUTPUT>')
        return response

    def convert_to_chat_format(self, text: str) -> str:
        reg_str = "<\|im_start\|>(.*?)<\|im_end\|>"
        res = re.findall(reg_str, text, flags=re.DOTALL)
        chat = []
        for turn in res:
            role, content = turn.split("\n", 1)
            t = {"role": role, "content": content}
            chat.append(t)
        return chat
    
    @staticmethod
    def get_model_context_length(openai_config: OpenaiConfig) -> int:
        return openai_config.max_context_length

