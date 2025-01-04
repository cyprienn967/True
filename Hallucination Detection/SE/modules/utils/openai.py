import os
import hashlib
from tenacity import retry, wait_random_exponential, retry_if_not_exception_type

from openai import OpenAI
from .SE_config import SEConfig


CLIENT = OpenAI(api_key=os.environ.get('OPENAI_API_KEY', False))


class KeyError(Exception):
    """OpenAIKey not provided in environment variable."""
    pass


# @retry(retry=retry_if_not_exception_type(KeyError), wait=wait_random_exponential(min=10, max=20))
def predict(prompt, temperature=1.0, model='gpt-4o', logprobs=True):
    """Predict with GPT models."""

    if not CLIENT.api_key:
        raise KeyError('Need to provide OpenAI API key in environment variable `OPENAI_API_KEY`.')

    if isinstance(prompt, str):
        messages = [
            {'role': 'user', 'content': prompt},
        ]
    else:
        messages = prompt
        
    if model == 'gpt-4':
        model = 'gpt-4-0613'
    elif model == 'gpt-3.5':
        model = 'gpt-3.5-turbo-1106'

    output = CLIENT.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=500,
        logprobs=logprobs,
        top_logprobs=1,
        temperature=temperature,
    )
    response = output.choices[0].message.content
    
    if logprobs:
        logits = [token_info.logprob for token_info in output.choices[0].logprobs.content]
        return response, logits, None
    else:
        return response, None, None


# @retry(retry=retry_if_not_exception_type(KeyError), wait=wait_random_exponential(min=1, max=10))
# def sample_predict(prompt, num_samples, model='gpt-4', temperature=1.0, logprobs=0, config: SEConfig=None):
    

class GPTModel:
    def __init__(self, model_name='gpt-4'):
        self.model_name = model_name

    def predict(self, prompt, temperature=1.0):
        return predict(prompt, temperature, model=self.model_name)



def md5hash(string):
    return int(hashlib.md5(string.encode('utf-8')).hexdigest(), 16)
