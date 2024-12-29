import torch
import os

from uncertainty.utils.SE_config import SEConfig
from uncertainty.utils import utils
from uncertainty.utils import openai as oai



response, logprobs = oai.predict("what year is it?", temperature=1.0, model='gpt-4', logprobs=True)

print("hi!")

print(response)

print(logprobs)