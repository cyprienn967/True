import torch
import os

from modules.utils.SE_config import SEConfig
from modules.utils import utils
from True.Hallucination_Detection.utils import openai as oai



response, logprobs = oai.predict("what year is it?", temperature=1.0, model='gpt-4', logprobs=True)

print("hi!")

print(response)

print(logprobs)