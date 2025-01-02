from dataclasses import dataclass, field
import json
import logging
from typing import Optional, List
import os
from pathlib import Path
from openai import OpenAI

CLIENT = OpenAI(api_key=os.environ.get('OPENAI_API_KEY', False))

@dataclass
class OpenaiConfig:
  """
  Configuration for OpenAI engine
  """
  config_setting: Optional[str] = "gpt-4-32k"
  api_key: Optional[str] = os.environ.get('GPT_API_KEY', None)
  use_chat_completions: Optional[bool] = True
  max_parallelism: Optional[int] = 1
  max_context_length: Optional[int] = 8192
