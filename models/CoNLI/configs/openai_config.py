from dataclasses import dataclass, field
import json
import logging
from typing import Optional, List
import os
from pathlib import Path
from openai import OpenAI

CLIENT = OpenAI(api_key=os.environ.get('OPENAI_API_KEY', False))

@dataclass
class OpenaiArguments:
  """
  Configuration for OpenAI engine
  """
  config_setting: Optional[str] = "gpt-4-32k"
  api_key: Optional[str] = os.environ.get('GPT_API_KEY', None)
  use_chat_completions: Optional[bool] = True
  max_parallelism: Optional[int] = 2
  max_context_length: Optional[int] = 8192

def create_openai_arguments(config_setting_key: str, max_parallelism: int, config_file: str = None) -> OpenaiArguments:
  # if config_file is None:
  #   # use default config file. This seems a bit strange - assume some file outside of current package folder
  #   config_file = (Path(__file__).absolute()).parent.parent / 'configs' / 'aoai_config.json'
  # with open(config_file, "r") as config_file:
  #   config = json.load(config_file)
  
  # if config_setting_key is None:
  #   logging.warning(f"AOAI config setting key is None, using default config setting key")
  #   if len(config) > 1:
  #     raise ValueError(f"AOAI config setting key is None, but config file has more than 1 setting. Please specify the config setting key")
  #   config_setting_key = list(config.keys())[0]

  # if config_setting_key not in config:
  #   raise ValueError(f"AOAI config setting {config_setting_key} not found in {config_file}")

  openai_args = OpenaiArguments()
  openai_args.max_parallelism = max_parallelism
  return openai_args