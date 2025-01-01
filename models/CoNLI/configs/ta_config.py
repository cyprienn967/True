from dataclasses import dataclass, field
import json
import logging
from typing import Optional, List
import os
from pathlib import Path

@dataclass
class TAArguments:
  """
  configuration for Text Analytics
  """
  config_file: Optional[str] = field(
    metadata={"help": "The pre-defined TA config setting"}
  )
  endpoint: Optional[str] = field(
    metadata={"help": "The endpoint to call TA"}
  )
  config_setting: Optional[str] = field(
    default='ta-health', metadata={"help": "The pre-defined TA config setting"}
  )
  api_key: Optional[str] = field(
    default=os.environ.get('LANGUAGE_KEY', None), metadata={"help": "API key to call openai gpt"}
  )
  entities: Optional[List[str]] = field(
    default_factory=list, metadata={"help": "the selected entities to be detected"}
  )

def create_ta_arguments(config_key: str, ta_config_file: str = None):
  if ta_config_file is None:
    # use default config file. This seems a bit strange - assume some file outside of current package folder
    ta_config_file = (Path(__file__).absolute()).parent.parent / 'configs' / 'ta_config.json'

  with open(ta_config_file, "r") as config_file:
    config = json.load(config_file)

  if config_key is None:
    logging.warning(f"TA config setting key is None, using default config setting key")
    if len(config) > 1:
      raise ValueError(f"TA config setting key is None, but config file has more than 1 setting. Please specify the config setting key")
    config_key = list(config.keys())[0]

  if config_key not in config:
    raise ValueError(f"TA config setting {config_key} not found in {config_file}")

  settings = config[config_key]

  ta_args = TAArguments(ta_config_file, settings['ENDPOINT'])
  ta_args.config_setting = config_key
  ta_args.api_key = settings['API_KEY']
  if 'ENTITIES' in settings:
    ta_args.entities = settings['ENTITIES']
  else:
    ta_args.entities = None  # leave this to None so we can inject default entities list later

  return ta_args
