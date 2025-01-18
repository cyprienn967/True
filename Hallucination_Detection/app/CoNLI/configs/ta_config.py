from dataclasses import dataclass, field
import json
import logging
from typing import Optional, List
import os
from pathlib import Path

class TAConfig:
  """
  configuration for Text Analytics
  """
  def __init__(self, config_file: Optional[str] = "", endpoint: Optional[str] = "https://true.cognitiveservices.azure.com/", config_setting: Optional[str] = "ta-general", api_key: Optional[str] = None, entities: Optional[List[str]] = None):
    self.config_file = config_file
    self.endpoint = endpoint
    self.config_setting = config_setting
    self.api_key = api_key if api_key is not None else "DB6j8qcDRzbMce2L6jRcoGQqALCg2BkDNxNf30XMyzEo6DxEhOmvJQQJ99BAACYeBjFXJ3w3AAAaACOGtddd" # os.environ.get('LANGUAGE_KEY', None)
    self.entities = entities if entities is not None else ["Person",
        "PersonType",
        "Location",
        "Event",
        "Skill",
        "DateTime_DateRange",
        "DateTime_Duration",
        "Quantity_Number",
        "Quantity_Currency"]

# def create_ta_arguments(config_key: str, ta_config_file: str = None):
#   if ta_config_file is None:
#     # use default config file. This seems a bit strange - assume some file outside of current package folder
#     ta_config_file = (Path(__file__).absolute()).parent.parent / 'configs' / 'ta_config.json'

#   with open(ta_config_file, "r") as config_file:
#     config = json.load(config_file)

#   if config_key is None:
#     if len(config) > 1:
#       raise ValueError(f"TA config setting key is None, but config file has more than 1 setting. Please specify the config setting key")
#     config_key = list(config.keys())[0]

#   if config_key not in config:
#     raise ValueError(f"TA config setting {config_key} not found in {config_file}")

#   settings = config[config_key]

#   ta_args = TAConfig(ta_config_file, settings['ENDPOINT'])
#   ta_args.config_setting = config_key
#   ta_args.api_key = settings['API_KEY']
#   if 'ENTITIES' in settings:
#     ta_args.entities = settings['ENTITIES']
#   else:
#     ta_args.entities = None  # leave this to None so we can inject default entities list later

#   return ta_args
