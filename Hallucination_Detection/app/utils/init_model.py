from typing import List
from .openai import GPTModel
from ..SemanticEntropy.modules.pretrained_models.huggingface_models import HuggingfaceModel

def init_model(model_name: str, api_key: str = None):
  if 'llama' in model_name.lower() or 'falcon' in model_name or 'mistral' in model_name.lower():
    model = HuggingfaceModel(
      model_name, stop_sequences='default',
      max_new_tokens=500
    )
  elif 'gpt' in model_name.lower():
    model = GPTModel(api_key, model_name)
  else:
    raise ValueError(f'Unknown model_name `{model_name}`.')
  return model