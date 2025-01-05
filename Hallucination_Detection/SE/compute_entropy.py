"""Compute uncertainty measures after generating answers."""
from collections import defaultdict
import logging
import os
import pickle
import numpy as np
import wandb

from .modules.semantic_entropy import get_semantic_ids
from .modules.semantic_entropy import logsumexp_by_id
from .modules.semantic_entropy import predictive_entropy
from .modules.semantic_entropy import predictive_entropy_rao
from .modules.semantic_entropy import cluster_assignment_entropy
from .modules.semantic_entropy import context_entails_response
from .modules.semantic_entropy import EntailmentDeberta
from .modules.semantic_entropy import EntailmentGPT4
from .modules.semantic_entropy import EntailmentGPT35
from .modules.semantic_entropy import EntailmentGPT4Turbo
from .modules.semantic_entropy import EntailmentGPT4o
from .modules.semantic_entropy import EntailmentGPT4oMini
from .modules.semantic_entropy import EntailmentLlama
from .modules.utils import utils
from .modules.utils.SE_config import SEConfig


def compute_entropy(config: SEConfig, prompt, full_responses, most_likely_answer):
  entropies = defaultdict(list)
  result_dict = {}
  result_dict['semantic_ids'] = []
  
  if config.entailment_model == 'deberta':
    entailment_model = EntailmentDeberta()
  elif config.entailment_model == 'gpt-4':
    entailment_model = EntailmentGPT4()
  elif config.entailment_model == 'gpt-3.5':
    entailment_model = EntailmentGPT35()
  elif config.entailment_model == 'gpt-4-turbo':
    entailment_model = EntailmentGPT4Turbo()
  elif config.entailment_model == 'gpt-4o':
    entailment_model = EntailmentGPT4o()
  elif config.entailment_model == 'gpt-4o-mini':
    entailment_model = EntailmentGPT4oMini()
  elif 'llama' in config.entailment_model.lower():
    entailment_model = EntailmentLlama(config.entailment_model)
  else:
    raise ValueError
  
  if not config.use_all_generations:
    if config.use_num_generations == -1:
      raise ValueError
    responses = [fr[0] for fr in full_responses[:config.use_num_generations]]
    log_liks = [r[1] for r in full_responses[:config.use_num_generations]]
  else:
    responses = [fr[0] for fr in full_responses]
    log_liks = [r[1] for r in full_responses] 

  for i in log_liks:
    assert i

  if config.compute_context_entails_response:
    # Compute context entails answer baseline.
    entropies['context_entails_response'].append(context_entails_response(
      prompt, responses, entailment_model))

  # Compute semantic ids.
  semantic_ids = get_semantic_ids(responses,
                                  model=entailment_model,
                                  strict_entailment=config.strict_entailment,
                                  question=prompt)
  result_dict['semantic_ids'].append(semantic_ids)

  # Compute entropy from frequencies of cluster assignments.
  entropies['cluster_assignment_entropy'].append(cluster_assignment_entropy(semantic_ids))

  # Length normalization of generation probabilities.
  log_liks_agg = [np.mean(log_lik) for log_lik in log_liks]

  # Compute naive entropy.
  entropies['regular_entropy'].append(predictive_entropy(log_liks_agg))

  # Compute semantic entropy.
  log_likelihood_per_semantic_id = logsumexp_by_id(semantic_ids, log_liks_agg, agg='sum_normalized')
  pe = predictive_entropy_rao(log_likelihood_per_semantic_id)
  entropies['semantic_entropy'].append(pe)
  
  return entropies, semantic_ids