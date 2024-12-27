from flask import Flask, request, jsonify
import gc
import os
import logging
import random
from tqdm import tqdm

import numpy as np
import torch
import wandb

from SE_config import SEConfig
from uncertainty.uncertainty_measures.semantic_entropy import get_semantic_ids
from uncertainty.uncertainty_measures.semantic_entropy import logsumexp_by_id
from uncertainty.uncertainty_measures.semantic_entropy import predictive_entropy
from uncertainty.uncertainty_measures.semantic_entropy import predictive_entropy_rao
from uncertainty.uncertainty_measures.semantic_entropy import cluster_assignment_entropy
from uncertainty.uncertainty_measures.semantic_entropy import context_entails_response
from uncertainty.uncertainty_measures.semantic_entropy import EntailmentDeberta
from uncertainty.uncertainty_measures.semantic_entropy import EntailmentGPT4
from uncertainty.uncertainty_measures.semantic_entropy import EntailmentGPT35
from uncertainty.uncertainty_measures.semantic_entropy import EntailmentGPT4Turbo
from uncertainty.uncertainty_measures.semantic_entropy import EntailmentLlama
from uncertainty.utils import utils
from collections import defaultdict

# edit this
def make_prompt(use_context, context, question, answer):
  prompt = ''
  if use_context and (context is not None):
      prompt += f"Context: {context}\n"
  prompt += f"Question: {question}\n"
  if answer:
      prompt += f"Answer: {answer}\n\n"
  else:
      prompt += 'Answer:'
  return prompt


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
  
  return entropies, result_dict


app = Flask(__name__)

@app.route('/')
def home():
  return "Semantic Entropy Checker API"

@app.route('/api/data', methods=['GET'])
def get_data():
  id = request.args.get('id')
  if not id:
    return jsonify({"error": "ID parameter is required"}), 400
  api_key = request.args.get('api_key')
  if not api_key:
    return jsonify({"error": "API Key parameter is required"}), 400
  prompt = request.args.get('prompt')
  if not prompt:
    return jsonify({"error": "Prompt parameter is required"}), 400
  in_context = request.args.get('in_context', default=False, type=bool)
  inference_temperature = request.args.get('inference_temperature', default=1, type=float)
  
  # Fetch data from API
  
  config = SEConfig()
  
  model = utils.init_model(config.model)
  generations, results_dict = {}, {}

  # current_input = make_prompt(in_context, context, question, None)
  # local_prompt = prompt + current_input

  full_responses = []
  num_generations = 1 + config.num_generations

  for i in range(num_generations):
    # Temperature for first generation is always `0.1`.
    temperature = 0.1 if i == 0 else inference_temperature

    predicted_answer, token_log_likelihoods, embedding = model.predict(prompt, temperature)
    embedding = embedding.cpu() if embedding is not None else None

    if i == 0:
      most_likely_answer_dict = {
        'response': predicted_answer,
        'token_log_likelihoods': token_log_likelihoods,
        'embedding': embedding,}
      generations[example['id']].update({
        'most_likely_answer': most_likely_answer_dict,
        'reference': utils.get_reference(example)})
    else:
      # Aggregate predictions over num_generations.
      full_responses.append(
        (predicted_answer, token_log_likelihoods, embedding))

  # Append all predictions for this example to `generations`.
  generations[example['id']]['responses'] = full_responses

  sample_data = {
    "id": id,
    "name": name,
    "description": description
  }
  return jsonify(sample_data)

# @app.route('/api/data', methods=['POST'])
# def post_data():
#   data = request.get_json()
#   response = {
#     "message": "Data received",
#     "data": data
#   }
#   return jsonify(response), 201

if __name__ == '__main__':
  app.run(debug=True)