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
from uncertainty.data.data_utils import load_ds
from uncertainty.utils import utils
from compute_uncertainty_measures import main as main_compute

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


def compute_entropy():
  if not args.use_all_generations:
    log_liks = [r[1] for r in full_responses[:args.use_num_generations]]
  else:
    log_liks = [r[1] for r in full_responses]

  for i in log_liks:
    assert i

  if args.compute_context_entails_response:
    # Compute context entails answer baseline.
    entropies['context_entails_response'].append(context_entails_response(
      context, responses, entailment_model))

  if args.condition_on_question and args.entailment_model == 'deberta':
    responses = [f'{question} {r}' for r in responses]

  # Compute semantic ids.
  semantic_ids = get_semantic_ids(
    responses, model=entailment_model,
    strict_entailment=args.strict_entailment, example=example)

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
  accuracies, generations, results_dict = [], {}, {}

  # current_input = make_prompt(in_context, context, question, None)
  # local_prompt = prompt + current_input

  full_responses = []

  # We sample one low temperature answer on which we will compute the
  # accuracy and args.num_generation high temperature answers which will
  # be used to estimate the entropy variants.

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