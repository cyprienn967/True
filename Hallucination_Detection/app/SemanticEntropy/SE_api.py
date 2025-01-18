from flask import Flask, request, jsonify
import gc
import os
import logging
import random
from tqdm import tqdm

import numpy as np
import json
import torch

from modules.utils.SE_config import SEConfig
from .compute_entropy import compute_entropy
from modules.utils import utils
from Hallucination_Detection.utils import openai as oai
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


app = Flask(__name__)

# Initialize the model once
config = SEConfig()
model = utils.init_model(config)

@app.route('/')
def home():
  # response, entropies = oai.predict("hi!", temperature=1.0, model='gpt-4', logprobs=True)
  # data = {"response": response, "entropies": json.dump(entropies)}
  # return jsonify(data)
  return "Model initialized"

@app.route('/api/data', methods=['GET'])
def get_entropies():
  request_data = request.get_json()
  
  id = request_data["id"]
  if not id:
    return jsonify({"error": "id parameter is required"}), 400
  
  api_key = request_data["api_key"]
  if not api_key:
    return jsonify({"error": "api_key parameter is required"}), 400
  
  prompt = request_data["prompt"]
  if not prompt:
    return jsonify({"error": "prompt parameter is required"}), 400
  
  model_name = request_data["model"]
  if not model_name:
    return jsonify({"error": "model parameter is required"}), 400
  
  in_context = request_data.get("in_context", False)
  context = request_data.get("context", None)
  
  inference_temperature = request.args.get('inference_temperature', default=1, type=float)

  

  generations, results_dict = {}, {}

  # current_input = make_prompt(in_context, context, question, None)
  # local_prompt = prompt + current_input

  full_responses = []
  sampled_responses = []
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
    full_responses.append((predicted_answer, token_log_likelihoods, embedding))
    sampled_responses.append(predicted_answer)

  # Append all predictions for this example to `generations`.
  # generations['responses'] = full_responses
  
  entropies, semantic_ids = compute_entropy(config, prompt, full_responses, most_likely_answer_dict)

  # Return data
  entropy_data = {
    "output": most_likely_answer_dict['response'],
    "all responses": json.dumps(sampled_responses),
    "semantic ids": json.dumps(semantic_ids),
    "entropies": json.dumps(entropies)
  }
  return jsonify(entropy_data)

  # def pookie(aidan):
  #   if we get married: 
  #     i will be happy
  #   else:
  #     i will die and aidan is fake


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