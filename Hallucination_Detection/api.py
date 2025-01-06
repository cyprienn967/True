from flask import Flask, jsonify, request
import argparse
import json
import logging
import os
from pathlib import Path
import time
from tqdm import tqdm
from .CoNLI.configs.nli_config import DetectionConfig
from .CoNLI.configs.openai_config import OpenaiConfig
from .CoNLI.configs.ta_config import TAConfig
from .CoNLI.modules.data.data_loader import DataLoader
from .CoNLI.modules.entity_detector import EntityDetectorFactory
from .CoNLI.modules.sentence_selector import SentenceSelectorFactory
from .CoNLI.modules.hallucination_detector import HallucinationDetector
from .CoNLI.modules.hd_constants import AllHallucinations, FieldName
from .CoNLI.modules.utils.conversion_utils import str2bool
from .CoNLI.modules.utils.aoai_utils import AOAIUtil
from .CoNLI.modules.data.response_preprocess import hypothesis_preprocess_into_sentences
from .SemanticEntropy.modules.utils.SE_config import SEConfig
from .SemanticEntropy.compute_entropy import compute_entropy
from .utils.init_model import init_model
from .utils.process_tokens import process_tokens
from .ConfidenceFilter.extract_keywords import extract_keywords


app = Flask(__name__)

detection_config = DetectionConfig()
openai_config = OpenaiConfig()
ta_config = TAConfig()

sentence_selector = SentenceSelectorFactory.create_sentence_selector(detection_config.sentence_selector_type)
entity_detector = EntityDetectorFactory.create_entity_detector(detection_config.entity_detector_type, ta_config=ta_config)

detection_agent = HallucinationDetector(
  sentence_selector=sentence_selector,
  entity_detector=entity_detector,
  openai_config=openai_config,
  detection_config=detection_config,
  entity_detection_parallelism=1)


se_config = SEConfig()

@app.route('/')
def home():
  return jsonify({"message": "Welcome to the Hallucination Detection API!"})


# initializes the model that the client wants to query and detect hallucinations on
@app.route('/model', methods=['POST'])
def initialize_model():
  request_data = request.get_json()
  
  model_name = request_data["model"]
  if not model_name:
    return jsonify({"error": "model parameter is required"}), 400
  
  try:
    model_instance = init_model(model_name)
    app.config['MODEL'] = model_instance
  except Exception as e:
    return jsonify({"error": str(e)}), 500
  
  return jsonify({"message": "Model initialized successfully!"})


@app.route('/detect', methods=['GET'])
def run_hallucination_detection():
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
  
  in_context = request_data.get("in_context", False)
  if isinstance(in_context, str):
    in_context = in_context.lower() == 'true'
  context = request_data.get("context", None)
  inference_temperature = request_data.get("inference_temperature", 1.0)
  
  # Check if model is initialized
  model_instance = app.config.get('MODEL')
  if model_instance is None:
    return jsonify({"error": "Model not initialized! Please initialize a model through API endpoint /model"}), 500
  
  response, token_log_likelihoods, tokens_raw = model_instance.predict(prompt, temperature, max_completion_tokens=250)
  
  tokens, probs = process_tokens(token_log_likelihoods, tokens_raw)
  
  hypotheses = hypothesis_preprocess_into_sentences(response)
  # sentences = {i: hypothesis for i, hypothesis in enumerate(hypotheses)}
  keyword_dict = extract_keywords(response)
  
  filtered_hypotheses = [] # if keywords have low generated likelihood, they get flagged
  
  for i in range(len(hypotheses)):
    hypothesis = hypotheses[i]
    keywords = keyword_dict.get(i, [])
    
    pointer1 = 0
    pointer2 = 0
    while pointer1 < len(hypothesis) and pointer2 < len(keywords):
      if hypothesis[pointer1] == keywords[pointer2]:
        if probs[i][pointer1] < 0.8:
          filtered_hypotheses.append(hypothesis)
        pointer1 += 1
        pointer2 += 1
      else:
        pointer1 += 1

  
  
  full_responses = []
  sampled_responses = []
  num_generations = 1 + se_config.num_generations

  # Generate responses to calculate semantic entropy
  for i in range(num_generations):
    # Temperature for first generation is always `0.1`.
    temperature = 0.1 if i == 0 else inference_temperature

    predicted_answer, token_log_likelihoods, tokens = model_instance.predict(prompt, 
                                                                                temperature, 
                                                                                max_completion_tokens=se_config.max_completion_tokens)

    if i == 0:
      most_likely_answer_dict = {
        'response': predicted_answer,
        'token_log_likelihoods': token_log_likelihoods,
        'tokens': tokens,}
    full_responses.append((predicted_answer, token_log_likelihoods, tokens))
    sampled_responses.append(predicted_answer)

  # Append all predictions for this example to `generations`.
  # generations['responses'] = full_responses
  
  entropies, semantic_ids = compute_entropy(se_config, prompt, full_responses, most_likely_answer_dict)

  # Return data
  entropy_data = {
    "output": most_likely_answer_dict['response'],
    "all responses": json.dumps(sampled_responses),
    "semantic ids": json.dumps(semantic_ids),
    "entropies": json.dumps(entropies)
  }
  
  # Detect hallucinations with CoNLI
  if (in_context):
    full_prompt = context + prompt
    allHallucinations = []
    retval_jsonl = []
    
    # response_raw = model_instance.get_chat_completion(model=model_name, prompt=full_prompt, temperature=inference_temperature)
    # response = response_raw.choices[0].message.content
    response = most_likely_answer_dict['response']
    
    hallucinations = detection_agent.detect_hallucinations(full_prompt, hypotheses)
    for h in hallucinations:
      allHallucinations.append(h)
    num_sentences = len(hypotheses)
    num_hallucinations = len(hallucinations)
    hallucination_rate = num_hallucinations / num_sentences if num_sentences > 0 else 0.0
    hallucinated = num_hallucinations > 0
    retval_jsonl.append(
      {
        AllHallucinations.HALLUCINATED: hallucinated,
        AllHallucinations.HALLUCINATION_SCORE: hallucination_rate,
        AllHallucinations.HALLUCINATIONS: hallucinations,
        AllHallucinations.NUM_TOTAL_SENTENCES: num_sentences,
        AllHallucinations.NUM_TOTAL_HALLUCINATIONS: num_hallucinations,
      })
  else:
    retval_jsonl = "CoNLI not available. Please provide context to detect hallucinations."
  
  return jsonify({"hallucination_data": retval_jsonl, "entropy_data": entropy_data})
