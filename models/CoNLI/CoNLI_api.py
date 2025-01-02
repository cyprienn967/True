from flask import Flask, jsonify, request
import argparse
import json
import logging
import os
from pathlib import Path
import time
from tqdm import tqdm
from CoNLI.configs.nli_config import DetectionConfig
from CoNLI.configs.openai_config import OpenaiConfig
from CoNLI.configs.ta_config import TAConfig
from CoNLI.modules.data.data_loader import DataLoader
from CoNLI.modules.entity_detector import EntityDetectorFactory
from CoNLI.modules.sentence_selector import SentenceSelectorFactory
from CoNLI.modules.hallucination_detector import HallucinationDetector
from CoNLI.modules.hd_constants import AllHallucinations, FieldName
from CoNLI.modules.utils.conversion_utils import str2bool
from CoNLI.modules.utils.aoai_utils import AOAIUtil
from CoNLI.modules.data import hypothesis_preprocess_into_sentences

app = Flask(__name__)

model = AOAIUtil()

detection_config = DetectionConfig()
openai_config = OpenaiConfig()
ta_config = TAConfig()

sentence_selector = SentenceSelectorFactory.create_sentence_selector(detection_config.sentence_selector_type)
entity_detector = EntityDetectorFactory.create_entity_detector(detection_config.entity_detector_type, ta_args=ta_config)

detection_agent = HallucinationDetector(
  sentence_selector=sentence_selector,
  entity_detector=entity_detector,
  openai_config=openai_config,
  detection_config=detection_config,
  entity_detection_parallelism=1)

@app.route('/')
def home():
  return jsonify({"message": "Welcome to the CoNLI API!"})


@app.route('/conli', methods=['GET'])
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
  
  model_name = request_data["model"]
  if not model_name:
    return jsonify({"error": "model parameter is required"}), 400
  
  in_context = request_data.get("in_context", False)
  context = request_data.get("context", None)
  
  inference_temperature = request.args.get('inference_temperature', default=1, type=float)

  allHallucinations = []
  retval_jsonl = []
  
  response_raw = model.get_chat_completion(model=model_name, prompt=prompt, temperature=inference_temperature)
  response = response_raw.choices[0].message.content
  
  hypotheses = hypothesis_preprocess_into_sentences(response)
  
  hallucinations = detection_agent.detect_hallucinations(id, prompt, hypotheses)
  for h in hallucinations:
    allHallucinations.append(h)
  num_sentences = len(hypotheses)
  num_hallucinations = len(hallucinations)
  hallucination_rate = num_hallucinations / num_sentences if num_sentences > 0 else 0.0
  hallucinated = num_hallucinations > 0
  retval_jsonl.append(
    {
      AllHallucinations.DATA_ID: data_id,
      AllHallucinations.HALLUCINATED: hallucinated,
      AllHallucinations.HALLUCINATION_SCORE: hallucination_rate,
      AllHallucinations.HALLUCINATIONS: hallucinations,
      AllHallucinations.NUM_TOTAL_SENTENCES: num_sentences,
      AllHallucinations.NUM_TOTAL_HALLUCINATIONS: num_hallucinations,
    })
