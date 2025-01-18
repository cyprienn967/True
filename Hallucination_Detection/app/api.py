from flask import Flask, jsonify, request, make_response
import argparse
import json
import logging
import os
from pathlib import Path
import time
from tqdm import tqdm
import nltk
from app.CoNLI.configs.nli_config import DetectionConfig
from app.CoNLI.configs.openai_config import OpenaiConfig
from app.CoNLI.configs.ta_config import TAConfig
from app.CoNLI.modules.data.data_loader import DataLoader
from app.CoNLI.modules.entity_detector import EntityDetectorFactory
from app.CoNLI.modules.sentence_selector import SentenceSelectorFactory
from app.CoNLI.modules.hallucination_detector import HallucinationDetector
from app.CoNLI.modules.hd_constants import AllHallucinations, FieldName
from app.CoNLI.modules.utils.conversion_utils import str2bool
from app.CoNLI.modules.utils.aoai_utils import AOAIUtil
from app.CoNLI.modules.data.response_preprocess import hypothesis_preprocess_into_sentences
from app.SemanticEntropy.modules.utils.SE_config import SEConfig
from app.SemanticEntropy.compute_entropy import compute_entropy
from app.utils.init_model import init_model
from app.utils.inference_config import InferenceConfig
from app.ConfidenceFilter.process_tokens import process_tokens, filter_hypotheses
from app.ConfidenceFilter.create_validation import get_topic, generate_validation_prompt
from app.ConfidenceFilter.extract_keywords import extract_keywords
from app.SelfContradiction.self_contradiction import check_contradiction
from concurrent.futures import ThreadPoolExecutor
import awsgi

# logging.basicConfig(filename="newfile.log", filemode='w')
# logger = logging.getLogger()

app = Flask(__name__)

nltk.download('punkt_tab')


openai_config = OpenaiConfig()
ta_config = TAConfig()
se_config = SEConfig()

# detection_config = DetectionConfig()
# sentence_selector = SentenceSelectorFactory.create_sentence_selector(detection_config.sentence_selector_type)
# entity_detector = EntityDetectorFactory.create_entity_detector(detection_config.entity_detector_type, ta_config=ta_config)
# detection_agent = HallucinationDetector(
#   sentence_selector=sentence_selector,
#   entity_detector=entity_detector,
#   openai_config=openai_config,
#   detection_config=detection_config,
#   entity_detection_parallelism=1)


inference_config = InferenceConfig()

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
  api_key = request_data["api_key"]
  if api_key:
    os.environ['OPENAI_API_KEY'] = api_key
    app.config['API_KEY'] = api_key
  
  try:
    model_instance = init_model(model_name, api_key)
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
  api_key = app.config.get('API_KEY')
  if model_instance is None:
    return jsonify({"error": "Model not initialized! Please initialize a model through API endpoint /model"}), 500
  
  full_prompt = context + prompt if in_context else prompt
  response, token_log_likelihoods, tokens_raw = model_instance.predict(full_prompt, inference_temperature, max_completion_tokens=250)
  
  tokens, probs = process_tokens(token_log_likelihoods, tokens_raw)
  print(response)
  
  hypotheses = hypothesis_preprocess_into_sentences(response)
  num_hypotheses = len(hypotheses)
  # sentences = {i: hypothesis for i, hypothesis in enumerate(hypotheses)}
  keyword_list = extract_keywords(inference_config.keyword_model, api_key, response)
  print(keyword_list)
  # for i in range(len(probs)):
  #   print(f"{tokens[i]}: {probs[i]}")
  
  hypothesis_evaluations, hallucinated_keywords = filter_hypotheses(hypotheses, keyword_list, probs)
  print(hallucinated_keywords)
  
  # flags hypotheses that contain low likelihood keywords
  labeled_hypotheses = {i: {"hypothesis": hypotheses[i]['text'], "is_hallucinated": hypothesis_evaluations[i], "hallucinated_keywords": hallucinated_keywords[i]} for i in range(num_hypotheses)}

  # return jsonify({"hallucinations": hallucinated_keywords, "response": response, "hypotheses": hypotheses, "filtered_hypotheses": filtered_hypotheses, "keywords": keyword_dict, "probs": probs})
  is_hallucinated_se = [False for i in range(num_hypotheses)]
  hallucinated_keywords_se = {i: [] for i in range(num_hypotheses)}
  
  topic = get_topic(prompt, api_key)
  print(f"Topic: {topic}")
  
  with ThreadPoolExecutor(max_workers=6) as executor:
    def process_hypothesis(i):
      local_labeled = labeled_hypotheses[i]
      local_hallucinated_se = is_hallucinated_se[i]
      local_hallucinated_keywords_se = hallucinated_keywords_se[i]

      if not hypothesis_evaluations[i]:
        local_labeled["semantic_entropy"] = None
        return (i, local_labeled, local_hallucinated_se, local_hallucinated_keywords_se)

      local_labeled["semantic_entropy"] = []
      local_labeled["validation_prompts"] = []
      hypothesis = hypotheses[i]["text"]

      if i == 0:
        hypothesis_context = ""
      else:
        hypothesis_context = hypotheses[i - 1]["text"]

      for hallucination in hallucinated_keywords[i]:
        validation_prompt = generate_validation_prompt(inference_config.validation_model,
                                 api_key,
                                 hypothesis,
                                 hypothesis_context,
                                 topic,
                                 hallucination[0])
        full_responses = []
        sampled_responses = []

        for j in range(se_config.num_generations):
          predicted_answer, sample_logits, sample_tokens = model_instance.predict(
            validation_prompt, inference_temperature, max_completion_tokens=se_config.max_completion_tokens
          )
          full_responses.append((predicted_answer, sample_logits, sample_tokens))
          sampled_responses.append(predicted_answer)
          print(predicted_answer)

        entropies, semantic_ids = compute_entropy(se_config, api_key, validation_prompt, full_responses)
        entropy_data = {
          "all responses": json.dumps(sampled_responses),
          "semantic ids": json.dumps(semantic_ids),
          "entropies": json.dumps(entropies)
        }

        full_responses.append((response, token_log_likelihoods, tokens))
        local_labeled["semantic_entropy"].append(entropies["semantic_entropy"][0])
        local_labeled["validation_prompts"].append(validation_prompt)

        if (entropies["semantic_entropy"][0] > se_config.entropy_threshold) or \
           (entropies["semantic_entropy"][0] < se_config.contradiction_threshold and
          check_contradiction(inference_config.contradiction_model,
                    api_key,
                    hypothesis,
                    sampled_responses,
                    se_config.contradiction_num_samples)):
          local_hallucinated_se = True
          local_hallucinated_keywords_se.append(hallucination)

      return (i, local_labeled, local_hallucinated_se, local_hallucinated_keywords_se)

    futures = [executor.submit(process_hypothesis, i) for i in range(num_hypotheses)]
    for f in futures:
      i, updated_labeled, updated_hallucinated_se, updated_hallucinated_keywords_se = f.result()
      labeled_hypotheses[i] = updated_labeled
      is_hallucinated_se[i] = updated_hallucinated_se
      hallucinated_keywords_se[i] = updated_hallucinated_keywords_se
  
      
  return jsonify({"response": response, "hallucinations": hallucinated_keywords_se, "is_hallucinated_se": is_hallucinated_se, "hallucination_data": labeled_hypotheses})
    
  # Detect hallucinations with CoNLI
  if (in_context):
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

@app.errorhandler(404)
def resource_not_found(e):
  return make_response(jsonify(error='Not found!'), 404)


def handler(event, context):
  return awsgi.response(app, event, context)