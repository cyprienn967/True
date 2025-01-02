
import logging
import math
import pandas as pd
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List
from pathlib import Path

import CoNLI.modules.utils.gpt_output_utils as gpt_output_utils
from CoNLI.configs.nli_config import DetectionConfig
from CoNLI.configs.openai_config import Openai_Config
from CoNLI.modules.entity_detector import EntityDetectorBase, GenTAEntityDetector
from CoNLI.modules.hallucination_detection_prompt import hallucination_detection_prompt
from CoNLI.modules.hd_constants import FieldName
from CoNLI.modules.sentence_selector import SentenceSelectorBase
from CoNLI.modules.utils.sentence_splitter import SentenceSplitter
from CoNLI.modules.utils.aoai_utils import AOAIUtil

def count_tokens(text : str) -> int:
    import re
    return len(re.findall(r'\w+', text))

class HallucinationDetector:
    # Dependency injection the entity_detector
    def __init__(self,
                 sentence_selector : SentenceSelectorBase,
                 entity_detector : EntityDetectorBase,
                 openai_config : Openai_Config = Openai_Config(),
                 detection_config : DetectionConfig = DetectionConfig(),
                 entity_detection_parallelism: int = 1,
                 entity_detection_batch: int = 25,
                 ) -> None:
        self._entity_detector = entity_detector
        self._sentence_selector = sentence_selector
        if not (self._entity_detector or self._sentence_selector):
            raise Exception("Both sentence-level and entity-level hd are disabled. Please change config to enable at least one.")
        
        self._sentence_splitter = SentenceSplitter()
        
        self._openai_config = openai_config
        self._detection_config = detection_config
        self._prompt_util = hallucination_detection_prompt(use_chat_completions = openai_config.use_chat_completions,
                                                           max_prompt_tokens = openai_config.max_context_length)
        self._entity_detection_batch = entity_detection_batch
        self._entity_detection_parallelism = entity_detection_parallelism
        

    def _add_entities_to_sentences(self, sentences : List[Dict]) -> List[Dict]:
        if isinstance(self._entity_detector, GenTAEntityDetector):
            batch_len = min(self._entity_detection_batch, 5)
        elif isinstance(self._entity_detector):
            batch_len = min(self._entity_detection_batch, 25)
        else:
            batch_len = self._entity_detection_batch

        sentences_df = pd.DataFrame(sentences)
        sentences_text = sentences_df[FieldName.SENTENCE_TEXT].tolist()
        sentence_batches = [sentences_text[x:x+batch_len] for x in range(0, len(sentences_text), batch_len)]
        max_workers = min(max(self._entity_detection_parallelism, 1), len(sentence_batches))
        hd_entities = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for batch in executor.map(self._entity_detector.detect_entities, sentence_batches):
                hd_entities += batch
        n_entities = sum(len(x) for x in hd_entities)
        sentences_df[FieldName.HD_ENTITY] = hd_entities
        sentences = sentences_df.to_dict('records')
        return sentences, n_entities
    
    def detect_hallucinations(self, source : str, sentences : List[Dict]) -> List[Dict]:
        perf_counters = {}
        perf_counters["n_gpt_requests"] = 0
        perf_counters["n_gpt_calls"] = 0
        perf_counters["n_source_tokens"] = count_tokens(source)
        hd_result = []
        if self._sentence_selector:
            perf_counters["n_sentences"] = len(sentences)
            # step # 3.1 select sentences send for HD
            n_content_tokens = 0
            for s in sentences:
                is_selected, hd_sentence = self._sentence_selector.select_sentence(s[FieldName.SENTENCE_TEXT])
                s[FieldName.HD_ENTITY] = set([hd_sentence]) if is_selected else set([])
                n_content_tokens += count_tokens(s[FieldName.SENTENCE_TEXT])

            perf_counters["n_content_tokens"] = n_content_tokens
            # step #3.2: do hallucination detection with extra information
            hd_result = self.do_hallucation_detection(source, sentences, perf_counters=perf_counters, sentence_level_hd=True)

            # skip hallucination sentences for 2nd round on entity-level hd
            is_hallucination_sentence_ids = set([x[FieldName.SENTENCE_ID] for x in hd_result])
            sentences = [s for s in sentences if s[FieldName.SENTENCE_ID] not in is_hallucination_sentence_ids]

        # Add hd_result into sentences
        if self._entity_detector and len(sentences) > 0:
            # step #2.1: detect entities in current sentence send for HD
            sentences, perf_counters["n_entities"] = self._add_entities_to_sentences(sentences)
            # step #2.2: do hallucination detection with extra information
            hd_result += self.do_hallucation_detection(source, sentences, perf_counters=perf_counters, sentence_level_hd=False)
        else:
            perf_counters["n_entities"] = None

        hd_result = sorted(
            hd_result,
            key=lambda d: (
                d[FieldName.SENTENCE_ID],
                d[FieldName.DETECTION_TYPE],
                d[FieldName.SENTENCE_TEXT]))
        return hd_result
    
    def do_hallucation_detection(self, 
                                 source : str,
                                 sentences : List[Dict],
                                 sentence_level_hd : bool,
                                 perf_counters: dict) -> List[Dict]:
        batch_size = self._detection_config.batch_size
        max_parallelism = self._openai_config.max_parallelism

        items, results = [], []
        for sentence in sentences:
            sentence_id = sentence[FieldName.SENTENCE_ID]
            sentence_text = sentence[FieldName.SENTENCE_TEXT].strip()
                
            # sending one request per entity span
            for hdEntity in sentence[FieldName.HD_ENTITY]:
                request = {
                    'Hypothesis': hdEntity.hypothesis,
                    'DetectedEntityType': hdEntity.entity_type if not sentence_level_hd else '',
                    'DetectionType': hdEntity.detection_type,
                    'DetectedEntityCleaned': '',
                    'SentenceId': sentence_id,
                    'Sentence': sentence_text,
                    }
                items.append(request)
        count = len(items)
        perf_counters["n_gpt_requests"] += count
        npayloads = math.ceil(count / batch_size)
        gpt_request_payloads = [
            self.create_payload(
                items = items[i * batch_size: min((i + 1) * batch_size, count)],
                src = source,
                promptUtil = self._prompt_util,
            )
            for i in range(npayloads)
        ]
        perf_counters["n_gpt_calls"] += len(gpt_request_payloads)
        if len(gpt_request_payloads) > 0:
            gpt_results_raw = list()
            max_workers = min(max(max_parallelism, 1), len(gpt_request_payloads))
            with tqdm(total=len(gpt_request_payloads), leave=False) as pbar2:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {
                        executor.submit(
                            self.process_payload_by_GPT,
                            payload,
                            self.aoaiUtil,
                            self._openai_config,
                            self._detection_config): payload
                        for payload in gpt_request_payloads
                    }
                    
                    for future in as_completed(futures):
                        gpt_results_raw.append(future.result())
                        pbar2.update(1)
            results += HallucinationDetector.parse_gpt_results(gpt_results_raw)
            
        return results
    
    @staticmethod
    def create_payload(items, src, promptUtil : hallucination_detection_prompt) -> Dict:
        prompt_to_send_to_gpt = promptUtil.create_batch_prompt(src, items, 4096)  # need to add this and the prompt
        return {'prompt': prompt_to_send_to_gpt, 'items': items}

    # send payload to GPT endpoint and get back the results
    @staticmethod
    def process_payload_by_GPT(payload, aoaiUtil : AOAIUtil, openai_args : Openai_Config, detection_config : DetectionConfig) -> Dict:

        outputs = []
        try:
            if openai_args.use_chat_completions:
                gpt_response = aoaiUtil.get_chat_completion(
                    messages = payload['prompt'],
                    temperature = detection_config.temperature,
                    top_p = detection_config.top_p, 
                    max_tokens = detection_config.max_tokens,
                    frequency_penalty = detection_config.freq_penalty,
                    presence_penalty = detection_config.presence_penalty,
                    n=detection_config.n)
                choices = gpt_response['choices']
                for choice in choices:
                    outputs.append(choice.message.content)
                payload['gpt_raw_output'] = outputs
            else:
                gpt_response = aoaiUtil.get_completion(
                    prompt = payload['prompt'],
                    max_tokens = detection_config.max_tokens,
                    temperature = detection_config.temperature,
                    top_p = detection_config.top_p,
                    frequency_penalty = detection_config.freq_penalty,
                    presence_penalty = detection_config.presence_penalty,
                    logprobs = detection_config.log_prob,
                    n=detection_config.n)
                choices = gpt_response['choices']
                for choice in choices:
                    outputs.append(choice.text)
                payload['gpt_raw_output'] = outputs
        except Exception as exc:
            logging.warning(f"Failed to call GPT: output format wrong!")
            logging.warning(f'Exception: {exc}')
            payload['gpt_raw_output'] = [ 'the format of gpt output is wrong' ]

        return payload

    @staticmethod
    def parse_gpt_results_single(gpt_result_raw) -> list:
        gpt_result_cooked = []
        # extraction depends on the prompts
        for generation in range(len(gpt_result_raw["gpt_raw_output"])):
            ans = gpt_output_utils.parse_gpt_batch(gpt_result_raw["gpt_raw_output"][generation], len(gpt_result_raw["items"]))  # this is the the result for each item
            for i, item in enumerate(gpt_result_raw["items"]):
                if ans[i]['IsHallucination']:
                    # At this point we think we've found a hallucination
                    gpt_result_cooked.append({
                        FieldName.SENTENCE_ID: item['SentenceId'],
                        FieldName.DETECTION_TYPE: item["DetectionType"],
                        FieldName.SENTENCE_TEXT: item['Hypothesis'],
                        FieldName.NAME: item['DetectedEntityCleaned'],
                        FieldName.TYPE: item['DetectedEntityType'],
                        FieldName.REASON: ans[i]['Reason']
                    })
        return gpt_result_cooked

    @staticmethod
    def parse_gpt_results(gpt_results_raw) -> List[Dict]:
        gpt_results_cooked = []
        for gpt_result_raw in gpt_results_raw:
            gpt_results_cooked += HallucinationDetector.parse_gpt_results_single(gpt_result_raw)
        return gpt_results_cooked

