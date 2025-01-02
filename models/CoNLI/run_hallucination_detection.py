import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
import os
from pathlib import Path
import time
from tqdm import tqdm
from CoNLI.configs.nli_config import DetectionConfig
from CoNLI.configs.openai_config import create_openai_arguments
from CoNLI.configs.ta_config import create_ta_arguments
from CoNLI.modules.data.data_loader import DataLoader
from CoNLI.modules.entity_detector import EntityDetectorFactory
from CoNLI.modules.sentence_selector import SentenceSelectorFactory
from CoNLI.modules.hallucination_detector import HallucinationDetector
from CoNLI.modules.hd_constants import AllHallucinations, FieldName
from CoNLI.modules.utils.conversion_utils import str2bool

def get_optional_field(hallucination, field_name, default_value = ''):
    if field_name in hallucination:
        return str(hallucination[field_name])
    else:
        return default_value

def get_required_field(hallucination, field_name):
    return str(hallucination[field_name])

def save_hallucinations(hallucinations, output_folder : str):
    hallucination_finalresults = os.path.join(output_folder, 'HallucinationFinal.tsv')
    
    # Sort all of the hallucinations by data and sentence id
    # hallucinations contextual order before passing to requester
    hallucinations = sorted(
            hallucinations,
            key=lambda d: (
                d[FieldName.DATA_ID],
                d[FieldName.SENTENCE_ID],
                d[FieldName.DETECTION_TYPE],
                d[FieldName.SENTENCE_TEXT]
            )
        )
    
    with open(hallucination_finalresults, 'w') as outFinal:
        outFinal.write('data_id\tsentenceid\tdetectiontype\tspan\treason\tname\ttype\n')
    
        required_field_names = [
                        FieldName.DATA_ID,
                        FieldName.SENTENCE_ID,
                        FieldName.DETECTION_TYPE,
                        FieldName.SENTENCE_TEXT,
                        FieldName.REASON
                        ]
        optional_field_names = [
                        FieldName.NAME,
                        FieldName.TYPE
                        ]
        for h in hallucinations:
            field_values = [get_required_field(h, fn) for fn in required_field_names] + [get_optional_field(h, fn) for fn in optional_field_names]
            outFinal.write('\t'.join(field_values) + '\n')

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_hypothesis',
        required=True,
        help='The folder where all of your raw responses are located; It can also load sentence-level tsv file with columns: DataID, SentenceID, Sentence. The Sentence will be the input strings for detection',
        type=str)
    parser.add_argument(
        '--entity_detector_type',
        default="text_analytics",
        help='entity detector type: pass_through, text_analytics. If ensembled, you must also specify as ensembled:type1,type2 ...',
        type=str)
    parser.add_argument(
        '--sentence_selector_type',
        default="pass_through",
        help='entity detector type: pass_through, None. If ensembled, you must also specify as ensembled:type1,type2 ...',
        type=str)
    parser.add_argument(
        '--ta_config_file',
        default=(Path(__file__).absolute()).parent/"configs"/"ta_config.json",
        help='JSON file holding the text analytics endpoint configs',
        type=str)
    parser.add_argument(
        '--max_parallel_data',
        default=2,
        help='The maximum number of data to process in parallel.  If set to 1, will run sequentially',
        type=int)
    parser.add_argument(
        '--max_parallelism',
        default=2,
        help='The maximum number of GPT requests to send in parallel per Hallucination Detection Module.  If set to 1, will run sequentially',
        type=int)
    parser.add_argument(
        '--entity_detection_parallelism',
        default=2,
        help='The maximum number of entity detection batches to process in parallel per Hallucination Detection Module.  If set to 1, will run sequentially',
        type=int)
    parser.add_argument('--gpt_batch_size', default=1, type=int)
    parser.add_argument('--log_level', default='info')
    parser.add_argument('--logfile_name', default=None)

    args = parser.parse_args()

    args.max_parallel_data = max(args.max_parallel_data, 1)
    args.max_parallelism = max(args.max_parallelism, 1)
    args.entity_detection_parallelism = max(args.entity_detection_parallelism, 1)
    args.test_mode = max(args.test_mode, 0)
    
    return args

def run_hallucination_detection(prompt):
    
    logging.info('Starting Hallucination Detection')

    openai_args = create_openai_arguments(2)
    ta_args = create_ta_arguments('ta-general', ta_config_file=args.ta_config_file)
    detection_config = DetectionConfig()
    detection_config.batch_size = 1
    
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    # print('Disabling Azure Telemetry (Doesnt handle parallelism well)')
    # os.environ['AZURE_CORE_COLLECT_TELEMETRY'] = 'false'

    start_time = time.time()

    # hallucination_result_folder = os.path.join(args.output_folder, 'hallucinations')
    # os.makedirs(hallucination_result_folder, exist_ok=True)
    # intermediate_result_folder = os.path.join(args.output_folder, 'intermediate')
    # os.makedirs(intermediate_result_folder, exist_ok=True)

    # dataloader = DataLoader(
    #     hypothesis=args.input_hypothesis,
    #     src_folder=args.input_src,
    #     test_mode=args.test_mode)

    # hypothesis = dataloader._hypothesis  # Not used
    # source_docs = dataloader._src_docs
    # hyp_sentences_preproc = dataloader._hypothesis_preproc_sentences
    # data_ids = dataloader._data_ids

    sentence_selector = SentenceSelectorFactory.create_sentence_selector(detection_config.sentence_selector_type)
    entity_detector = EntityDetectorFactory.create_entity_detector(detection_config.entity_detector_type, ta_args=ta_args)

    detection_agent = HallucinationDetector(
        sentence_selector=sentence_selector,
        entity_detector=entity_detector,
        openai_config=openai_args,
        detection_config=detection_config,
        entity_detection_parallelism=args.entity_detection_parallelism)

    allHallucinations = []
    retval_jsonl = []
    
    hypotheses = hypothesis_preprocess_into_sentences(hypothesis)
    
    hallucinations = detection_agent.detect_hallucinations(id, prompt, sentences)
    for h in hallucinations:
        allHallucinations.append(h)
    num_sentences : int = len(hyp_sentences_preproc[data_id])
    num_hallucinations : int = len(hallucinations)
    hallucination_rate : float = num_hallucinations / num_sentences if num_sentences > 0 else 0.0
    hallucinated : bool = num_hallucinations > 0
    retval_jsonl.append(
        {
            AllHallucinations.DATA_ID: data_id,
            AllHallucinations.HALLUCINATED: hallucinated,
            AllHallucinations.HALLUCINATION_SCORE: hallucination_rate,
            AllHallucinations.HALLUCINATIONS: hallucinations,
            AllHallucinations.NUM_TOTAL_SENTENCES: num_sentences,
            AllHallucinations.NUM_TOTAL_HALLUCINATIONS: num_hallucinations,
        })

    #detection_agent.PrintHallucinations(allHallucinations)
    save_hallucinations(allHallucinations, intermediate_result_folder)
    # detection_agent.SortDebugHallucinations()   

    # end_time = time.time() - start_time
    # print('Hallucination Detection Has Finished')
    # print(f'Total wall-clock time: {end_time} seconds')
    # print(f'Final output written to {outputFilePath}')
