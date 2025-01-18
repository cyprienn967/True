import nltk
from typing import Dict, List

class ResponsePreprocess:
    def __init__(self, skip_starts_with_set, replace_set, filter_start_set):
        self._tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
        self._skip_starts = skip_starts_with_set
        self._replace = replace_set
        self._filter_start = filter_start_set

    def preprocess(self, text: str) -> List[Dict[str, str]]:
        sentences = []
        sentence_id = 0

        for line in text.split('\n'):
            for sent in self._tokenizer.tokenize(line):
                sent = sent.replace('__lf1__', '').replace('__lf2__', '').strip()

                if self._should_skip(sent):
                    continue

                sent = self._clean_sentence(sent)

                if len(sent) > 3:
                    sentences.append({'sentence_id': sentence_id, 'text': sent})
                    sentence_id += 1

        return sentences

    def _should_skip(self, sentence: str) -> bool:
        # return ((sentence.startswith('#') and sentence.endswith('#')) or
        #         sentence.isupper() or
        #         any(sentence.startswith(word) for word in self._skip_starts))
        return False

    def _clean_sentence(self, sentence: str) -> str:
        for word in self._replace:
            sentence = sentence.replace(word, '')
        for word in self._filter_start:
            if sentence.startswith(word):
                sentence = sentence[len(word):]
        return sentence.replace('<|im_end|>', '').strip()
    
def hypothesis_preprocess_into_sentences(hypothesis: str) -> List[Dict[str, str]]:
    # Configure response preprocessing module
    # There is an expectation that the input data is cleaned before running Hallucination Detection
    preprocess_skipStartsWithSet = set([])
    preprocess_replaceWordSet = set(["*"])
    preprocess_filterStartSet = set(["1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.", "10."])
    rp = ResponsePreprocess(
        skip_starts_with_set=preprocess_skipStartsWithSet,
        replace_set=preprocess_replaceWordSet,
        filter_start_set=preprocess_filterStartSet)
    return rp.preprocess(hypothesis)
