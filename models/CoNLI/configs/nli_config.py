from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class DetectionConfig:
    # Configuration for OpenAI GPT Model
    temperature: Optional[float] = 0
    top_p: Optional[float] = 0.6
    max_tokens: Optional[int] = 2048
    # Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim.
    freq_penalty: Optional[float] = 0
    # Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.
    presence_penalty: Optional[float] = 0
    log_prob: Optional[int] = 0
    batch_size: Optional[int] = 1
    n: Optional[int] = 1
    sentence_selector_type: Optional[str] = "pass_through"
    entity_detector_type: Optional[str] = "ta-general"

@dataclass
class MitigationConfig:
    # Configuration for OpenAI GPT Model
    temperature: Optional[float] = 0
    top_p: Optional[float] = 0.6
    max_tokens: Optional[int] = 1024
    # Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim.
    freq_penalty: Optional[float] = 0
    # Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.
    presence_penalty: Optional[float] = 0
    log_prob: Optional[int] = 0
    batch_size: Optional[int] = 10
    n: Optional[int] = 1