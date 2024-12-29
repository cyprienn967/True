import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from retrieval import hybrid_retrieve

# Load an NLI model for contradiction checking
nli_tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")
nli_model = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli")
nli_model.eval()

def nli_check(claim: str, evidence: str) -> str:
    """
    Returns the NLI label: 'entailment', 'neutral', or 'contradiction'
    using roberta-large-mnli.
    """
    inputs = nli_tokenizer.encode_plus(claim, evidence,
                                       return_tensors="pt",
                                       truncation=True,
                                       max_length=256)
    with torch.no_grad():
        logits = nli_model(**inputs).logits
    label_id = logits.argmax(dim=1).item()
    # roberta-large-mnli: 0->contradiction, 1->neutral, 2->entailment
    if label_id == 0:
        return "contradiction"
    elif label_id == 1:
        return "neutral"
    else:
        return "entailment"

def verify_partial_answer(statement_text: str, top_k: int = 3) -> bool:
    docs = hybrid_retrieve(statement_text, top_k=top_k)

    contradiction_count = 0
    entailment_count = 0
    neutral_count = 0

    for doc in docs:
        label = nli_check(statement_text, doc)
        if label == "contradiction":
            contradiction_count += 1
        elif label == "entailment":
            entailment_count += 1
        else:
            neutral_count += 1

    # For example, if at least 1 doc => entailment, override any contradictions:
    if entailment_count > 0:
        return True
    # Otherwise if contradiction_count > 0 => fail
    if contradiction_count > 0:
        return False
    # If everything is neutral => pass
    return True
