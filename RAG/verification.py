import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from retrieval import retrieve_relevant_docs

# Load an NLI model for contradiction checks
nli_tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")
nli_model = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli")
nli_model.eval()

def nli_check(claim: str, evidence: str) -> bool:

    # Returns True if claim is NOT contradicted by evidence (for now at least)
    # RLMNLI output lbels
    #  0: contradiction
    #  1: neutral
    #  2: entailment
    #  consider contradiction => return False.

    inputs = nli_tokenizer.encode_plus(claim, evidence, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = nli_model(**inputs).logits
    label_id = logits.argmax(dim=1).item()
    return label_id != 0  #see above

def verify_partial_answer(partial_text: str, top_k: int = 3) -> bool:
    
    docs = retrieve_relevant_docs(partial_text, top_k=top_k)
    for doc in docs:
        if not nli_check(partial_text, doc):
            print("\nDEBUG: Verification failed!")
            print("DEBUG: partial_text =", partial_text)
            print("DEBUG: doc that caused contradiction =", doc)
            return False
    return True
