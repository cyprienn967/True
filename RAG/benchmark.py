import requests
import pickle
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

TEST_SAMPLES = [
    {
        "query": "What is the capital of France?",
        "expected_answer": "Paris"
    },
    {
        "query": "What is the capital of Germany?",
        "expected_answer": "Berlin"
    },
    {
        "query": "Which planet is the largest in our Solar System?",
        "expected_answer": "Jupiter"
    },
    {
        "query": "Is Mars inhabited by humans?",
        "expected_answer": "no"
    },
    {
        "query": "Who painted The Starry Night?",
        "expected_answer": "Vincent van Gogh"
    },
    {
        "query": "Which mountain is the highest on Earth?",
        "expected_answer": "Everest"
    },
    {
        "query": "What is the official language of Brazil?",
        "expected_answer": "Portuguese"
    },
    {
        "query": "Who developed JavaScript?",
        "expected_answer": "Brendan Eich"
    },
    {
        "query": "Where is the Taj Mahal located?",
        "expected_answer": "Agra"
    },
    {
        "query": "When did the Titanic sink?",
        "expected_answer": "1912"
    },
    {
        "query": "What is the speed of light?",
        "expected_answer": "299,792 km/s"
    },
    {
        "query": "Who developed the theory of relativity?",
        "expected_answer": "Einstein"
    },
]

def measure_accuracy(generated_text: str, expected: str) -> bool:
    if not generated_text or not expected:
        return False
    return expected.lower() in generated_text.lower()

def measure_contradiction_or_hallucination(generated_text: str, reference_list: list) -> (bool, bool):
    # currently ery naive logic:
    # - contradiction if we find "largest planet is" in gen but it references e.g. 'Saturn' or 'Mars' instead of 'Jupiter' which doesn't
    # quitttttttteeee work always but oh well
    # - Hallucination if references random weird stuff but so far so good
    known_facts = " ".join(reference_list).lower()
    gen_lower = generated_text.lower()

    contradiction = False
    # example naive rule:
    if "largest planet" in gen_lower and "jupiter" not in gen_lower:
        contradiction = True

    # similarly, if "capital of france" in gen_lower and "paris" not in gen_lower => contradiction
    if "capital of france" in gen_lower and "paris" not in gen_lower:
        contradiction = True
    # lowkey a bit of a hack but works for now will update this logic soon
    # tbh new logic could be if references something the NLI doesn't consider relevant its a hallucination
    hallucination = False
    weird_phrases = ["unicorn", "mermaid city", "alien invasion", "time travel device", "narnia"]
    for phrase in weird_phrases:
        if phrase in gen_lower:
            hallucination = True
            break

    return contradiction, hallucination

def gpt2_generate(query: str) -> str:

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()

    tokenizer.pad_token_id = tokenizer.eos_token_id

    encoded = tokenizer(query, return_tensors="pt", padding=True, truncation=True)
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    max_length = input_ids.shape[1] + 40

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_length=max_length,
            do_sample=False,
            attention_mask=attention_mask,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

def run_benchmark(use_api_verification: bool):
    # 1) for each sample gen gpt2 output
    # 2) If use_api_verification=True => call /verify_step
    # 3) measure accuracy, contradiction, hallucination
    total_samples = len(TEST_SAMPLES)
    correct_count = 0
    contradiction_count = 0
    hallucination_count = 0

    with open("knowledge_base.pkl", "rb") as f:
        knowledge_base = pickle.load(f)

    for i, sample in enumerate(TEST_SAMPLES):
        query = sample["query"]
        expected = sample["expected_answer"]

        raw_gen = gpt2_generate(query)

        if use_api_verification:
            conv_id = f"benchmark-{i}"
            url = "http://localhost:8000/verify_step"
            payload = {
                "conversation_id": conv_id,
                "partial_text": raw_gen,
                "auto_correct": True
            }
            resp = requests.post(url, json=payload)
            if resp.status_code != 200:
                print(f"Error calling API: {resp.status_code} {resp.text}")
                verified_text = raw_gen
            else:
                verified_text = resp.json()["verified_text"]
        else:
            verified_text = raw_gen

        # measure correctness
        if measure_accuracy(verified_text, expected):
            correct_count += 1

        # measure contradiction/hallucination
        is_contr, is_hall = measure_contradiction_or_hallucination(verified_text, knowledge_base)
        if is_contr:
            contradiction_count += 1
        if is_hall:
            hallucination_count += 1

        # debug print
        print("\n=======================")
        print(f"Query: {query}")
        print(f"Raw GPT-2 Gen: {raw_gen}")
        print(f"Verified/Final: {verified_text}")
        print(f"Expected: {expected}")
        print(f"Correct?: {measure_accuracy(verified_text, expected)}")
        print(f"Contradiction?: {is_contr}")
        print(f"Hallucination?: {is_hall}")

    accuracy = correct_count / total_samples
    contradiction_rate = contradiction_count / total_samples
    hallucination_rate = hallucination_count / total_samples

    print("\nBenchmark Results:")
    print(f"Accuracy: {accuracy*100:.2f}%")
    print(f"Contradiction Rate: {contradiction_rate*100:.2f}%")
    print(f"Hallucination Rate: {hallucination_rate*100:.2f}%\n")

if __name__ == "__main__":
    print("=== Baseline: GPT-2 only (no verification) ===")
    run_benchmark(use_api_verification=False)

    print("\n=== With Verification Pipeline ===")
    run_benchmark(use_api_verification=True)
