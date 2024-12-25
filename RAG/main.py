import openai
import requests


openai.api_key = "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"

API_URL = "http://127.0.0.1:8000/check_hallucination"

def generate_response(prompt):
    # gen reply
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
        {"role": "system", "content": "You are an assistant."},
        {"role": "user", "content": prompt}
        ],
        max_tokens=50,
        stream=False
    )
    generated_text = response.choices[0].message.content.strip()

    # Call the hallucination detection API
    payload = {"generated_text": generated_text}
    api_response = requests.post(API_URL, json=payload).json()

    if api_response["hallucinations"]:
        print("Potential hallucination detected!")
        print("Relevant documents:", api_response["relevant_docs"])
        # Optionally regenerate
    else:
        print("Response is likely factual.")
        print("Relevant documents:", api_response["relevant_docs"])

    return generated_text

# test the system
prompt = "What is the population of Mars?"
response = generate_response(prompt)
print("LLM Response:", response)
