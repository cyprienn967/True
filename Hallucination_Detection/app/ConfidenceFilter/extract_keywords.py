from ..utils.init_model import init_model
from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics.aio import TextAnalyticsClient
import os
import nltk

def extract_keywords(keyword_model, api_key, input: str):
  
  if (len(input) <= 5):
    return input.split()
  
  if ("gpt" in keyword_model.lower()):
    
    model = init_model(keyword_model, api_key)
    
    sentences = nltk.sent_tokenize(input)
    
    prompt = "For every sentence in the following text, extract the key words and numbers, separated by commas. Output them in a numbered list. Do not output unnamed entities that aren't crucial: "
    full_prompt = prompt + "\n".join([f"{i+1}. {sentence}" for i, sentence in enumerate(sentences)])
    
    output_raw, _, _ = model.predict(prompt=full_prompt, max_completion_tokens=1000)
    
    print(output_raw)
    
    output_split = output_raw.split("\n")
    try:
      output_cleaned = [line.split('. ', 1)[1] for line in output_split]
    except:
      output_cleaned = output_split
    output_list = [[keyword.strip() for keyword in line.split(",")] for line in output_cleaned]
    
    flattened_output_list = [keyword for sublist in output_list for keyword in sublist]
    
    return flattened_output_list
    # # dictionary of lists of key words for every sentence
    # output_dict = {i: keyword_list for i, keyword_list in enumerate(output_list)} if output_list else {}
    
    # return output_dict
  
  else:
    endpoint = os.environ.get('LANGUAGE_ENDPOINT', None)
    
    ta_client = TextAnalyticsClient(endpoint=endpoint, credential=api_key)
    
    keywords = ta_client.ExtractKeyPhrases