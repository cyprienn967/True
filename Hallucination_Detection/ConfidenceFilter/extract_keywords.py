from ..utils.init_model import init_model

def extract_keywords(input: str):
  
  model = init_model("gpt-4o-mini")
  
  prompt = "For every sentence in the following text, extract the key words and numbers, separated by commas. Output them in a numbered list. Do not output anything else: "
  full_prompt = prompt + input
  
  output_raw, _, _ = model.predict(prompt=full_prompt, max_completion_tokens=1000)
  
  output_split = output_raw.split("\n")
  output_cleaned = [line.split('. ', 1)[1] for line in output_split]
  output_list = [[keyword.strip() for keyword in line.split(",")] for line in output_cleaned]
  # dictionary of lists of key words for every sentence
  output_dict = {i: keyword_list for i, keyword_list in enumerate(output_list)} if output_list else {}
  
  return output_dict