from ..utils.init_model import init_model

# try chain of thought or few shot
def get_topic(input):
  
  model = init_model("gpt-4o-mini")
  
  prompt = "Extract the key topic of the following instruction. Answer only with the key topic. Do not output anything else: "
  full_prompt = prompt + input
  
  output_raw, _, _ = model.predict(prompt=full_prompt, max_completion_tokens=10)
  
  return output_raw.rstrip('.,!?')
  

def generate_validation_prompt(model_name, sentence, context, topic, concept):
  
  model = init_model(model_name)
  
  prompt = f"You are given a statement about {topic}, along with context. Your task is to generate a very simple question that verifies the correctness of the term <{concept}> in the statement. Do not attempt to verify anything else.\nContext: {context}\nStatement: {sentence}"
  
  question, _, _ = model.predict(prompt=prompt, max_completion_tokens=50)
  
  return f"Given the topic {topic}, " + question + " Answer the question in one simple sentence, as briefly as possible: "