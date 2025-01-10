from ..utils.init_model import init_model

def check_contradiction(model_name, hypothesis, sampled_responses, num_samples):
  
  model = init_model(model_name)
  
  num_samples = min(num_samples, len(sampled_responses))
  prompt = "Given the following statement, determine if any of the following responses prove that it must be false. Answer 'yes' if any of the responses contradict it, and 'no' if they do not contradict it. Do not output anything else: "
  prompt += f"\nStatement: {hypothesis}"
  
  for i in range(num_samples):
    prompt += f"\nResponse {i+1}: {sampled_responses[i]}"
    
  output, _, _ = model.predict(prompt=prompt, max_completion_tokens=5)
  print(output)
  
  return output.lower() == "yes"