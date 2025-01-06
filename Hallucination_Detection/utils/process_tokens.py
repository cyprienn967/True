import math

# this only works for gpt models, other tokenizers begin prefixes with ## for subwords
# def process_tokens(logliks, tokens):
  
#   if (len(logliks) != len(tokens)):
#     raise ValueError("Length of logliks and tokens must be the same")
  
#   probs = []
#   words = []
  
#   for i in range(len(tokens)):
    
#     if i == 0:
#       words.append(tokens[i])
#       probs.append(math.exp(logliks[i]))
#       continue
      
#     # if (tokens[i][0].isalpha() and tokens[i-1][-1].isalpha()) or (tokens[i][0].isdigit() and tokens[i-1][-1].isdigit()):
#     #   logits.append(logliks[i])
#     if tokens[i][0].isspace():
#       words.append(tokens[i][1:])
#       probs.append(math.exp(logliks[i]))
#     else:
#       words[-1] += tokens[i]
#       probs[-1] = min(probs[-1], math.exp(logliks[i]))
      
#   return words, probs

def process_tokens(logliks, tokens, keywords):
  
  if (len(logliks) != len(tokens)):
    raise ValueError("Length of logliks and tokens must be the same")
  
  probs = []
  words = []
  
  for i in range(len(tokens)):
    
    if i == 0:
      words.append(tokens[i])
      probs.append(math.exp(logliks[i]))
      continue
      
    # if (tokens[i][0].isalpha() and tokens[i-1][-1].isalpha()) or (tokens[i][0].isdigit() and tokens[i-1][-1].isdigit()):
    #   logits.append(logliks[i])
    if tokens[i][0].isspace():
      words.append(tokens[i][1:])
      probs.append(math.exp(logliks[i]))
    else:
      words[-1] += tokens[i]
      probs[-1] = min(probs[-1], math.exp(logliks[i]))
      
  return words, probs

tokens = ["hello", " world", " this", " is", " a", " test", "ing", " code", "!"]
logits = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
print(process_tokens(logits, tokens))
