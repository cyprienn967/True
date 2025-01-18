import math
from typing import List

# this only works for gpt models, other tokenizers begin prefixes with ## for subwords
def process_tokens(logliks, tokens):
  
  print(f"Tokens: {tokens}")
  
  if (len(logliks) != len(tokens)):
    raise ValueError("Length of logliks and tokens must be the same")
  
  probs = []
  words = []
  
  for i in range(len(tokens)):
    
    if i == 0:
      words.append(tokens[i])
      probs.append(math.exp(logliks[i]))
      continue
  
    # if the token is a newline character, there are no spaces so we have to manually add them
    if tokens[i] == ".\n\n" or tokens[i] == ".\n" or tokens[i] == ".\n\n\n" or tokens[i] == ".\n\n\n\n":
      if i < len(tokens) - 1:
        tokens[i+1] = " " + tokens[i+1]
      continue
    # if (tokens[i][0].isalpha() and tokens[i-1][-1].isalpha()) or (tokens[i][0].isdigit() and tokens[i-1][-1].isdigit()):
    #   logits.append(logliks[i])
    if tokens[i][0].isspace():
      words.append(tokens[i][1:])
      probs.append(math.exp(logliks[i]))
    else:
      words[-1] += tokens[i]
      probs[-1] = min(probs[-1], math.exp(logliks[i]))
      
  print(words, probs)
      
  return words, probs


def filter_hypotheses(hypotheses, keyword_list, probs):
  
  hypothesis_evaluations = [False] * len(hypotheses) # if keywords have low generated likelihood, they get flagged
  hallucinated_keywords = {i: [] for i in range(len(hypotheses))} # (keyword, likelihood) pairs
  
  token_pointer = 0
  keyword_pointer = 0
  
  for i in range(len(hypotheses)):
    hypothesis = hypotheses[i]['text']
    print(hypothesis)
    
    word_pointer = 0
    
    while word_pointer < len(hypothesis.split()) and keyword_pointer < len(keyword_list):
      
      first_keyword = keyword_list[keyword_pointer].split()[0]
      # print(hypothesis.split()[pointer1], first_keyword)
      first_hypothesis_word = hypothesis.split()[word_pointer]
      
      if first_hypothesis_word.lower() == first_keyword.lower() or first_hypothesis_word[:-1].lower() == first_keyword.lower() or first_hypothesis_word[:-2].lower() == first_keyword.lower(): # found a keyword
        print(f"Found first keyword match: {keyword_list[keyword_pointer]}")
        num_words = len(keyword_list[keyword_pointer].split())
        hypothesis_words = ' '.join(hypothesis.split()[word_pointer:word_pointer+num_words])
        print(f"Hypothesis words: {hypothesis_words}, Num words: {num_words}")
        
        if hypothesis_words.lower() == keyword_list[keyword_pointer].lower() or hypothesis_words[:-1].lower() == keyword_list[keyword_pointer].lower():
          print(f"Found full keyword match: {keyword_list[keyword_pointer]}")
          min_likelihood = 1.0
          
          for j in range(num_words):
            min_likelihood = min(min_likelihood, probs[token_pointer+j])
            
          print(f"Min likelihood: {min_likelihood}")
          
          if min_likelihood < 0.8: # if the keyword has low likelihood, flag it
            # print(token_pointer)
            hypothesis_evaluations[i] = True
            hallucinated_keywords[i].append((keyword_list[keyword_pointer], min_likelihood))
            print("flagged!")
            print(min_likelihood)
            
          word_pointer += num_words
          keyword_pointer += 1
          token_pointer += num_words
          
        else:
          word_pointer += 1
          token_pointer += 1
          
      else:
        word_pointer += 1
        token_pointer += 1
      
  return hypothesis_evaluations, hallucinated_keywords
      
      
# words = [
#         "If",
#         "it",
#         "is",
#         "2025,",
#         "then",
#         "the",
#         "current",
#         "year",
#         "is",
#         "2025."
#     ]
# probs = [
#         0.6142640772853443,
#         0.9228905926611642,
#         0.9999762043451211,
#         0.8145903664082348,
#         0.9998869269092439,
#         0.9957515042013829,
#         0.18241278693294902,
#         1.0,
#         0.9994974910999104,
#         0.9450310829677123
#     ]

# hypotheses = [{'text': "If it is 2025, then the current year is 2025."}]
# keyword_dict = {0: ["2025",
#             "current year",
#             "2025"]}

# print(filter_hypotheses(hypotheses, keywords, probs))