import math

# this only works for gpt models, other tokenizers begin prefixes with ## for subwords
def process_tokens(logliks, tokens):
  
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


def filter_hypotheses(hypotheses, keyword_dict, probs):
  filtered_hypotheses = [] # if keywords have low generated likelihood, they get flagged
  token_pointer = 0
  
  for i in range(len(hypotheses)):
    hypothesis = hypotheses[i]['text']
    keywords = keyword_dict.get(i, [])
    
    pointer1 = 0
    pointer2 = 0
    while pointer1 < len(hypothesis.split()) and pointer2 < len(keywords):
      
      first_keyword = keywords[pointer2].split()[0]
      # print(hypothesis.split()[pointer1], first_keyword)
      
      if hypothesis.split()[pointer1] == first_keyword or hypothesis.split()[pointer1][:-1] == first_keyword: # found a keyword
        # print(keyword)
        num_words = len(keywords[pointer2].split())
        hypothesis_words = ' '.join(hypothesis.split()[pointer1:pointer1+num_words])
        
        if hypothesis_words == keywords[pointer2] or hypothesis_words[:-1] == keywords[pointer2]:
          for j in range(num_words):
            if probs[token_pointer+j] < 0.8: # if the keyword has low likelihood, flag it
              # print(token_pointer)
              filtered_hypotheses.append(i)
              break
          pointer1 += num_words
          pointer2 += num_words
          token_pointer += num_words
        else:
          pointer1 += 1
          token_pointer += 1
      else:
        pointer1 += 1
        token_pointer += 1
      
  return filtered_hypotheses
      
      
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