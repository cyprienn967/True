

class KeywordConfig():
  def __init__(self):
    self.model = "gpt-4o-mini"
    self.max_completion_tokens = 1000
    self.prompt = "For every sentence in the following text, extract the key words and numbers. Output them in a numbered list. Do not output anything else: "
    