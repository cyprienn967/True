import openai

openai.api_key = "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"

try:
    models = openai.Model.list()
    print("Available models:")
    for model in models["data"]:
        print(model["id"])
except Exception as e:
    print(f"Error: {e}")


#AIDAN WARNIG: THIS CODE DOESNT EVEN WORK AND I DONT KNOW WHY

