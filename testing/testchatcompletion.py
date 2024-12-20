import openai 

#This works now, was just having to figure out API calls and such

openai.api_key = "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"

try:
    response = openai.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"},
        ],
        model="gpt-3.5-turbo",
        max_tokens = 5,
        temperature = 0.7,
    )
    print(response.choices[0].message.content)
except Exception as e:
    print(f"Error: {e}")
