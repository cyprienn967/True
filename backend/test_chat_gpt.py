import openai
import sys

def get_average_word_count(api_key):
    openai.api_key = api_key

    prompts = [
        "What are some examples of bias in AI?",
        "Can AI systems reinforce bias?",
        "What is algorithmic bias?",
        "How does bias affect decision-making in AI?",
        "Explain bias in machine learning models.",
        "How can we mitigate bias in AI systems?",
        "What is the impact of bias in hiring algorithms?",
        "Give an example of bias in facial recognition systems.",
        "Explain bias in recommendation algorithms.",
        "How do training datasets introduce bias?"
    ]

    total_words = 0

    for prompt in prompts:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages= [
                    {"role": "system", "content": "You are an unbiased assistant."},
                    {"role": "user", "content": prompt},
                ],
            max_tokens=20,
        )
        text = response.choices[0].message.content.strip()
        word_count = len(text.split())
        total_words += word_count

    average_word_count = total_words / len(prompts)
    return average_word_count

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_chat_gpt.py <API_KEY>")
        sys.exit(1)

    api_key = sys.argv[1]
    average = get_average_word_count(api_key)
    print(average)
