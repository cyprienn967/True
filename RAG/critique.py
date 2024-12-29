import openai

def correct_after_critique(original_text: str, relevant_docs: list[str]) -> str:

    docs_string = "\n".join(f"- {doc}" for doc in relevant_docs)
    system_prompt = (
        "You are a factual correctness assistant. Below are relevant docs:\n"
        f"{docs_string}\n\n"
        "Rewrite the partial text so that it is factually correct and consistent:\n"
        f"{original_text}\n"
        "Return just the corrected text, 1-2 sentences max."
    )

    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
            ],
            max_tokens=150,
            temperature=0.0,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[Correction Error: {str(e)}]"
