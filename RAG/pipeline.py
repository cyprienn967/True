import openai
from retrieval import hybrid_retrieve

def chain_of_thought_reader(query: str, top_k: int = 3) -> str:
    # example retriever-reader pipeline that uses chain-of-thought prompting. yes
    # 1) retrieve top_k docs 2) provide them as context to an LLM 3) ask LLM to reason step-by-step internally but provide a final concise answer
    relevant_docs = hybrid_retrieve(query, top_k=top_k)
    system_prompt = (
        "You are a helpful reading comprehension assistant. We retrieved these references:\n"
        + "\n".join(f"- {doc}" for doc in relevant_docs)
        + "\nUse them to accurately answer the user's query. "
        + "You may do chain-of-thought reasoning internally, but only provide a concise final answer."
    )
    user_prompt = f"Question: {query}"

    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.5,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[Reader Error: {str(e)}]"
