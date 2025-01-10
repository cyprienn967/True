import json

def extract_questions(input_file, output_file):
    """
    Reads training12b_new.json, extracts q_id, question, ideal_answer
    and writes out to questions.json with label=0 (non-hallucinated).
    """

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # data["questions"] assumed to be a list of question objects
    questions_list = []
    
    for q in data["questions"]:
        q_id = q.get("id", "")
        question_text = q.get("body", "")
        
        # 'ideal_answer' might be a list of strings or a single string.
        # If it's a list, we can join them. If it's a single string, just use it.
        ans = q.get("ideal_answer", "")
        if isinstance(ans, list):
            ideal_answer_str = " ".join(ans)  # Combine them into one string
        else:
            ideal_answer_str = ans
        
        # Build a simplified record
        record = {
            "q_id": q_id,
            "question": question_text,
            "ideal_answer": ideal_answer_str,
            "label": 0  # By default, this is the correct (non-hallucinated) answer
        }
        
        questions_list.append(record)

    # Write out to JSON
    with open(output_file, 'w', encoding='utf-8') as f_out:
        json.dump(questions_list, f_out, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    input_file = "training12b_new.json"
    output_file = "questions.json"
    extract_questions(input_file, output_file)
    print(f"Extraction complete! Created {output_file}")
