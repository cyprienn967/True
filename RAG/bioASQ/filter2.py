import json


input_file = "ID_filtered_bioASQ.json"  
output_file = "simple_ID_filtered_bioASQ.json"  

# Fields to extract
fields_to_extract = ["id", "body", "ideal_answer", "exact_answer"]

try:
    
    with open(input_file, "r") as f:
        data = json.load(f)

    
    questions = data.get("questions", [])

    
    simplified_questions = [
        {field: question.get(field, None) for field in fields_to_extract}
        for question in questions
    ]

    
    with open(output_file, "w") as f:
        json.dump({"questions": simplified_questions}, f, indent=2)

    print(f"Simplified dataset saved to {output_file}. Total questions: {len(simplified_questions)}")

except FileNotFoundError:
    print(f"Error: The file {input_file} was not found.")
except json.JSONDecodeError:
    print(f"Error: The file {input_file} is not a valid JSON file.")
