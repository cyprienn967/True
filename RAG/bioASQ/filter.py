import json


keywords = [
    "infection", "infectious", "virus", "bacteria", "fungi", "pathogen",
    "tuberculosis", "malaria", "HIV", "influenza", "COVID-19", "antibiotic",
    "antiviral", "SARS-CoV-2", "Plasmodium", "Mycobacterium", "Ebola", "Zika",
    "dengue", "hepatitis", "measles", "smallpox", "cholera", "pneumonia", "sepsis"
]


def is_infectious_disease_question(question_body):
    question_lower = question_body.lower()
    return any(keyword in question_lower for keyword in keywords)

# FP
input_file = "training12b_new.json"
output_file = "ID_filtered_bioASQ.json"

try:
    # load
    with open(input_file, "r") as f:
        data = json.load(f)

 
    questions = data.get("questions", [])


    filtered_questions = [
        question for question in questions
        if is_infectious_disease_question(question.get("body", ""))
    ]

    
    filtered_data = {"questions": filtered_questions}

    
    with open(output_file, "w") as f:
        json.dump(filtered_data, f, indent=2)

    print(f"Filtered dataset saved to {output_file}. Total questions: {len(filtered_questions)}")

except FileNotFoundError:
    print(f"Error: The file {input_file} was not found.")
except json.JSONDecodeError:
    print(f"Error: The file {input_file} is not a valid JSON file.")
