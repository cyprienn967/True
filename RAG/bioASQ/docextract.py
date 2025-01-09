import json

def extract_document_links(input_file, output_file):
    """
    Extracts document links from the 'documents' section in a JSON file and saves them to a text file.

    Args:
        input_file (str): Path to the input JSON file.
        output_file (str): Path to the output text file.
    """
    try:
        # Open and read the JSON file
        with open(input_file, 'r') as f:
            data = json.load(f)

        # Extract document links
        links = []
        for question in data.get("questions", []):
            if 'documents' in question:
                links.extend(question['documents'])

        # Write the links to the output text file
        with open(output_file, 'w') as f:
            for link in links:
                f.write(link + '\n')

        print(f"Successfully extracted {len(links)} document links to {output_file}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
input_file = 'ID_filtered_bioASQ.json'  # Replace with your actual JSON file path
output_file = 'documents.txt'  # Output text file name
extract_document_links(input_file, output_file)
