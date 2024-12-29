import json
import matplotlib.pyplot as plt
import os

def visualize_toxicity(json_file):
    # Construct the full path for the JSON file
    input_dir = os.path.join("..", "results")
    json_path = os.path.join(input_dir, json_file)

    with open(json_path, 'r') as f:
        data = json.load(f)

    # Extract the array of toxicity scores
    toxicity_scores = data["toxicity_results"]["toxicity"]

    # Calculate the counts for toxic and non-toxic entries
    toxic_count = sum(1 for score in toxicity_scores if score > 0.5)
    non_toxic_count = len(toxicity_scores) - toxic_count

    labels = ['Toxic', 'Non-Toxic']
    sizes = [toxic_count, non_toxic_count]
    colors = ['red', 'green']

    plt.figure(figsize=(6, 6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=140)
    plt.title("Toxicity Distribution")

    # Save the pie chart
    output_dir = os.path.join("..", "..", "Web", "W", "E", "public", "bias")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "toxicity_pie_chart.png"))
    plt.close()

# Example usage
if __name__ == "__main__":
    visualize_toxicity("toxicity_evaluation_gpt2.json")
