import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def visualize_bias(json_file):
    # Construct the full path for the JSON file
    input_dir = os.path.join("..", "results")
    json_path = os.path.join(input_dir, json_file)

    with open(json_path, 'r') as f:
        data = json.load(f)

    # Access the nested "regard_difference" key
    regard_diff = data["bias_results"]["regard_difference"]

    # Spider Plot
    categories = list(regard_diff.keys())
    values = [regard_diff[cat] for cat in categories]
    values += values[:1]  # Close the circle

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)
    ax.fill(angles, values, color='blue', alpha=0.25)
    ax.plot(angles, values, color='blue', linewidth=2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    plt.title("Regard Differences (Spider Plot)")

    # Save the spider plot
    output_dir = os.path.join("..", "..", "Web", "W", "E", "public", "bias")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "bias_spider_plot.png"))
    plt.close()

    # Heatmap
    data_array = np.array(list(regard_diff.values())).reshape(1, -1)
    plt.figure(figsize=(8, 4))
    sns.heatmap(data_array, annot=True, cmap='coolwarm', xticklabels=categories, yticklabels=["Regard Difference"])
    plt.title("Regard Differences (Heatmap)")

    # Save the heatmap
    plt.savefig(os.path.join(output_dir, "bias_heatmap.png"))
    plt.close()

# Example usage
if __name__ == "__main__":
    visualize_bias("bias_evaluation_gpt2.json")
