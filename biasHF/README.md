NB: HONEST DOES CURRENTLY NOT WORK FOR DATASET REASONS HENCE WHY IS COMMENTED OUT IN MAIN.PY WILL FIX SOON

Files: 
1. main.py: executes the three main scripts contained in scripts/ + generates .json files containing results in results/
2. scripts/evaluate_toxicity.py: evaluates likelihood of model generating toxic content by using toxicity metric from HF evaluate, output includes overall toxicity ratio and raw toxicity scores for each completion
3. scripts/evaluate_bias.py : evalautes language polarity (bias) in model completions using regard metric. output provides scores for positice, negative, neutral and other polarity categories highlighting diffs between groups
4. scripts/evaluate_honest.py: assesses harmful or hurtful completions by model by using HONEST metric to evaluate stereotypes. 

JSON file interp:

toxicity_ratio: the proportion of toxic completions
raw_toxicity_scores: detailed toxicity scores for each completion
interpretation: a high toxicity ratio indicates that model frequently generates toxic content


regard_difference: diff in language polarity between each group
intepretation: a +ve difference suggests more fabvourable language for one group compared to another

honest_score_per_grpp: hurtfulness scores for each demographic
interpretation: higher scores indicate more hurtful completions for a group


HOW TO RUN:
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py

how to present data (will do tmo hopefully)
1. either like just simple bar graphs/charts
2. interactively: streamlit for dynamic data exploring, dash for data visualization, plotly for heatmaps (heatmap could be cool)



