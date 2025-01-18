readme for the Hallucination API:

So far:
in data/, scripts to extract questions and make new file with hallucinated and non hallucinated QA pairs (questions_with_hallucinations.json)
in scripts/, run_feature_extraction for each QA runs extract_last_token_embedding() and extract_mean_of_last_4_layers()
we then store both the QA pairs and the internal state embeddings in a new file called questions_with_embeddings.jsonl (json lines)
next steps: feed embeddings into a classifier 
ran feature extraction to generate hidden probs (output in file data/questions_with_embeddings)

Next:
Run MLP (wrote script - train_mlp_classifier.py) [don't know if correct haven't tested as of now]
SAVE TILL LAST - Integrated advanced feauture + more fine tuning for better results (i.e. attention scores, last-layer mean for MLP training etc.)
Integration with real time API by running a uvicorn server (can see basic pipeline to set this up in True/RAG/server.py)
Test generation (bioGPT on questions with and without HAPI) - track BERT + ROUGE scores + (number of hallucinations? would process all data after generation through a post-hoc hallucination detector or something of the sort)
Measure AUC, correlation with human labels etc.


