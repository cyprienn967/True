step-byu-step retrieval and verification API


1. Hybrid Retrieval (FAISS + BM25)  
2. Multi-Document Aggregator for NLI Verification  
3. Retriever–Reader Pipeline with Chain-of-Thought Prompting  
4. LLM-on-LLM Critique for post-generation correction  
5. Benchmarking comparing GPT-2 outputs with/without the verification pipeline

Create a virtual environment and install requirements:
bash
python -m venv venv
source venv/bin/activate
pip install fastapi uvicorn openai sentence-transformers faiss-cpu torch transformers rank-bm25 requests
(requirements file isn't up to date i cba rn)

run: 
python knowledge_base.py to build knowledge base 
(KNOWLEDGE BASE IS BUILT WHEN HAS bm25_index.pkl, knowledge_base.pkl, knowledge_index)
(if already built no need to re run python script but if update knowledge base, delete those files and re run)
uvicorn server:app --host 0.0.0.0 --port 8000 --reload

in another terminal:
python benchmark.py
