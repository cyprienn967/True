STEPS TO RUN:
cd RAG
python -m venv venv (to activate virtual environment)
source venv/bin/activate (virtual environment)
pip install -r requirements.txt (install requirements)
python knowledge_base.py (to create knowledge base - creates file called knowledge_base.pkl)
uvicorn server:app --reload (to run server which you can access at http://127.0.0.1:8000)

To then test THE API:
curl -X POST -H "Content-Type: application/json" \
  -d '{
    "conversation_id":"my-convo-1",
    "partial_text":"Mars has 50 inhabitants",
    "auto_correct": true
  }' \
  http://127.0.0.1:8000/verify_step

(put the above in a seperate terminal (cd into RAG first as well))
check convo:
curl http://127.0.0.1:8000/conversation/my-convo-1

Possible next steps off the top of my head (on top of your next steps):
1. chunking + metadata : better retrieving
2. cross encoder re ranking : after retreive top-k w/ SBERT pass to cross encoder to re rank for better acc
3. more verification layers: quick substring check, NLI check, entity check
4. persistent store: instead of conversation-store.py store partial steps in vertical DB/cahce
