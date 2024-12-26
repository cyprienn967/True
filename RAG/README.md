uhhhhhh possible next steps type shit (u more knowledgeable than me on this tho)
1. chunking + metadata : better retrieving
2. cross encoder re ranking : after retreive top-k w/ SBERT pass to cross encoder to re rank for better acc
3. more verification layers: quick substring check, NLI check, entity check
4. persistent store: instead of conversation-store.py store partial steps in vertical DB/cahce



to run: in RAG: python knowledge_base.py
uvicorn server:app --reload

after LLM gens partial step:
curl -X POST -H "Content-Type: application/json" \
  -d '{
    "conversation_id":"my-convo-1",
    "partial_text":"Mars has 50 inhabitants",
    "auto_correct": true
  }' \
  http://127.0.0.1:8000/verify_step

check convo:
curl http://127.0.0.1:8000/conversation/my-convo-1

