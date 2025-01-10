import logging
import transformers
import uvicorn
from fastapi import FastAPI
from hapi import router as hapi_router

# Suppress the roberta / new weights warnings
logging.getLogger("transformers").setLevel(logging.ERROR)
transformers.logging.set_verbosity_error()

app = FastAPI(
    title="Hallucination-Aware BioGPT API (HAPI)",
    version="1.0.0",
    description="Minimal API for BioGPT with MLP-based hallucination detection"
)

# Include our HAPI router
app.include_router(hapi_router, prefix="/hapi")

@app.get("/")
def root():
    return {"message": "Hello from HAPI server!"}

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
