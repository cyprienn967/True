import uvicorn
from fastapi import FastAPI
from hapi import router as hapi_router

app = FastAPI(
    title="Hallucination-Aware BioGPT API (HAPI)",
    version="1.0.0",
    description="Minimal example server with a hallucination detection pipeline."
)

app.include_router(hapi_router, prefix="/hapi")

@app.get("/")
def root():
    return {"message": "yo"}

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
