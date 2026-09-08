from fastapi import FastAPI

app = FastAPI(
    title="Vector Database From Scratch",
    description="A NumPy-based exact and IVF-Flat vector database.",
    version="0.1.0",
)


@app.get("/health")
def health():
    return {"status": "ok"}