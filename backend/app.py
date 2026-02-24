import os
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from rag import answer_question
from config import get_settings
from ingest import ingest_one_pdf

app = FastAPI(title="Course RAG Backend")


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    sources: list


@app.get("/health")
def health():
    s = get_settings()
    return {
        "status": "ok",
        "collection": s.collection_name,
        "chroma_dir": s.chroma_dir,
        "top_k": s.top_k,
        "embeddings_provider": s.embeddings_provider,
        "openai_model": s.openai_model,
    }


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    q = req.question.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Question is empty")
    answer, sources = answer_question(q)
    return {"answer": answer, "sources": sources}


@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):

    filename = file.filename or "uploaded.pdf"
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are supported")

    backend_dir = Path(__file__).resolve().parent
    pdf_dir = backend_dir / "data" / "pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)

    safe_name = Path(filename).name
    save_path = pdf_dir / safe_name

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    with open(save_path, "wb") as f:
        f.write(content)

    try:
        chunk_count = ingest_one_pdf(str(save_path))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to ingest PDF: {e}")

    return {
        "status": "ok",
        "filename": safe_name,
        "saved_to": str(save_path),
        "chunks_added": chunk_count,
    }