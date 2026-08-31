"""
FastAPI Backend for Finance Invoice RAG Bot

Endpoints:
- POST /query — Ask a question about invoices
- POST /ingest — Upload documents for indexing
- GET /health — Health check
- GET /stats — Index statistics
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os

from core.domain_chunker import MultiDocChunker
from core.hybrid_search import HybridSearchEngine
from core.llm_pipeline import LLMPipeline


app = FastAPI(
    title="Finance Invoice RAG Bot",
    description="Ask questions about your invoices using RAG with domain-aware chunking",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
chunker = MultiDocChunker()
search_engine = HybridSearchEngine()
pipeline = LLMPipeline(search_engine)


class QueryRequest(BaseModel):
    question: str
    top_k: int = 3
    use_hybrid: bool = True


class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]
    groundedness_score: float
    is_grounded: bool
    unsupported_claims: List[str]
    retrieval_used: bool


class IngestRequest(BaseModel):
    documents: List[dict]  # [{"text": "...", "doc_type": "invoice", "source": "file.pdf"}]


@app.get("/health")
def health_check():
    return {"status": "ok", "indexed_chunks": len(search_engine.chunks)}


@app.get("/stats")
def get_stats():
    return {
        "total_chunks": len(search_engine.chunks),
        "index_built": search_engine.is_built,
        "model": search_engine.embedding_model_name,
    }


@app.post("/query", response_model=QueryResponse)
def query_invoices(request: QueryRequest):
    """Ask a question about the indexed invoices."""
    if not search_engine.is_built:
        raise HTTPException(status_code=400, detail="No documents indexed yet. Upload documents first.")

    response = pipeline.query(request.question, top_k=request.top_k)

    return QueryResponse(
        answer=response.answer,
        sources=response.sources,
        groundedness_score=response.groundedness_score,
        is_grounded=response.is_grounded,
        unsupported_claims=response.unsupported_claims,
        retrieval_used=response.retrieval_used,
    )


@app.post("/ingest")
def ingest_documents(request: IngestRequest):
    """Ingest documents into the search index."""
    all_chunks = []

    for doc in request.documents:
        text = doc.get("text", "")
        doc_type = doc.get("doc_type", "invoice")
        source = doc.get("source", "unknown")

        chunks = chunker.chunk_document(text, doc_type, source)
        all_chunks.extend(chunks)

    if not all_chunks:
        raise HTTPException(status_code=400, detail="No chunks generated from documents.")

    search_engine.build_index(all_chunks)

    return {
        "status": "ok",
        "chunks_indexed": len(all_chunks),
        "documents_processed": len(request.documents),
    }


@app.post("/ingest/files")
async def ingest_files(files: List[UploadFile] = File(...)):
    """Upload and ingest text files."""
    all_chunks = []

    for file in files:
        content = await file.read()
        text = content.decode("utf-8")

        chunks = chunker.chunk_document(text, "invoice", file.filename)
        all_chunks.extend(chunks)

    if not all_chunks:
        raise HTTPException(status_code=400, detail="No chunks generated from files.")

    search_engine.build_index(all_chunks)

    return {
        "status": "ok",
        "chunks_indexed": len(all_chunks),
        "files_processed": len(files),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
