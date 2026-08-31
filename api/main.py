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
    allow_origins=["*"],  # ponytail: restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)

import logging
logger = logging.getLogger(__name__)

# Initialize components
chunker = MultiDocChunker()
search_engine = HybridSearchEngine()
pipeline = LLMPipeline(search_engine)

MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10MB
MAX_DOCUMENTS = 100
MAX_QUERY_LENGTH = 2000


class QueryRequest(BaseModel):
    question: str
    top_k: int = 3
    use_hybrid: bool = True

    def __init__(self, **data):
        super().__init__(**data)
        if len(self.question) > MAX_QUERY_LENGTH:
            raise ValueError(f"Question too long (max {MAX_QUERY_LENGTH} chars)")


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
    if len(request.documents) > MAX_DOCUMENTS:
        raise HTTPException(status_code=400, detail=f"Too many documents (max {MAX_DOCUMENTS})")

    all_chunks = []
    for doc in request.documents:
        text = doc.get("text", "")
        if len(text) > MAX_UPLOAD_SIZE:
            logger.warning(f"Document too large: {len(text)} chars, skipping")
            continue
        doc_type = doc.get("doc_type", "invoice")
        source = doc.get("source", "unknown")
        chunks = chunker.chunk_document(text, doc_type, source)
        all_chunks.extend(chunks)

    if not all_chunks:
        raise HTTPException(status_code=400, detail="No chunks generated from documents.")

    search_engine.build_index(all_chunks)
    logger.info(f"Indexed {len(all_chunks)} chunks from {len(request.documents)} documents")

    return {
        "status": "ok",
        "chunks_indexed": len(all_chunks),
        "documents_processed": len(request.documents),
    }


@app.post("/ingest/files")
async def ingest_files(files: List[UploadFile] = File(...)):
    """Upload and ingest text files."""
    if len(files) > MAX_DOCUMENTS:
        raise HTTPException(status_code=400, detail=f"Too many files (max {MAX_DOCUMENTS})")

    all_chunks = []
    for file in files:
        content = await file.read()
        if len(content) > MAX_UPLOAD_SIZE:
            logger.warning(f"File too large: {file.filename} ({len(content)} bytes), skipping")
            continue
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            logger.warning(f"File not valid UTF-8: {file.filename}, skipping")
            continue
        chunks = chunker.chunk_document(text, "invoice", file.filename)
        all_chunks.extend(chunks)

    if not all_chunks:
        raise HTTPException(status_code=400, detail="No chunks generated from files.")

    search_engine.build_index(all_chunks)
    logger.info(f"Indexed {len(all_chunks)} chunks from {len(files)} files")

    return {
        "status": "ok",
        "chunks_indexed": len(all_chunks),
        "files_processed": len(files),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
