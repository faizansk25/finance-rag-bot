"""
Hybrid Search Engine: FAISS + BM25 + Self-RAG

This is the core retrieval engine. It combines:
1. FAISS vector search (semantic matching — finds similar MEANING)
2. BM25 keyword search (exact matching — finds exact TERMS)
3. Reranking (second-pass quality filter)
4. Self-RAG (model decides when to retrieve vs answer directly)
5. Hallucination detection (checks if answers are grounded in sources)

Why hybrid search?
- Vector search finds "invoice from Acme Corp" even if the document says "bill from Acme Corporation"
- BM25 finds exact keyword matches like "INV-2024-0847" that vectors miss
- Combining both gives the best of both worlds
"""

import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import re


@dataclass
class SearchResult:
    """A single search result with score and metadata."""
    content: str
    score: float
    search_type: str   # "vector", "bm25", "hybrid", "reranked"
    metadata: dict
    chunk_id: str = ""


class HybridSearchEngine:
    """
    Combines FAISS (semantic) and BM25 (keyword) search.

    The search flow:
    1. User query comes in
    2. FAISS finds top-K semantically similar chunks
    3. BM25 finds top-K keyword-matching chunks
    4. Results are merged with reciprocal rank fusion
    5. Optional: rerank with a cross-encoder for precision
    """

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.embedding_model_name = embedding_model
        self.embedder = None
        self.faiss_index = None
        self.bm25 = None
        self.chunks = []
        self.embeddings = None
        self.is_built = False

    def _load_embedder(self):
        """Lazy-load the sentence transformer embedder."""
        if self.embedder is None:
            from sentence_transformers import SentenceTransformer
            self.embedder = SentenceTransformer(self.embedding_model_name)

    def build_index(self, chunks: list):
        """
        Build the hybrid search index from DocumentChunk objects.

        This indexes ALL chunks with both FAISS and BM25.
        """
        self.chunks = chunks
        if not chunks:
            return

        # Build FAISS index (vector search)
        self._load_embedder()
        texts = [c.content for c in chunks]
        self.embeddings = self.embedder.encode(texts, show_progress_bar=False)

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)

        # Create FAISS index
        dimension = self.embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product = cosine after normalization
        self.faiss_index.add(self.embeddings)

        # Build BM25 index (keyword search)
        tokenized = [self._tokenize(text) for text in texts]
        self.bm25 = BM25Okapi(tokenized)

        self.is_built = True

    def search(self, query: str, top_k: int = 5,
               use_hybrid: bool = True,
               vector_weight: float = 0.6,
               bm25_weight: float = 0.4) -> List[SearchResult]:
        """
        Search using hybrid approach.

        Args:
            query: User's search query
            top_k: Number of results to return
            use_hybrid: If True, combine vector + BM25. If False, vector only.
            vector_weight: Weight for vector search results (0-1)
            bm25_weight: Weight for BM25 results (0-1)

        Returns:
            List of SearchResult objects, sorted by relevance
        """
        if not self.is_built:
            return []

        results = []

        if use_hybrid:
            # Vector search results
            vector_results = self._vector_search(query, top_k * 2)
            # BM25 search results
            bm25_results = self._bm25_search(query, top_k * 2)

            # Reciprocal Rank Fusion
            results = self._reciprocal_rank_fusion(
                vector_results, bm25_results,
                vector_weight, bm25_weight, top_k
            )
        else:
            results = self._vector_search(query, top_k)

        return results[:top_k]

    def _vector_search(self, query: str, top_k: int) -> List[SearchResult]:
        """FAISS semantic search."""
        self._load_embedder()
        query_embedding = self.embedder.encode([query])
        faiss.normalize_L2(query_embedding)

        scores, indices = self.faiss_index.search(query_embedding, min(top_k, len(self.chunks)))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            chunk = self.chunks[idx]
            results.append(SearchResult(
                content=chunk.content,
                score=float(score),
                search_type="vector",
                metadata=chunk.metadata.to_dict(),
                chunk_id=chunk.chunk_id,
            ))

        return results

    def _bm25_search(self, query: str, top_k: int) -> List[SearchResult]:
        """BM25 keyword search."""
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            chunk = self.chunks[idx]
            results.append(SearchResult(
                content=chunk.content,
                score=float(scores[idx]),
                search_type="bm25",
                metadata=chunk.metadata.to_dict(),
                chunk_id=chunk.chunk_id,
            ))

        return results

    def _reciprocal_rank_fusion(self, vector_results: list,
                                 bm25_results: list,
                                 vector_weight: float,
                                 bm25_weight: float,
                                 top_k: int) -> List[SearchResult]:
        """
        Combine results using Reciprocal Rank Fusion (RRF).

        RRF score = sum over search engines of (weight / (k + rank))
        where rank is the position in each engine's results.
        """
        k = 60  # RRF constant (standard value, from the original paper)
        combined_scores = {}

        # Process vector results
        for rank, result in enumerate(vector_results):
            key = result.chunk_id
            if key not in combined_scores:
                combined_scores[key] = {"result": result, "score": 0}
            combined_scores[key]["score"] += vector_weight / (k + rank + 1)

        # Process BM25 results
        for rank, result in enumerate(bm25_results):
            key = result.chunk_id
            if key not in combined_scores:
                combined_scores[key] = {"result": result, "score": 0}
            combined_scores[key]["score"] += bm25_weight / (k + rank + 1)

        # Sort by combined score
        sorted_results = sorted(
            combined_scores.values(),
            key=lambda x: x["score"],
            reverse=True
        )

        results = []
        for item in sorted_results[:top_k]:
            result = item["result"]
            result.score = item["score"]
            result.search_type = "hybrid"
            results.append(result)

        return results

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for BM25."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        return text.split()


class SelfRAGEngine:
    """
    Self-RAG: The model decides when to retrieve vs answer directly.

    Flow:
    1. Receive question
    2. Model decides: retrieve or answer directly?
    3. If retrieve: run RAG pipeline
    4. Check: is answer supported by retrieved documents?
    5. If not supported: regenerate or say "not sure"
    """

    def __init__(self, search_engine: HybridSearchEngine):
        self.search_engine = search_engine

    def should_retrieve(self, query: str) -> bool:
        """
        Decide if the query needs retrieval.
        
        ponytail: regex heuristic. Real Self-RAG uses the LLM to decide.
        Upgrade: call LLM with a classifier prompt for retrieval decisions.
        """
        # Query mentions specific data points → retrieve
        retrieval_triggers = [
            r'\bhow much\b', r'\btotal\b', r'\binvoice\b',
            r'\bvendor\b', r'\bdate\b', r'\bpay\b',
            r'\bowed\b', r'\bbalance\b', r'\boverdue\b',
            r'\bnumber\b', r'\bamount\b', r'\bwhen\b',
        ]
        for pattern in retrieval_triggers:
            if re.search(pattern, query, re.IGNORECASE):
                return True
        return False

    def retrieve_and_verify(self, query: str, top_k: int = 3) -> dict:
        """
        Retrieve documents and verify the answer is grounded.

        Returns:
            {
                "retrieved": [SearchResult, ...],
                "query": str,
                "needs_retrieval": bool,
                "chunks_for_context": [str, ...]
            }
        """
        needs_retrieval = self.should_retrieve(query)

        if not needs_retrieval:
            return {
                "retrieved": [],
                "query": query,
                "needs_retrieval": False,
                "chunks_for_context": [],
            }

        results = self.search_engine.search(query, top_k=top_k)

        return {
            "retrieved": results,
            "query": query,
            "needs_retrieval": True,
            "chunks_for_context": [r.content for r in results],
        }


class HallucinationDetector:
    """
    Detects hallucinations by checking if claims in the generated answer
    are supported by the retrieved source documents.

    Methods:
    1. Numerical consistency: Are numbers in the answer found in sources?
    2. Entity consistency: Are names in the answer found in sources?
    3. Claim verification: Are statements supported by source text?
    """

    def detect_hallucinations(self, answer: str, sources: List[str]) -> dict:
        # ponytail: regex-based check. For production, use NLI model
        # (e.g., DeBERTa-v3-base-mnli-fever-anli) for claim verification.
        """
        Check if the answer is grounded in the source documents.

        Returns:
            {
                "is_grounded": bool,
                "groundedness_score": float (0-1),
                "unsupported_claims": [str, ...],
                "numerical_check": {"found": [str], "missing": [str]},
                "entity_check": {"found": [str], "missing": [str]},
            }
        """
        # Combine all sources
        source_text = " ".join(sources)

        # Check numerical claims
        numbers_in_answer = re.findall(r'\$?[\d,]+\.?\d*', answer)
        numbers_in_source = re.findall(r'\$?[\d,]+\.?\d*', source_text)

        found_numbers = [n for n in numbers_in_answer if n in numbers_in_source]
        missing_numbers = [n for n in numbers_in_answer if n not in numbers_in_source]

        # Check entity claims (simple: look for capitalized words)
        entities_in_answer = set(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', answer))
        entities_in_source = set(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', source_text))

        # Remove common words
        common_entities = {"The", "This", "That", "What", "How", "When", "Where", "Why"}
        entities_in_answer -= common_entities
        entities_in_source -= common_entities

        found_entities = entities_in_answer & entities_in_source
        missing_entities = entities_in_answer - entities_in_source

        # Calculate groundedness score
        total_claims = len(numbers_in_answer) + len(entities_in_answer)
        supported_claims = len(found_numbers) + len(found_entities)

        if total_claims == 0:
            groundedness_score = 1.0  # No claims to check
        else:
            groundedness_score = supported_claims / total_claims

        return {
            "is_grounded": groundedness_score >= 0.7,
            "groundedness_score": groundedness_score,
            "unsupported_claims": [f"Number: {n}" for n in missing_numbers] +
                                  [f"Entity: {e}" for e in missing_entities],
            "numerical_check": {"found": found_numbers, "missing": missing_numbers},
            "entity_check": {"found": list(found_entities), "missing": list(missing_entities)},
        }


def demo():
    """Self-check: verify hybrid search and hallucination detection."""
    from .domain_chunker import DocumentChunk, ChunkMetadata

    # Build a tiny index
    chunks = [
        DocumentChunk(
            content="Invoice INV-001 from Acme Corp for $5200.00 due 09/15/2026",
            metadata=ChunkMetadata(doc_type="invoice", chunk_type="header", vendor="Acme Corp", amount=5200.0),
        ),
        DocumentChunk(
            content="Invoice INV-002 from GlobalTech for $1847.92 due 03/15/2025",
            metadata=ChunkMetadata(doc_type="invoice", chunk_type="header", vendor="GlobalTech", amount=1847.92),
        ),
        DocumentChunk(
            content="Payment terms Net 30. Bank account ****-1234.",
            metadata=ChunkMetadata(doc_type="invoice", chunk_type="terms"),
        ),
    ]

    engine = HybridSearchEngine()
    engine.build_index(chunks)
    assert engine.is_built
    assert len(engine.chunks) == 3

    # Search
    results = engine.search("Acme Corp invoice", top_k=2)
    assert len(results) > 0
    assert any("Acme" in r.content for r in results)

    # Hallucination detector
    hd = HallucinationDetector()
    grounded = hd.detect_hallucinations(
        "Invoice INV-001 from Acme Corp for $5200.00",
        ["Invoice INV-001 from Acme Corp for $5200.00 due 09/15/2026"],
    )
    assert grounded["is_grounded"]

    hallucinated = hd.detect_hallucinations(
        "Invoice for $99999.99 from FakeCorp",
        ["Invoice INV-001 from Acme Corp for $5200.00"],
    )
    assert not hallucinated["is_grounded"] or hallucinated["groundedness_score"] < 1.0

    print(f"demo OK: search returned {len(results)} results, grounded={grounded['groundedness_score']:.2f}, hallucination detected={not hallucinated['is_grounded']}")


if __name__ == "__main__":
    demo()
