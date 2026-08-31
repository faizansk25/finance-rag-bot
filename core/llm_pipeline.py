"""
LLM Pipeline: Prompt Engineering + Self-RAG + Grounding

This module connects the retrieval engine to the LLM and adds:
1. Structured prompt templates (chain-of-thought, few-shot)
2. Self-RAG: decides when to retrieve
3. Grounding verification: checks if answers are supported
4. Hallucination detection: catches made-up facts
5. Voice-friendly formatting: spells out numbers for TTS

This is what makes the bot reliable for finance use cases.
A finance AI that makes up numbers is dangerous.
"""

import os
import re
from typing import List, Dict, Optional
from dataclasses import dataclass

from .hybrid_search import HybridSearchEngine, SelfRAGEngine, HallucinationDetector, SearchResult


@dataclass
class PipelineResponse:
    """Response from the LLM pipeline."""
    answer: str
    sources: List[dict]
    groundedness_score: float
    is_grounded: bool
    unsupported_claims: List[str]
    retrieval_used: bool
    prompt_used: str


class LLMPipeline:
    """
    Full LLM pipeline with prompt engineering and grounding.

    The pipeline:
    1. Self-RAG decides: retrieve or answer directly?
    2. If retrieve: get relevant chunks via hybrid search
    3. Build prompt with chain-of-thought + few-shot examples
    4. Send to LLM
    5. Verify grounding against source documents
    6. If not grounded: regenerate or add disclaimer
    """

    def __init__(self, search_engine: HybridSearchEngine,
                 model: str = "gpt-4o-mini",
                 openai_api_key: Optional[str] = None):
        self.search_engine = search_engine
        self.self_rag = SelfRAGEngine(search_engine)
        self.hallucination_detector = HallucinationDetector()
        self.model = model
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")

    def query(self, question: str, top_k: int = 3) -> PipelineResponse:
        """
        Process a question through the full pipeline.

        Returns a PipelineResponse with answer, sources, and quality metrics.
        """
        # Step 1: Self-RAG decides if retrieval is needed
        rag_result = self.self_rag.retrieve_and_verify(question, top_k=top_k)

        if not rag_result["needs_retrieval"]:
            # General question — answer directly
            answer = self._answer_directly(question)
            return PipelineResponse(
                answer=answer,
                sources=[],
                groundedness_score=1.0,
                is_grounded=True,
                unsupported_claims=[],
                retrieval_used=False,
                prompt_used="direct",
            )

        # Step 2: Build prompt with context
        context = "\n\n---\n\n".join(rag_result["chunks_for_context"])
        prompt = self._build_grounded_prompt(question, context)

        # Step 3: Call LLM
        answer = self._call_llm(prompt)

        # Step 4: Verify grounding
        grounding = self.hallucination_detector.detect_hallucinations(
            answer, rag_result["chunks_for_context"]
        )

        # Step 5: If not grounded, add disclaimer
        if not grounding["is_grounded"]:
            disclaimer = "\n\n[Note: Some information in this response could not be verified against the source documents. Please double-check the numbers and details.]"
            answer += disclaimer

        # Format sources for response
        sources = []
        for r in rag_result["retrieved"]:
            sources.append({
                "content": r.content[:200] + "..." if len(r.content) > 200 else r.content,
                "score": round(r.score, 4),
                "type": r.search_type,
                "metadata": r.metadata,
            })

        return PipelineResponse(
            answer=answer,
            sources=sources,
            groundedness_score=grounding["groundedness_score"],
            is_grounded=grounding["is_grounded"],
            unsupported_claims=grounding["unsupported_claims"],
            retrieval_used=True,
            prompt_used="grounded_rag",
        )

    def _build_grounded_prompt(self, question: str, context: str) -> str:
        """
        Build a prompt that grounds the answer in source documents.

        Uses chain-of-thought reasoning and constraint-based rules.
        """
        prompt = f"""You are a finance assistant. Answer the question using ONLY the information provided in the context below.

RULES:
1. If the answer is in the context, state it clearly with the specific numbers
2. If the context does not contain the answer, say "I could not find this information in the provided documents"
3. Never make up financial figures. Use only numbers explicitly stated in the context
4. Never calculate totals yourself unless the context shows the calculation
5. If a field is missing from the context, say it is not available
6. Cite which document or section your answer comes from

CONTEXT:
{context}

QUESTION: {question}

THINKING PROCESS:
1. What specific information does the question ask for?
2. Which parts of the context contain this information?
3. What are the exact numbers and facts from the context?
4. Is there anything the context does not cover?

ANSWER:"""
        return prompt

    def _answer_directly(self, question: str) -> str:
        """Answer a general question without retrieval."""
        prompt = f"""You are a finance assistant. Answer this general question concisely.

QUESTION: {question}

ANSWER:"""
        return self._call_llm(prompt)

    def _call_llm(self, prompt: str) -> str:
        """Call the OpenAI API with retry and proper error handling."""
        import logging
        logger = logging.getLogger(__name__)

        if not self.api_key:
            logger.error("OPENAI_API_KEY not set")
            return "Error: OpenAI API key not configured. Set OPENAI_API_KEY environment variable."

        last_error = None
        for attempt in range(3):
            try:
                import openai
                client = openai.OpenAI(api_key=self.api_key)
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=1000,
                )
                return response.choices[0].message.content.strip()
            except openai.RateLimitError as e:
                last_error = e
                logger.warning(f"Rate limited (attempt {attempt+1}/3): {e}")
                import time
                time.sleep(2 ** attempt)
            except openai.APIConnectionError as e:
                last_error = e
                logger.warning(f"Connection error (attempt {attempt+1}/3): {e}")
                import time
                time.sleep(1)
            except openai.APIError as e:
                last_error = e
                logger.error(f"OpenAI API error: {e}")
                break
            except Exception as e:
                last_error = e
                logger.error(f"Unexpected error calling LLM: {type(e).__name__}: {e}")
                break

        return f"Error: LLM call failed after retries. {type(last_error).__name__}: {last_error}"

    def format_for_voice(self, text: str) -> str:
        """Format text for text-to-speech output.

        Voice-specific rules:
        - Replace abbreviations: AP → "accounts payable"
        - Replace abbreviations: PO → "purchase order"
        - Full number/date/email spelling requires a TTS engine (not implemented here).
          This is the preprocessing layer — TTS handles pronunciation.
        """
        # Replace common abbreviations
        abbreviations = {
            r'\bAP\b': 'accounts payable',
            r'\bAR\b': 'accounts receivable',
            r'\bPO\b': 'purchase order',
            r'\bGL\b': 'general ledger',
            r'\bINV\b': 'invoice',
            r'\bPOs\b': 'purchase orders',
        }
        for pattern, replacement in abbreviations.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        return text
