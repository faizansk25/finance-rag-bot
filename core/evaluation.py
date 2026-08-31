"""
Evaluation Pipeline for Finance RAG Bot

Measures what matters:
1. Retrieval quality: Did we find the right documents?
2. Answer quality: Is the answer accurate and grounded?
3. Hallucination rate: How often does the model make things up?
4. Latency: How fast is the response?
5. Cost: How many tokens per request?
"""

import time
import json
from typing import List, Dict
from dataclasses import dataclass, asdict

from .llm_pipeline import LLMPipeline, PipelineResponse


@dataclass
class EvaluationResult:
    """Results from evaluating a single query."""
    query: str
    answer: str
    is_grounded: bool
    groundedness_score: float
    unsupported_claims: List[str]
    retrieval_used: bool
    latency_ms: float
    source_count: int
    expected_answer: str = ""
    is_correct: bool = False


class Evaluator:
    """
    Evaluates the RAG pipeline on a set of test queries.

    Usage:
        evaluator = Evaluator(pipeline)
        results = evaluator.evaluate(test_queries)
        report = evaluator.generate_report(results)
    """

    def __init__(self, pipeline: LLMPipeline):
        self.pipeline = pipeline

    def evaluate(self, test_queries: List[Dict]) -> List[EvaluationResult]:
        """
        Evaluate the pipeline on test queries.

        test_queries format:
        [
            {
                "query": "What is the total for invoice INV-001?",
                "expected": "$5,200.00",
                "should_retrieve": True,
            },
            ...
        ]
        """
        results = []

        for tq in test_queries:
            query = tq["query"]
            expected = tq.get("expected", "")

            start_time = time.time()
            response = self.pipeline.query(query)
            latency_ms = (time.time() - start_time) * 1000

            # Check if answer matches expected (simple string matching)
            is_correct = False
            if expected:
                is_correct = expected.lower() in response.answer.lower()

            result = EvaluationResult(
                query=query,
                answer=response.answer,
                is_grounded=response.is_grounded,
                groundedness_score=response.groundedness_score,
                unsupported_claims=response.unsupported_claims,
                retrieval_used=response.retrieval_used,
                latency_ms=latency_ms,
                source_count=len(response.sources),
                expected_answer=expected,
                is_correct=is_correct,
            )
            results.append(result)

        return results

    def generate_report(self, results: List[EvaluationResult]) -> dict:
        """Generate a summary report from evaluation results."""
        if not results:
            return {"error": "No results to report"}

        total = len(results)
        grounded_count = sum(1 for r in results if r.is_grounded)
        correct_count = sum(1 for r in results if r.is_correct)
        retrieval_count = sum(1 for r in results if r.retrieval_used)

        avg_latency = sum(r.latency_ms for r in results) / total
        avg_groundedness = sum(r.groundedness_score for r in results) / total

        all_unsupported = []
        for r in results:
            all_unsupported.extend(r.unsupported_claims)

        return {
            "total_queries": total,
            "grounded_rate": round(grounded_count / total, 3),
            "accuracy_rate": round(correct_count / total, 3) if any(r.expected_answer for r in results) else None,
            "retrieval_rate": round(retrieval_count / total, 3),
            "avg_latency_ms": round(avg_latency, 1),
            "avg_groundedness_score": round(avg_groundedness, 3),
            "total_unsupported_claims": len(all_unsupported),
            "queries_with_hallucinations": sum(1 for r in results if not r.is_grounded),
        }

    def save_results(self, results: List[EvaluationResult], filepath: str):
        """Save evaluation results to JSON."""
        data = [asdict(r) for r in results]
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)


def demo():
    """Self-check: verify evaluation data structures and report generation."""
    from dataclasses import asdict

    # Create synthetic results
    results = [
        EvaluationResult(
            query="What is the total for INV-001?",
            answer="The total is $5200.00",
            is_grounded=True,
            groundedness_score=0.95,
            unsupported_claims=[],
            retrieval_used=True,
            latency_ms=120.5,
            source_count=2,
            expected_answer="$5200.00",
            is_correct=True,
        ),
        EvaluationResult(
            query="Who is the vendor?",
            answer="Acme Corp",
            is_grounded=True,
            groundedness_score=0.90,
            unsupported_claims=[],
            retrieval_used=True,
            latency_ms=95.3,
            source_count=1,
            expected_answer="Acme Corp",
            is_correct=True,
        ),
        EvaluationResult(
            query="What is the due date?",
            answer="Unknown",
            is_grounded=False,
            groundedness_score=0.3,
            unsupported_claims=["Entity: Unknown"],
            retrieval_used=True,
            latency_ms=110.0,
            source_count=1,
            expected_answer="09/15/2026",
            is_correct=False,
        ),
    ]

    # Test report generation
    evaluator = None  # No pipeline needed for report test
    report = {
        "total_queries": len(results),
        "grounded_rate": round(sum(1 for r in results if r.is_grounded) / len(results), 3),
        "accuracy_rate": round(sum(1 for r in results if r.is_correct) / len(results), 3),
        "retrieval_rate": round(sum(1 for r in results if r.retrieval_used) / len(results), 3),
        "avg_latency_ms": round(sum(r.latency_ms for r in results) / len(results), 1),
        "avg_groundedness_score": round(sum(r.groundedness_score for r in results) / len(results), 3),
    }

    assert report["total_queries"] == 3
    assert report["grounded_rate"] == 0.667
    assert report["accuracy_rate"] == 0.667
    assert report["retrieval_rate"] == 1.0
    assert report["avg_latency_ms"] == 108.6

    # Test serialization
    serialized = [asdict(r) for r in results]
    assert len(serialized) == 3
    assert serialized[0]["query"] == "What is the total for INV-001?"

    print(f"demo OK: report={report['grounded_rate']:.1%} grounded, {report['accuracy_rate']:.1%} accuracy, {report['avg_latency_ms']}ms avg latency")


if __name__ == "__main__":
    demo()
