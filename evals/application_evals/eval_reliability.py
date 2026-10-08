"""
Operational eval: RELIABILITY

Reliability measures whether the RAG application can successfully
serve requests without failing.

We measure:
    - success rate
    - error rate
    - retry rate

Retries are important because a system may eventually succeed while
still being flaky on the first attempt.

There is no RagPipeline object in this project (see eval_latency.py), so the
adapter below runs the same production chain per request:
    get_retriever(question, document_id) -> retriever.invoke -> generate
Exactly like graph/streaming.py, just without DB/history/SSE.

Run from the project root (same as the other application evals):
    python -m evals.application_evals.eval_reliability
"""

# ============================================================
# 1. IMPORTS & ENV
# ============================================================
import os
import sys
import time

# Add backend to path so we can import src modules (same as other app evals)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "backend"))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", "backend", ".env"))

from src.generator import generate  # type: ignore
from src.rag.retriever import get_retriever  # type: ignore


# ============================================================
# 2. CONFIG
# ============================================================
# (question, document_id) tuples -- production retrieval is filtered by
# document_id, so every sample needs both (same set as eval_latency.py).
QUESTIONS = [
    ("What is Capgras' syndrome?", 21),
    ("How do L1 and L2 regularization differ in how they affect a neural network's weights?", 27),
    ("How does max pooling reduce the size of a feature map in a convolutional network?", 27),
    ("What is the concordance rate for schizophrenia in monozygotic twins?", 21),
    ("What biological and psychological theories have been proposed for the aetiology of schizophrenia?", 21),
]

REPEATS = 5

MAX_RETRIES = 2
BACKOFF_BASE_S = 0.5

# SLO: the offline pass/fail line for reliability
MIN_SUCCESS_RATE_PCT = 99.0


# ============================================================
# 3. PIPELINE ADAPTER
# ============================================================
# One production request, no retries -- retries are the eval's job.
def run_once(question: str, document_id: int) -> str:
    retriever = get_retriever(question, document_id)
    docs = retriever.invoke(question)
    context = [doc.page_content for doc in docs]
    return generate(question, context)


# ============================================================
# 4. RELIABILITY TRACKER
# ============================================================
class Reliability:

    def __init__(self):
        self.calls = 0
        self.successes = 0
        self.failures = 0
        self.retries = 0


# ============================================================
# 5. RETRY WRAPPER
# ============================================================
def call_with_retries(question, document_id, reliability):

    reliability.calls += 1

    for attempt in range(MAX_RETRIES + 1):

        try:
            run_once(question, document_id)

            reliability.successes += 1

            return True

        except Exception as e:

            if attempt < MAX_RETRIES:

                reliability.retries += 1

                # exponential backoff
                time.sleep(
                    BACKOFF_BASE_S * (2 ** attempt)
                )

            else:

                reliability.failures += 1

                print(
                    f"FAILED after {MAX_RETRIES} retries: {e}"
                )

                return False


# ============================================================
# 6. BENCHMARK
# ============================================================
def benchmark():

    reliability = Reliability()

    print("Measuring reliability...")

    for question, document_id in QUESTIONS:

        for _ in range(REPEATS):

            call_with_retries(question, document_id, reliability)

    return reliability


# ============================================================
# 7. REPORT
# ============================================================
def report(rel):

    success_rate = (
        100 * rel.successes / rel.calls
        if rel.calls else 0
    )

    error_rate = (
        100 * rel.failures / rel.calls
        if rel.calls else 0
    )

    retry_rate = (
        100 * rel.retries / rel.calls
        if rel.calls else 0
    )

    print("\n" + "=" * 60)
    print("RELIABILITY")
    print("=" * 60)

    print(f"total requests : {rel.calls}")
    print(f"successful     : {rel.successes}")
    print(f"failed         : {rel.failures}")

    print("-" * 60)

    print(f"success rate   : {success_rate:.2f}%")
    print(f"error rate     : {error_rate:.2f}%")
    print(f"retry rate     : {retry_rate:.2f}%")

    print("-" * 60)

    # --- budget verdict (the offline pass/fail) ---
    verdict = "PASS" if success_rate >= MIN_SUCCESS_RATE_PCT else "FAIL"
    print(
        f"SLO: success rate >= {MIN_SUCCESS_RATE_PCT:.1f}%  ->  "
        f"{success_rate:.2f}%   [{verdict}]"
    )
    print("=" * 60)
    print("note: retry rate > 0 means the first attempt was flaky even if")
    print("success rate is 100% after retries -- watch both numbers.")


# ============================================================
# 8. ENTRYPOINT
# ============================================================
def main():

    reliability = benchmark()

    report(reliability)


if __name__ == "__main__":
    main()
