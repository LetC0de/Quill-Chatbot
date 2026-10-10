# eval_rag_pipeline.py
"""
Pipeline-level evaluation -- FULL chain, exactly like production:

    query -> get_retriever (Qdrant) -> chunks -> generate -> answer

NO isolation (unlike component evals). Bad retrieval -> bad context ->
bad answer, and the metrics reflect it. This measures real user-facing quality.

    python -m regression_test.eval_rag_pipeline
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "backend", ".env"))
load_dotenv()

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
)
from deepeval.models.llms.openai_model import OpenAIModel
from deepeval.evaluate.configs import CacheConfig, ErrorConfig

from src.rag.retriever import get_retriever  # type: ignore
from src.generator import generate  # type: ignore
from regression_test.harness import load_goldens, summarize_by_metric, print_summary

GOLDEN_PATH = os.path.join(
    os.path.dirname(__file__), "..", "eval_golden_datasets", "faithfulness_dataset.json"
)
JUDGE_MODEL_NAME = "nvidia/nemotron-3-super-120b-a12b:free"
JUDGE_MODEL = OpenAIModel(
    model=JUDGE_MODEL_NAME,
    api_key=os.getenv("API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
    generation_kwargs={
        "extra_body": {"reasoning": {"enabled": False}},
    },
)
JUDGE_MODEL.model_data.supports_json = True
JUDGE_MODEL.model_data.supports_structured_outputs = False

THRESHOLD = 0.7


def run(_rag=None):
    # NOTE: `_rag` is kept (positionally compatible) for run_suite's
    # injected-pipeline contract, but this project has no single pipeline
    # object -- the chain (get_retriever -> generate) is run inline per query
    # below, exactly like evals/pipeline_evals/eval_rag_pipeline.py.
    del _rag

    # 1. LOAD queries (we only need the queries --- context comes from the pipeline now)
    goldens = load_goldens(GOLDEN_PATH)

    # 2. RUN THE FULL PIPELINE per query -- retrieve REAL chunks, then generate.
    #    ([:5] same slice as evals/pipeline_evals/eval_rag_pipeline.py)
    test_cases = []
    for g in goldens[:5]:
        # RETRIEVE -- same call the production graph makes (filtered by document)
        retriever = get_retriever(g["query"], g["document_id"])
        retrieved = retriever.invoke(g["query"])
        context = [doc.page_content for doc in retrieved]

        # GENERATE -- answer grounded in whatever the retriever actually returned
        answer = generate(g["query"], context)

        test_cases.append(
            LLMTestCase(
                input=g["query"],
                actual_output=answer,            # what the generator produced
                retrieval_context=context,       # what the RETRIEVER returned
            )
        )

    # 3. THE THREE TRIAD METRICS
    metrics = [
        ContextualRelevancyMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=False),
        FaithfulnessMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=False),
        AnswerRelevancyMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=False),
    ]

    # 4. EVALUATE
    result = evaluate(
        test_cases=test_cases,
        metrics=metrics,
        cache_config=CacheConfig(write_cache=False, use_cache=False),
        error_config=ErrorConfig(ignore_errors=True),
        hyperparameters={
            "mode": "pipeline (retrieve -> generate, no isolation)",
            "retriever": "get_retriever (mmr k4 fetch10 / similarity k6)",
            "embedding_model": "mistral-embed",
            "chunk_size": 1000,
            "chunk_overlap": 150,
            "top_k": "4 (MMR) / 6 (summary)",
            "judge_model": JUDGE_MODEL_NAME,
            "golden_set": GOLDEN_PATH,
        },
    )
    return summarize_by_metric(result)


def run_local():
    """Standalone convenience: run the full chain."""
    return run()


if __name__ == "__main__":
    print_summary("rag_pipeline", run_local())
