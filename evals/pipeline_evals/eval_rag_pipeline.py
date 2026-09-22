"""
evals/pipeline_evals/eval_rag_pipeline.py
=========================================
Pipeline-level evaluation — FULL chain, exactly like production:

    query -> get_retriever (Qdrant) -> chunks -> generate -> answer

NO isolation (unlike component evals). Bad retrieval -> bad context ->
bad answer, and the metrics reflect it. This measures real user-facing quality.

    python -m evals.pipeline_evals.eval_rag_pipeline
"""

import json
import os
import sys

# Add backend to path so we can import src modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "backend"))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", "backend", ".env"))

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

GOLDEN_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "eval_golden_datasets", "faithfulness_dataset.json")
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



with open(GOLDEN_PATH, encoding="utf-8") as f:
    goldens = json.load(f)


# 2. RUN THE FULL PIPELINE per query — retrieve REAL chunks, then generate.
test_cases = []
for g in goldens[:5]:
    # RETRIEVE — same call the production graph makes (filtered by document)
    retriever = get_retriever(g["query"], g["document_id"])
    retrieved = retriever.invoke(g["query"])
    context = [doc.page_content for doc in retrieved]

    # GENERATE — answer grounded in whatever the retriever actually returned
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
evaluate(
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
