"""
evals/application_evals/eval_completeness.py
============================================
Application-level evaluation — FULL chain, exactly like production
(same pipeline as eval_correctness.py):

    question -> get_retriever (Qdrant, filtered by document_id) -> chunks
             -> generate -> answer

Then the G-Eval COMPLETENESS metric checks COVERAGE of the golden
ideal_answer (partial credit, not pass/fail).

    python -m evals.application_evals.eval_completeness
"""

import os
import sys
import json

# Add backend to path so we can import src modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "backend"))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", "backend", ".env"))

from deepeval import evaluate
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.models.llms.openai_model import OpenAIModel
from deepeval.evaluate.configs import AsyncConfig, CacheConfig, ErrorConfig
from deepeval.metrics import GEval
from deepeval.metrics.g_eval import Rubric

from src.rag.retriever import get_retriever  # type: ignore
from src.generator import generate  # type: ignore

GOLDEN_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "eval_golden_datasets", "correctness_dataset.json")
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


# 1. LOAD questions + ideal answers (ideal_answer = the key points we check coverage against)
with open(GOLDEN_PATH, encoding="utf-8") as f:
    goldens = json.load(f)


# 2. RUN THE FULL PIPELINE per query — retrieve REAL chunks, then generate.
test_cases = []
for g in goldens[:7]:
    # RETRIEVE — same call the production graph makes (filtered by document)
    retriever = get_retriever(g["question"], g["document_id"])
    retrieved = retriever.invoke(g["question"])
    context = [doc.page_content for doc in retrieved]

    # GENERATE — answer grounded in whatever the retriever actually returned
    answer = generate(g["question"], context)

    test_cases.append(
        LLMTestCase(
            input=g["question"],
            actual_output=answer,               # what the generator produced
            expected_output=g["ideal_answer"],  # reference for coverage check
            retrieval_context=context,          # what the RETRIEVER returned
        )
    )


# 3. THE COMPLETENESS METRIC (graded G-Eval – partial credit, not pass/fail)
completeness = GEval(
    name="Completeness",
    evaluation_steps=[
        "Identify the key points contained in the expected output.",
        "Check how many of those key points are addressed in the actual output.",
        "Penalize the actual output for each key point from the expected output that it omits or only partially covers.",
        "Judge coverage only. Do NOT lower the score because a covered point is stated incorrectly – factual correctness is judged separately.",
        "Do NOT penalize the actual output for adding extra information beyond the expected output.",
    ],
    rubric=[
        Rubric(
            score_range=(9, 10),
            expected_outcome="Addresses essentially all key points in the expected output.",
        ),
        Rubric(
            score_range=(5, 8),
            expected_outcome="Covers the main key points but misses one or more.",
        ),
        Rubric(
            score_range=(0, 4),
            expected_outcome="Misses several key points, or only partially covers the expected output.",
        ),
    ],
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
    ],
    threshold=THRESHOLD,
    model=JUDGE_MODEL,
    strict_mode=False,  # graded scale; strict_mode=True would collapse it to 0/1
    async_mode=False,
)


# 4. EVALUATE
evaluate(
    test_cases=test_cases,
    metrics=[completeness],
    async_config=AsyncConfig(run_async=False),
    cache_config=CacheConfig(write_cache=False, use_cache=False),
    error_config=ErrorConfig(ignore_errors=True),
    hyperparameters={
        "mode": "application (retrieve -> generate, full pipeline)",
        "retriever": "get_retriever (mmr k4 fetch10 / similarity k6)",
        "embedding_model": "mistral-embed",
        "chunk_size": 1000,
        "chunk_overlap": 150,
        "top_k": "4 (MMR) / 6 (summary)",
        "judge_model": JUDGE_MODEL_NAME,
        "golden_set": GOLDEN_PATH,
    },
)
