"""
evals/application_evals/eval_correctness.py
===========================================
Application-level evaluation — FULL chain, exactly like production:

    question -> get_retriever (Qdrant, filtered by document_id) -> chunks
             -> generate -> answer

Then the G-Eval CORRECTNESS metric compares the LIVE answer against the
golden ideal_answer (partial credit, not pass/fail).

    python -m evals.application_evals.eval_correctness
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


# 1. LOAD questions + ideal answers (ideal_answer is the CORRECT answer, our reference)
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
            expected_output=g["ideal_answer"],  # the CORRECT reference answer
            retrieval_context=context,          # what the RETRIEVER returned
        )
    )


# 3. THE CORRECTNESS METRIC (graded G-Eval – partial credit, not pass/fail)
correctness = GEval(
    name="Correctness",
    evaluation_steps=[
        "Compare only the factual claims in the actual output against the expected output.",
        "A claim is wrong only if it CONTRADICTS the expected output or is factually false. Judge truth, not completeness.",
        "A factually accurate answer must score at least 0.9 even if it is shorter, less detailed, or covers fewer points than the expected output.",
        "Do NOT deduct for brevity, missing elaboration, fewer examples, or omitted points – omissions are not errors here.",
        "Additional correct information must NEVER lower the score.",
        "Reserve low scores for answers that state something contradictory or factually incorrect.",
    ],
    rubric=[
        Rubric(
            score_range=(9, 10),
            expected_outcome="All stated claims are factually correct and consistent with the expected output. No contradictions. Brevity is fine.",
        ),
        Rubric(
            score_range=(5, 8),
            expected_outcome="Mostly correct but contains one minor inaccuracy or slightly imprecise claim.",
        ),
        Rubric(
            score_range=(0, 4),
            expected_outcome="Contains a clear factual error or a claim that contradicts the expected output.",
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
    async_mode=False
)

# 4. EVALUATE
evaluate(
    test_cases=test_cases,
    metrics=[correctness],
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
