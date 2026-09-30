"""
evals/application_evals/eval_style.py
======================================
Application-level evaluation — FULL chain, exactly like production
(same pipeline as eval_correctness.py / eval_completeness.py):

    question -> get_retriever (Qdrant, filtered by document_id) -> chunks
             -> generate -> answer

Then the G-Eval STYLE metric judges TONE only — reference-free,
it never looks at the golden ideal_answer.

    python -m evals.application_evals.eval_style
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


# 1. LOAD questions (only the question matters for style; ideal_answer is unused by the metric)
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
            actual_output=answer,           # what the generator produced
            retrieval_context=context,      # what the RETRIEVER returned
        )
    )


# 3. STYLE – reference-free, judges TONE only (no EXPECTED_OUTPUT in params)
style = GEval(
    name="Style",
    evaluation_steps=[
        "Judge only the teaching style and tone of the actual output, not whether it is factually correct or complete.",
        "Reward an intuitive, explanatory tone: plain language, the idea explained before any formula or jargon, and technical terms briefly unpacked when used.",
        "Reward a direct, conversational register written in prose,explain it out loud, rather than a dry, formal, or bullet-list tone.",
        "An analogy or concrete example is a BONUS when the concept is abstract, but a clear, direct, well-explained answer is fully acceptable and must NOT be penalized for not having one.",
        "Penalize answers that are stiff, bureaucratic, structured as a bare list with no explanation, or that use unexplained jargon.",
        "Do NOT reward or penalize based on correctness, completeness, or length – only on style and tone.",
    ],
    rubric=[
        Rubric(
            score_range=(9, 10),
            expected_outcome="Clearly in a CampusX teaching voice: intuitive, conversational prose that explains before it formalizes.",
        ),
        Rubric(
            score_range=(7, 8),
            expected_outcome="Clear, conversational, and well-explained in prose. Fully acceptable even without an analogy or example.",
        ),
        Rubric(
            score_range=(4, 6),
            expected_outcome="Understandable but somewhat flat, formal, or list-heavy in places.",
        ),
        Rubric(
            score_range=(0, 3),
            expected_outcome="Dry, stiff, bare-list, jargon-heavy, or robotic; does not read like a teaching explanation.",
        ),
    ],
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
    ],
    threshold=THRESHOLD,
    model=JUDGE_MODEL,
    strict_mode=False,  # graded scale; strict_mode=True would collapse it to 0/1
    async_mode=False,
)

# 4. EVALUATE
evaluate(
    test_cases=test_cases,
    metrics=[style],
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
