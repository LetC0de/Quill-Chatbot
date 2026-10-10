# eval_application.py
"""
Application-level evaluation -- FULL chain, exactly like production:

    question -> get_retriever (Qdrant, filtered by document_id) -> chunks
             -> generate -> answer

Bundles the three application-level GEval metrics (Correctness, Completeness,
Style) that evals/ keeps in separate files -- the pipeline runs ONCE per query
and all three metrics judge the same test case.

    python -m regression_test.eval_application
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "backend", ".env"))
load_dotenv()

from deepeval import evaluate
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.models.llms.openai_model import OpenAIModel
from deepeval.evaluate.configs import AsyncConfig, CacheConfig, ErrorConfig
from deepeval.metrics import GEval
from deepeval.metrics.g_eval import Rubric

from src.rag.retriever import get_retriever  # type: ignore
from src.generator import generate  # type: ignore
from regression_test.harness import load_goldens, summarize_by_metric, print_summary

GOLDEN_PATH = os.path.join(
    os.path.dirname(__file__), "..", "eval_golden_datasets", "correctness_dataset.json"
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
    # below, exactly like evals/application_evals/*.py.
    del _rag

    # 1. LOAD queries + ideal answers
    goldens = load_goldens(GOLDEN_PATH)

    # 2. RUN THE FULL PIPELINE per query -- retrieve REAL chunks, then generate.
    #    ([:7] same slice as evals/application_evals/eval_correctness.py et al.)
    test_cases = []
    for g in goldens[:7]:
        # RETRIEVE -- same call the production graph makes (filtered by document)
        retriever = get_retriever(g["question"], g["document_id"])
        retrieved = retriever.invoke(g["question"])
        context = [doc.page_content for doc in retrieved]

        # GENERATE -- answer grounded in whatever the retriever actually returned
        answer = generate(g["question"], context)

        test_cases.append(
            LLMTestCase(
                input=g["question"],
                actual_output=answer,               # what the generator produced
                expected_output=g["ideal_answer"],  # the CORRECT reference answer
                retrieval_context=context,          # what the RETRIEVER returned
            )
        )

    # 3. THREE APPLICATION-LEVEL QUALITY METRICS (evaluation_steps + rubric
    #    kept exactly as the original regression file had them)

    # 3a. CORRECTNESS --- reference-based, judges TRUTH (not coverage or length)
    correctness = GEval(
        name="Correctness",
        evaluation_steps=[
            "Compare only the factual claims in the actual output against the expected output.",
            "A claim is wrong only if it CONTRADICTS the expected output or is factually false. Judge truth, not completeness.",
            "A factually accurate answer must score at least 0.9 even if it is shorter or covers fewer points than the expected output.",
            "Do NOT deduct for brevity, missing elaboration, or omitted points --- omissions are not errors here.",
            "Additional correct information must NEVER lower the score.",
        ],
        rubric=[
            Rubric(score_range=(9, 10), expected_outcome="All stated claims are factually correct and consistent. No contradictions. Brevity is fine."),
            Rubric(score_range=(5, 8),  expected_outcome="Mostly correct but one minor inaccuracy."),
            Rubric(score_range=(0, 4),  expected_outcome="Contains a clear factual error or a claim that contradicts the expected output."),
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

    # 3b. COMPLETENESS --- reference-based, judges COVERAGE (not correctness)
    completeness = GEval(
        name="Completeness",
        evaluation_steps=[
            "Identify the key points contained in the expected output.",
            "Check how many of those key points are addressed in the actual output.",
            "Penalize the actual output for each key point from the expected output that it omits or only partially covers.",
            "Judge coverage only. Do NOT lower the score because a covered point is stated incorrectly --- factual correctness is judged separately.",
            "Do NOT penalize the actual output for adding extra information beyond the expected output.",
        ],
        rubric=[
            Rubric(score_range=(9, 10), expected_outcome="Addresses essentially all key points in the expected output."),
            Rubric(score_range=(5, 8),  expected_outcome="Covers the main key points but misses one or more."),
            Rubric(score_range=(0, 4),  expected_outcome="Misses several key points; only partially covers the expected output."),
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

    # 3c. STYLE --- reference-free, judges TONE only (no EXPECTED_OUTPUT param)
    style = GEval(
        name="Style",
        evaluation_steps=[
            "Judge only the teaching style and tone of the actual output, not whether it is factually correct or complete.",
            "Reward an intuitive, explanatory tone: plain language, the idea explained before any formula or jargon, and technical terms briefly unpacked when used.",
            "Reward a direct, conversational register written in prose, as a CampusX lecture would explain it out loud, rather than a dry, formal, or bullet-list tone.",
            "An analogy or concrete example is a BONUS when the concept is abstract, but a clear, direct, well-explained answer is fully acceptable and must NOT be penalized for not having one.",
            "Penalize answers that are stiff, bureaucratic, structured as a bare list with no explanation, or that use unexplained jargon.",
            "Do NOT reward or penalize based on correctness, completeness, or length --- only on style and tone.",
        ],
        rubric=[
            Rubric(score_range=(9, 10), expected_outcome="Clearly in a CampusX teaching voice: intuitive, conversational prose that explains before it formalizes."),
            Rubric(score_range=(7, 8),  expected_outcome="Clear, conversational, and well-explained in prose. Fully acceptable even without an analogy or example."),
            Rubric(score_range=(4, 6),  expected_outcome="Understandable but somewhat flat, formal, or list-heavy in places."),
            Rubric(score_range=(0, 3),  expected_outcome="Dry, stiff, bare-list, jargon-heavy, or robotic; does not read like a teaching explanation."),
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

    # 4. EVALUATE --- all three together (same test_cases, one pipeline pass)
    result = evaluate(
        test_cases=test_cases,
        metrics=[correctness, completeness, style],
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
    return summarize_by_metric(result)


def run_local():
    """Standalone convenience: run the full chain."""
    return run()


if __name__ == "__main__":
    print_summary("application", run_local())
