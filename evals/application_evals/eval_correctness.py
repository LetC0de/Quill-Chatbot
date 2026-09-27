import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

from deepeval import evaluate
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.models.llms.openai_model import OpenAIModel
from deepeval.evaluate.configs import CacheConfig, ErrorConfig
from deepeval.metrics import GEval
from deepeval.metrics.g_eval import Rubric

# repo root on sys.path so `src` works whether run from root or from evals/
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.rag_pipeline import RagPipeline

load_dotenv()

GOLDEN_PATH = str(ROOT_DIR / "goldens" / "correctness_goldens.json")  # question + ideal_answer
JUDGE_MODEL_NAME = "nvidia/nemotron-3-super-120b-a12b:free"    # nvidia/nemotron-3-super-120b-a12b:free
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


# 1. LOAD queries + ideal answers (ideal_answer is the CORRECT answer, our reference)
with open(GOLDEN_PATH) as f:
    goldens = json.load(f)


# 2. RUN THE FULL PIPELINE per query, build a test case from LIVE output
rag = RagPipeline()
test_cases = []

for g in goldens[:1]:
    result = rag.invoke(g["question"])          # retrieve → rerank → generate

    test_cases.append(
        LLMTestCase(
            input=g["question"],
            actual_output=result["answer"],      # what the generator produced
            expected_output=g["ideal_answer"],   # the CORRECT reference answer
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
)

# 4. EVALUATE
evaluate(
    test_cases=test_cases,
    metrics=[correctness],
    cache_config=CacheConfig(write_cache=False, use_cache=False),
    error_config=ErrorConfig(ignore_errors=True),
    hyperparameters={
        "retriever": "rerank_fetch10_k5",
        "embedding_model": "mistral-embed",
        "chunk_size": 1000,
        "chunk_overlap": 150,
        "top_k": 5,
        "judge_model": JUDGE_MODEL_NAME,
        "golden_set": GOLDEN_PATH,
    },
)