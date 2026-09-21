"""
evals/eval_generator.py
=======================
Component-level evaluation of the GENERATOR, in isolation.

Faithfulness: of the claims in the generated answer, how many are supported
by the context it was given? (Did the generator make things up?)

ISOLATION: we feed the generator the GOLDEN context (the known-good chunks
from the faithfulness dataset), NOT the retriever's output. So a low score
is purely the generator's fault — the context was already correct.

    python -m evals.eval_generator
"""

import os
import json
import sys

# Add backend to path so we can import src modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "backend"))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", "backend", ".env"))

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.models.llms.openai_model import OpenAIModel
from deepeval.evaluate.configs import CacheConfig, ErrorConfig
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric

from src.generator import generate   # type: ignore

GOLDEN_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "eval_golden_datasets", "faithfulness_dataset.json")
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



with open(GOLDEN_PATH, encoding="utf-8") as f:
    goldens = json.load(f)



test_cases = []
for g in goldens[:5]:
    context = g["ideal_context"]              
    answer = generate(g["query"], context)   

    test_cases.append(
        LLMTestCase(
            input=g["query"],
            actual_output=answer,             
            retrieval_context=context,       
            
        )
    )



metrics = [FaithfulnessMetric(
    threshold=THRESHOLD,
    model=JUDGE_MODEL,
    include_reason=False, 
),
AnswerRelevancyMetric(
    threshold=THRESHOLD, 
    model=JUDGE_MODEL, 
    include_reason=False)
]



evaluate(
    test_cases=test_cases,
    metrics=metrics,
    cache_config=CacheConfig(write_cache=False, use_cache=False),
    error_config=ErrorConfig(ignore_errors=True),
    hyperparameters={
        "retriever": "base_k5",
        "embedding_model": "mistral-embed",
        "chunk_size": 1000,
        "chunk_overlap": 150,
        "top_k": 5,
        "judge_model": JUDGE_MODEL_NAME,
        "golden_set": GOLDEN_PATH,
    },
)