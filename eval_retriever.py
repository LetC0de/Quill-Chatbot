import json
import os
import sys

# Quill ka backend `backend/` me hai — sys.path pehle add karo
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))

# .env sabse pehle load karo — warna Settings() ko keys nahi milenge
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "backend", ".env"))
load_dotenv()  # root .env fallback

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRecallMetric, ContextualPrecisionMetric
from deepeval.models.llms.openai_model import OpenAIModel
from deepeval.evaluate.configs import CacheConfig, ErrorConfig

# Quill ka actual retriever (document_id filtering wala) — .env ke BAAD import
from src.rag.retriever import get_retriever

GOLDEN_PATH = "golden_dataset.json"
JUDGE_MODEL_NAME = "nvidia/nemotron-3.5-lightning:free"
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


# 1. LOAD the golden set
with open(GOLDEN_PATH, encoding="utf-8") as f:
    goldens = json.load(f)


# 2. RUN THE RETRIEVER — Quill me get_retriever(query, document_id) call hota hai
test_cases = []

for g in goldens:
    retriever = get_retriever(g["query"], g["document_id"])
    retrieved = retriever.invoke(g["query"])
    retrieval_context = [doc.page_content for doc in retrieved]

    test_cases.append(
        LLMTestCase(
            input=g["query"],
            expected_output=g["ideal_answer"],
            retrieval_context=retrieval_context,
            actual_output="(generator not evaluated in this run)",
        )
    )


# 3. THE METRICS — async_mode=False => ek ke baad ek, rate-limit nahi
metrics = [
    ContextualRecallMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True, async_mode=False),
    ContextualPrecisionMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True, async_mode=False),
]


# 4. EVALUATE
evaluate(
    test_cases=test_cases,
    metrics=metrics,
    cache_config=CacheConfig(write_cache=False, use_cache=False),
    error_config=ErrorConfig(ignore_errors=True),
    hyperparameters={
        "retriever": "quill_mmr_k4_fetch10 / similarity_k6 (get_retriever)",
        "embedding_model": "mistral-embed",
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "top_k": "4 (MMR) / 6 (summary)",
        "judge_model": JUDGE_MODEL_NAME,
        "golden_set": GOLDEN_PATH,
    },
)
