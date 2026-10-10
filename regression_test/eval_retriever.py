# eval_retriever.py
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "backend", ".env"))
load_dotenv()

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRecallMetric, ContextualPrecisionMetric
from deepeval.models.llms.openai_model import OpenAIModel
from deepeval.evaluate.configs import CacheConfig, ErrorConfig

from src.rag.retriever import get_retriever  # type: ignore
from regression_test.harness import load_goldens, summarize_by_metric, print_summary

GOLDEN_PATH = os.path.join(
    os.path.dirname(__file__), "..", "eval_golden_datasets", "retriever_dataset.json"
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


def run(_retriever=None):
    # NOTE: `_retriever` is kept (positionally compatible) for run_suite's
    # injected-pipeline contract, but
    # this project's retriever is PER-QUERY (get_retriever(query, document_id)
    # builds a Qdrant-filtered retriever per golden), so the injection point
    # doesn't apply here -- we always call get_retriever for each golden.
    del _retriever

    # 1. LOAD the golden set --- the fixed, human-authored truth
    goldens = load_goldens(GOLDEN_PATH)

    # 2. RUN THE REAL RETRIEVER (same one the app uses) on each question to
    #    fill retrieval_context, then build one test case per golden.
    test_cases = []
    for g in goldens[:10]:
        retriever_i = get_retriever(g["query"], g["document_id"])
        retrieved = retriever_i.invoke(g["query"])
        retrieval_context = [doc.page_content for doc in retrieved]

        test_cases.append(
            LLMTestCase(
                input=g["query"],
                expected_output=g["ideal_answer"],
                retrieval_context=retrieval_context,
                actual_output="(generator not evaluated in this run)",
            )
        )

    # 3. THE METRICS --- recall (did we miss?) and precision (ranked well?)
    metrics = [
        ContextualRecallMetric(
            threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=False, async_mode=True
        ),
        ContextualPrecisionMetric(
            threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=False, async_mode=True
        ),
    ]

    # 4. EVALUATE --- every metric on every case, batched + parallel, printed report.
    #    hyperparameters travel with the run so the report is tagged with the config.
    result = evaluate(
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
    return summarize_by_metric(result)


def run_local():
    """Standalone convenience: run against the app's real retriever."""
    return run()


if __name__ == "__main__":
    print_summary("retriever", run_local())
