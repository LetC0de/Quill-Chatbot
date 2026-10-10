"""
Operational evals, merged: LATENCY, COST, RELIABILITY.

These three answer the OPERATIONAL question -- can the RAG app run reliably,
quickly, and economically? -- as opposed to the quality/safety evals, which
judge whether the answers are good. None of the three needs a golden dataset or
an LLM judge: they are software + telemetry measurements.

The one change from the three standalone files: each eval now exposes a run_*()
that PRINTS its report (for reading live) AND RETURNS a flat dict of metrics.
run_ops() runs all three against one pipeline and returns a single operational
snapshot -- the operational slice of the snapshot that regression testing will
diff against a baseline. The measurement logic itself is unchanged.

Per-eval notes, preserved:

  LATENCY   -- non-deterministic, so we take a distribution and report
               percentiles (p95/p99 tail, not the misleading mean) against an
               SLO -- not against a ground truth. We measure end-to-end AND
               time-to-first-token (perceived, what a streaming UI feels like),
               decompose retrieval vs generation, discard warmup runs (cold
               start), and stay single-user (load testing is a separate job).

  COST      -- derived, not measured: cost = tokens x price. Near-deterministic
               (temp=0 -> stable token counts), so it is an honest OFFLINE
               metric for unit economics: "can I afford to turn this on for N
               users/day?" is answerable before launch. Providers cache the
               fixed system prompt, so the real bill often runs BELOW this
               estimate; we surface cached tokens so you can watch it happen.

  RELIABILITY -- can we serve requests without failing? success / error / retry
               rates, with an exponential-backoff retry wrapper. On an ideal
               single-laptop setup this reads ~100% success; the numbers only
               get meaningful under real concurrency and flaky external APIs.

    python -m regression_test.eval_ops
"""

# ============================================================
# 1. IMPORTS & ENV
# ============================================================
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "backend", ".env"))
load_dotenv()

from src.rag.llm import llm  # type: ignore
from src.rag.prompt import concierge_prompt, prompt  # type: ignore
from src.rag.retriever import get_retriever  # type: ignore
from src.generator import generate  # type: ignore

# COST reads token usage off the AIMessage, so it composes prompt | llm directly
# and stops BEFORE the StrOutputParser() that generate() uses (which discards
# usage and returns a bare string). Same prompt, same model, real context.
measured_chain = prompt | llm
concierge_chain = concierge_prompt | llm   # no-context fallback, like generate()


# ============================================================
# 2. SHARED CONFIG
# ============================================================
# (question, document_id) tuples -- production retrieval is filtered by
# document_id, so every sample needs both (same set as evals/application_evals/
# eval_latency.py). Shapes are mixed on purpose: definition / comparison /
# mechanism / numerical / summary give short-to-long answers.
QUESTIONS = [
    ("What is Capgras' syndrome?", 21),
    ("How do L1 and L2 regularization differ in how they affect a neural network's weights?", 27),
    ("How does max pooling reduce the size of a feature map in a convolutional network?", 27),
    ("What is the concordance rate for schizophrenia in monozygotic twins?", 21),
    ("What biological and psychological theories have been proposed for the aetiology of schizophrenia?", 21),
]


# ============================================================
# 3. SHARED HELPERS
# ============================================================
# Linear-interpolation percentile -- same result as numpy.percentile, but
# written out so the tail is not a black box when you explain it in class.
def percentile(values, p):
    values = [v for v in values if not math.isnan(v)]
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return s[int(k)]
    return s[lo] * (hi - k) + s[hi] * (k - lo)


def col_avg(rows, key):
    return sum(r[key] for r in rows) / len(rows) if rows else 0.0


# One production request (exactly like graph/streaming.py): get_retriever
# -> retriever.invoke -> generate. There is no RagPipeline object in this
# project, so the adapters take (question, document_id) directly.
def run_once(question: str, document_id: int) -> str:
    retriever = get_retriever(question, document_id)
    docs = retriever.invoke(question)
    context = [doc.page_content for doc in docs]
    return generate(question, context)


# Streaming twin of src.generator.generate (eval-only; generator.py has no
# stream function). Same prompt/context logic as generate(); yields token
# deltas via llm.stream -- the sync twin of production's llm.astream in
# graph/streaming.py.
def generate_stream(query: str, context: list[str]):
    if not context:
        final_prompt = concierge_prompt.invoke({"question": query})
    else:
        context_parts = [f"[Page {i}] {chunk}" for i, chunk in enumerate(context, 1)]
        context_str = "\n\n".join(context_parts)
        final_prompt = prompt.invoke({"context": context_str, "question": query})

    for chunk in llm.stream(final_prompt):
        content = chunk.content
        if content and isinstance(content, str):
            yield content
        elif content:
            yield str(content)


# ============================================================================
# ============================  LATENCY  =====================================
# ============================================================================

# --- 4. LATENCY: CONFIG ---
LAT_REPEATS = 5           # measured runs per question -> samples = len(QUESTIONS)*LAT_REPEATS
LAT_WARMUP_RUNS = 2       # throwaway calls before measuring (cold start)
LAT_MEASURE_TTFT = True   # stream generation and clock time-to-first-token
LAT_STAGE_LEVEL = True    # split retrieval vs generation (implied when TTFT is on)

SLO_P95_MS = 3000         # end-to-end: full answer p95 under 3s
SLO_TTFT_P95_MS = 1200    # perceived: first visible token p95 under 1.2s


# --- 5. LATENCY: PIPELINE ADAPTERS (the ONE place you edit to match your API) ---
# End-to-end: full production chain per request.
def lat_end_to_end(question: str, document_id: int) -> str:
    return run_once(question, document_id)


# Stage-level (non-streaming): time retrieval vs generation. "retrieval" bundles
# get_retriever construction + query-embedding + vector search + the
# MMR/similarity pass.
def lat_stages(question: str, document_id: int):
    t0 = time.perf_counter()
    retriever = get_retriever(question, document_id)
    docs = retriever.invoke(question)
    context = [doc.page_content for doc in docs]
    t1 = time.perf_counter()
    answer = generate(question, context)
    t2 = time.perf_counter()
    return answer, {"retrieval": (t1 - t0) * 1000, "generation": (t2 - t1) * 1000}


# Stage-level (streaming): same retrieval, but stream generation and record the
# clock the instant the FIRST content token arrives.
#   ttft       = query submitted (t0) -> first visible token (what the user feels)
#   generation = generation start (t1) -> last token
# TTFT includes retrieval on purpose: the user waits through retrieval before the
# first token can stream, so perceived latency = retrieval + generation-prefill.
def lat_stages_streaming(question: str, document_id: int):
    t0 = time.perf_counter()
    retriever = get_retriever(question, document_id)
    docs = retriever.invoke(question)
    context = [doc.page_content for doc in docs]
    t1 = time.perf_counter()

    first_token_t = None
    pieces = []
    for piece in generate_stream(question, context):
        if first_token_t is None:
            first_token_t = time.perf_counter()   # clock the first non-empty chunk
        pieces.append(piece)
    t2 = time.perf_counter()

    answer = "".join(pieces)
    ttft_ms = (first_token_t - t0) * 1000 if first_token_t else float("nan")
    return answer, {
        "retrieval": (t1 - t0) * 1000,
        "generation": (t2 - t1) * 1000,
        "ttft": ttft_ms,
    }


# --- 6. LATENCY: BENCHMARK + AGGREGATE + REPORT ---
def lat_benchmark():
    # Warmup: run and DISCARD, so cold start does not pollute the stats.
    print(f"[latency] warming up ({LAT_WARMUP_RUNS} runs, discarded)...")
    for i in range(LAT_WARMUP_RUNS):
        question, document_id = QUESTIONS[i % len(QUESTIONS)]
        lat_end_to_end(question, document_id)

    total_ms, retrieval_ms, generation_ms, ttft_ms = [], [], [], []
    answer_lengths = []

    print("[latency] measuring...")
    for question, document_id in QUESTIONS:
        for _ in range(LAT_REPEATS):
            start = time.perf_counter()
            if LAT_MEASURE_TTFT:
                answer, stage = lat_stages_streaming(question, document_id)
                retrieval_ms.append(stage["retrieval"])
                generation_ms.append(stage["generation"])
                ttft_ms.append(stage["ttft"])
            elif LAT_STAGE_LEVEL:
                answer, stage = lat_stages(question, document_id)
                retrieval_ms.append(stage["retrieval"])
                generation_ms.append(stage["generation"])
            else:
                answer = lat_end_to_end(question, document_id)
            elapsed_ms = (time.perf_counter() - start) * 1000

            total_ms.append(elapsed_ms)
            answer_lengths.append(len(answer or ""))

    return {
        "total": total_ms,
        "retrieval": retrieval_ms,
        "generation": generation_ms,
        "ttft": ttft_ms,
        "answer_len": answer_lengths,
    }


def lat_summarize(samples):
    clean = [s for s in samples if not math.isnan(s)]
    return {
        "n": len(clean),
        "mean": sum(clean) / len(clean),
        "p50": percentile(clean, 50),
        "p95": percentile(clean, 95),
        "p99": percentile(clean, 99),
        "min": min(clean),
        "max": max(clean),
    }


def lat_print_row(label, s):
    print(f"{label:<12} | n={s['n']:<3} "
          f"mean={s['mean']:7.1f}  p50={s['p50']:7.1f}  "
          f"p95={s['p95']:7.1f}  p99={s['p99']:7.1f}  "
          f"min={s['min']:7.1f}  max={s['max']:7.1f}")


def lat_slo_line(label, p95, budget):
    verdict = "PASS" if p95 <= budget else "FAIL"
    print(f"SLO: {label:<22} p95 <= {budget:>5} ms  ->  p95 = {p95:7.0f} ms   [{verdict}]")


def lat_report(results):
    print("\n" + "=" * 78)
    print("LATENCY (milliseconds)")
    print("=" * 78)
    print(f"{'stage':<12} | {'samples':<5} {'mean':>11} {'p50':>11} "
          f"{'p95':>11} {'p99':>11} {'min':>11} {'max':>11}")
    print("-" * 78)

    total = lat_summarize(results["total"])
    lat_print_row("end-to-end", total)
    if results["ttft"]:
        lat_print_row("ttft", lat_summarize(results["ttft"]))   # perceived
    if results["retrieval"]:
        lat_print_row("retrieval", lat_summarize(results["retrieval"]))
        lat_print_row("generation", lat_summarize(results["generation"]))

    avg_len = sum(results["answer_len"]) / len(results["answer_len"])
    print("-" * 78)
    print(f"avg answer length: {avg_len:.0f} chars "
          f"(latency scales with output length -- keep in mind when comparing configs)")

    print("=" * 78)
    lat_slo_line("full answer", total["p95"], SLO_P95_MS)
    if results["ttft"]:
        lat_slo_line("first token (perceived)", lat_summarize(results["ttft"])["p95"], SLO_TTFT_P95_MS)
    print("=" * 78)


def run_latency(verbose=True):
    """Measure latency; print the report and return a flat metrics dict."""
    results = lat_benchmark()
    if verbose:
        lat_report(results)

    total = lat_summarize(results["total"])
    metrics = {
        "e2e_mean_ms": total["mean"],
        "e2e_p50_ms": total["p50"],
        "e2e_p95_ms": total["p95"],
        "e2e_p99_ms": total["p99"],
        "avg_answer_len": sum(results["answer_len"]) / len(results["answer_len"]),
        "slo_e2e_pass": total["p95"] <= SLO_P95_MS,
    }
    if results["ttft"]:
        ttft = lat_summarize(results["ttft"])
        metrics["ttft_p95_ms"] = ttft["p95"]
        metrics["slo_ttft_pass"] = ttft["p95"] <= SLO_TTFT_P95_MS
    if results["retrieval"]:
        metrics["retrieval_p95_ms"] = lat_summarize(results["retrieval"])["p95"]
        metrics["generation_p95_ms"] = lat_summarize(results["generation"])["p95"]
    return metrics


# ============================================================================
# =============================  COST  =======================================
# ============================================================================

# --- 7. COST: CONFIG ---
COST_REPEATS = 3          # cost is stable, so fewer repeats needed than latency

# Pricing: reference rates for gpt-4o-mini, USD per 1M tokens (verified Aug 2026).
# Prices change -- keep them here as constants, never buried in code, and
# re-check the provider's pricing page for the model configured in
# src/rag/llm.py before trusting a budget.
PRICE_INPUT_PER_1M        = 0.15    # cache-miss input
PRICE_CACHED_INPUT_PER_1M = 0.075   # cached (repeated prefix) input -- half price
PRICE_OUTPUT_PER_1M       = 0.60    # output (4x input -- long answers dominate)

# Business projection knobs (set these to YOUR reality).
QUERIES_PER_DAY = 2000              # expected doubt-solver traffic
USD_TO_INR      = 96.0              # approximate; set to the current rate

# Budget (the "SLO" for cost): the offline pass/fail line.
COST_BUDGET_PER_QUERY_USD = 0.0015  # e.g. must stay under ~0.13 INR / query


# --- 8. COST: TOKEN MEASUREMENT + COST MATH ---
# Retrieve real context (so input tokens reflect your actual retriever load,
# filtered by document_id exactly like production), then run one generation and
# read the token usage off the message.
def cost_measure_tokens(question: str, document_id: int):
    retriever = get_retriever(question, document_id)
    docs = retriever.invoke(question)
    context = [doc.page_content for doc in docs]

    # Same context formatting as src/generator.py: "[Page N]" labels, and the
    # concierge prompt when the retriever returned nothing.
    if not context:
        chain, inputs = concierge_chain, {"question": question}
    else:
        context_str = "\n\n".join(f"[Page {i}] {chunk}" for i, chunk in enumerate(context, 1))
        chain, inputs = measured_chain, {"question": question, "context": context_str}

    msg = chain.invoke(inputs)
    usage = msg.usage_metadata or {}
    if not usage:
        # Fallback for providers that only fill response_metadata
        rt = (msg.response_metadata or {}).get("token_usage", {})
        usage = {
            "input_tokens": rt.get("prompt_tokens", 0),
            "output_tokens": rt.get("completion_tokens", 0),
        }

    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    # cached prefix tokens, if the provider reports them
    details = usage.get("input_token_details") or {}
    cached_tokens = details.get("cache_read", 0) or 0

    return {"input": input_tokens, "output": output_tokens, "cached": cached_tokens}


# cost = uncached_input @ full rate + cached_input @ cached rate + output @ output rate
def cost_usd(input_tokens, output_tokens, cached_tokens):
    uncached_input = max(input_tokens - cached_tokens, 0)
    c_in     = uncached_input / 1_000_000 * PRICE_INPUT_PER_1M
    c_cached = cached_tokens  / 1_000_000 * PRICE_CACHED_INPUT_PER_1M
    c_out    = output_tokens  / 1_000_000 * PRICE_OUTPUT_PER_1M
    return {"input": c_in, "cached": c_cached, "output": c_out,
            "total": c_in + c_cached + c_out}


# --- 9. COST: BENCHMARK + REPORT ---
def cost_benchmark():
    rows = []
    print("[cost] measuring token usage...")
    for question, document_id in QUESTIONS:
        for _ in range(COST_REPEATS):
            tok = cost_measure_tokens(question, document_id)
            cost = cost_usd(tok["input"], tok["output"], tok["cached"])
            rows.append({**tok, **{f"cost_{k}": v for k, v in cost.items()}})
    return rows


def cost_report(rows):
    n = len(rows)
    avg_in     = col_avg(rows, "input")
    avg_out    = col_avg(rows, "output")
    avg_cached = col_avg(rows, "cached")
    avg_cost   = col_avg(rows, "cost_total")
    min_cost   = min(r["cost_total"] for r in rows)
    max_cost   = max(r["cost_total"] for r in rows)

    avg_cost_out = col_avg(rows, "cost_output")
    out_share = 100 * avg_cost_out / avg_cost if avg_cost else 0

    model = getattr(llm, "model_name", None) or "llm"
    print("\n" + "=" * 70)
    print(f"COST  ({model} @ ${PRICE_INPUT_PER_1M}/${PRICE_OUTPUT_PER_1M} per 1M in/out)")
    print("=" * 70)
    print(f"samples                : {n}")
    print(f"avg input tokens       : {avg_in:8.0f}   ({avg_cached:.0f} cached)")
    print(f"avg output tokens      : {avg_out:8.0f}")
    print("-" * 70)
    print(f"avg cost / query       : ${avg_cost:.6f}   (Rs {avg_cost * USD_TO_INR:.4f})")
    print(f"   min / max           : ${min_cost:.6f} / ${max_cost:.6f}   "
          f"<- tight range = cost is stable, unlike latency")
    print(f"   input vs output     : {100 - out_share:.0f}% input / {out_share:.0f}% output "
          f"(output is 4x the rate -> long answers dominate)")
    print("-" * 70)

    daily   = avg_cost * QUERIES_PER_DAY
    monthly = daily * 30
    print(f"projection @ {QUERIES_PER_DAY}/day :")
    print(f"   per day             : ${daily:8.2f}   (Rs {daily * USD_TO_INR:8.2f})")
    print(f"   per month           : ${monthly:8.2f}   (Rs {monthly * USD_TO_INR:8.2f})")
    print("=" * 70)

    verdict = "PASS" if avg_cost <= COST_BUDGET_PER_QUERY_USD else "FAIL"
    print(f"BUDGET: cost/query <= ${COST_BUDGET_PER_QUERY_USD:.6f}  ->  "
          f"${avg_cost:.6f}   [{verdict}]")
    print("=" * 70)
    print("note: production caching of the (large, fixed) system prompt can push the")
    print("real bill BELOW this estimate -- watch the 'cached' count grow online.")


def run_cost(verbose=True):
    """Measure cost; print the report and return a flat metrics dict."""
    rows = cost_benchmark()
    if verbose:
        cost_report(rows)

    avg_cost = col_avg(rows, "cost_total")
    avg_cost_out = col_avg(rows, "cost_output")
    out_share = 100 * avg_cost_out / avg_cost if avg_cost else 0.0
    return {
        "cost_per_query_usd": avg_cost,
        "cost_per_query_inr": avg_cost * USD_TO_INR,
        "avg_input_tokens": col_avg(rows, "input"),
        "avg_output_tokens": col_avg(rows, "output"),
        "avg_cached_tokens": col_avg(rows, "cached"),
        "output_cost_share_pct": out_share,
        "monthly_usd": avg_cost * QUERIES_PER_DAY * 30,
        "budget_pass": avg_cost <= COST_BUDGET_PER_QUERY_USD,
    }


# ============================================================================
# ==========================  RELIABILITY  ===================================
# ============================================================================

# --- 10. RELIABILITY: CONFIG ---
REL_REPEATS = 5
MAX_RETRIES = 2
BACKOFF_BASE_S = 0.5


# --- 11. RELIABILITY: TRACKER + RETRY WRAPPER ---
class Reliability:
    def __init__(self):
        self.calls = 0
        self.successes = 0
        self.failures = 0
        self.retries = 0        # NOTE: counts total retry ATTEMPTS, not requests
                                # that needed >=1 retry. Fine at 0 failures; see
                                # the run_reliability note if you want per-request.


def call_with_retries(question, document_id, reliability):
    reliability.calls += 1
    for attempt in range(MAX_RETRIES + 1):
        try:
            run_once(question, document_id)
            reliability.successes += 1
            return True
        except Exception as e:
            if attempt < MAX_RETRIES:
                reliability.retries += 1
                time.sleep(BACKOFF_BASE_S * (2 ** attempt))   # exponential backoff
            else:
                reliability.failures += 1
                print(f"[reliability] FAILED after {MAX_RETRIES} retries: {e}")
                return False


# --- 12. RELIABILITY: BENCHMARK + REPORT ---
def rel_benchmark():
    reliability = Reliability()
    print("[reliability] measuring...")
    for question, document_id in QUESTIONS:
        for _ in range(REL_REPEATS):
            call_with_retries(question, document_id, reliability)
    return reliability


def rel_report(rel):
    calls = rel.calls or 1
    success_rate = 100 * rel.successes / calls
    error_rate   = 100 * rel.failures / calls
    retry_rate   = 100 * rel.retries / calls

    print("\n" + "=" * 60)
    print("RELIABILITY")
    print("=" * 60)
    print(f"total requests : {rel.calls}")
    print(f"successful     : {rel.successes}")
    print(f"failed         : {rel.failures}")
    print("-" * 60)
    print(f"success rate   : {success_rate:.2f}%")
    print(f"error rate     : {error_rate:.2f}%")
    print(f"retry rate     : {retry_rate:.2f}%")
    print("=" * 60)


def run_reliability(verbose=True):
    """Measure reliability; print the report and return a flat metrics dict."""
    rel = rel_benchmark()
    if verbose:
        rel_report(rel)

    calls = rel.calls or 1
    return {
        "total_requests": rel.calls,
        "success_rate": 100 * rel.successes / calls,
        "error_rate": 100 * rel.failures / calls,
        "retry_rate": 100 * rel.retries / calls,
    }


# ============================================================
# 13. ENTRYPOINT  (runs all three, returns the operational snapshot)
# ============================================================
# One pipeline, three evals, one merged snapshot. The dotted keys become metric
# ids -- this dict is the operational slice of what regression testing diffs
# against a baseline.
def run_ops(pipeline=None, verbose=True):
    # NOTE: `pipeline` is kept (positionally compatible) for run_suite's
    # injected-pipeline contract, but this project has no pipeline object --
    # the chain (get_retriever -> generate) runs inline via run_once() above.
    del pipeline

    latency     = run_latency(verbose=verbose)
    cost        = run_cost(verbose=verbose)
    reliability = run_reliability(verbose=verbose)

    snapshot = {}
    snapshot.update({f"latency.{k}": v for k, v in latency.items()})
    snapshot.update({f"cost.{k}": v for k, v in cost.items()})
    snapshot.update({f"reliability.{k}": v for k, v in reliability.items()})
    return snapshot


def main():
    snapshot = run_ops(verbose=True)

    print("\n" + "=" * 70)
    print("OPERATIONAL SNAPSHOT  (feeds regression testing)")
    print("=" * 70)
    for key, value in snapshot.items():
        shown = f"{value:.6f}" if isinstance(value, float) else str(value)
        print(f"  {key:<30} {shown}")
    print("=" * 70)


if __name__ == "__main__":
    main()
