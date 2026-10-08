"""
Operational eval: COST.

Cost is not measured, it is DERIVED: cost = tokens x price. So the real work is
getting an honest token count, then multiplying by the current per-token rate.

Why cost is a legitimately OFFLINE metric (unlike latency):
  - tokens are near-deterministic. Same retrieved context + temperature=0 output
    => almost the same token counts every run. So cost barely moves run-to-run,
    whereas latency wobbled ~700ms on us with zero code changes.
  - that stability is what lets you estimate unit economics BEFORE launch:
    "can I afford to turn this on for N users/day?" is answerable offline.

Honest caveat baked in below: providers CACHE repeated prompt prefixes. Your big
faithfulness-first system prompt is identical on every call, so in production a
chunk of your input tokens bill at the cheaper cached rate -- meaning the real
bill is often LOWER than this offline estimate. We surface cached tokens so you
can see it happen.

How we get tokens: your generate() chain ends in a string return that throws
usage away. So we import the same prompt + llm and compose `prompt | llm`
ourselves, stopping one step early to read usage_metadata off the AIMessage.
Same prompt, same model, real retrieved context (filtered by document_id, like
production).

Run from the project root (same as the other application evals):
    python -m evals.application_evals.eval_cost
"""

# ============================================================
# 1. IMPORTS & ENV
# ============================================================
import os
import sys

# Add backend to path so we can import src modules (same as other app evals)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "backend"))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", "backend", ".env"))

from src.rag.llm import llm  # type: ignore
from src.rag.prompt import concierge_prompt, prompt  # type: ignore
from src.rag.retriever import get_retriever  # type: ignore

# Stop before the string-returning step so the AIMessage (with usage_metadata)
# survives. Same prompt + model production uses in src/generator.py.
measured_chain = prompt | llm
concierge_chain = concierge_prompt | llm   # no-context fallback, like generate()

# ============================================================
# 2. CONFIG
# ============================================================
# (question, document_id) tuples -- production retrieval is filtered by
# document_id, so every sample needs both (same set as eval_latency.py).
# Facts verified against chunks_dump (evals/component_evals/.deepeval/).
QUESTIONS = [
    ("What is Capgras' syndrome?", 21),
    ("How do L1 and L2 regularization differ in how they affect a neural network's weights?", 27),
    ("How does max pooling reduce the size of a feature map in a convolutional network?", 27),
    ("What is the concordance rate for schizophrenia in monozygotic twins?", 21),
    ("What biological and psychological theories have been proposed for the aetiology of schizophrenia?", 21),
]

REPEATS = 3       # cost is stable, so fewer repeats needed than latency

# --- Pricing, USD per 1M tokens. Prices change: keep them here as constants,
# never buried in code, and re-check the provider's pricing page for the model
# configured in src/rag/llm.py before trusting a budget. ---
# (reference rates for gpt-4o-mini, verified Aug 2026)
PRICE_INPUT_PER_1M        = 0.15    # cache-miss input
PRICE_CACHED_INPUT_PER_1M = 0.075   # cached (repeated prefix) input -- half price
PRICE_OUTPUT_PER_1M       = 0.60    # output (4x input -- long answers dominate)

# --- Business projection knobs (set these to YOUR reality) ---
QUERIES_PER_DAY = 2000              # expected doubt-solver traffic
USD_TO_INR      = 96.0              # approximate; set to the current rate

# --- Budget (the "SLO" for cost): the offline pass/fail line ---
COST_BUDGET_PER_QUERY_USD = 0.0015  # e.g. must stay under ~0.13 INR / query

# ============================================================
# 3. TOKEN MEASUREMENT
# ============================================================
# Retrieve real context (so input tokens reflect your actual retriever load,
# filtered by document_id exactly like production), then run one generation and
# read the token usage off the message.
def measure_tokens(question: str, document_id: int):
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

    return {
        "input": input_tokens,
        "output": output_tokens,
        "cached": cached_tokens,
    }

# ============================================================
# 4. COST MATH
# ============================================================
# cost = uncached_input @ full rate + cached_input @ cached rate + output @ output rate
def cost_usd(input_tokens, output_tokens, cached_tokens):
    uncached_input = max(input_tokens - cached_tokens, 0)
    c_in     = uncached_input / 1_000_000 * PRICE_INPUT_PER_1M
    c_cached = cached_tokens  / 1_000_000 * PRICE_CACHED_INPUT_PER_1M
    c_out    = output_tokens  / 1_000_000 * PRICE_OUTPUT_PER_1M
    return {"input": c_in, "cached": c_cached, "output": c_out,
            "total": c_in + c_cached + c_out}

# ============================================================
# 5. BENCHMARK LOOP
# ============================================================
def benchmark():
    rows = []
    print("Measuring token usage...")
    for question, document_id in QUESTIONS:
        for _ in range(REPEATS):
            tok = measure_tokens(question, document_id)
            cost = cost_usd(tok["input"], tok["output"], tok["cached"])
            rows.append({**tok, **{f"cost_{k}": v for k, v in cost.items()}})
    return rows

# ============================================================
# 6. AGGREGATE + REPORT
# ============================================================
def avg(rows, key):
    return sum(r[key] for r in rows) / len(rows)

def report(rows):
    n = len(rows)
    avg_in     = avg(rows, "input")
    avg_out    = avg(rows, "output")
    avg_cached = avg(rows, "cached")
    avg_cost   = avg(rows, "cost_total")
    min_cost   = min(r["cost_total"] for r in rows)
    max_cost   = max(r["cost_total"] for r in rows)

    # split: how much of the bill is input vs output
    avg_cost_out = avg(rows, "cost_output")
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

    # --- projection: the number a founder actually cares about ---
    daily   = avg_cost * QUERIES_PER_DAY
    monthly = daily * 30
    print(f"projection @ {QUERIES_PER_DAY}/day :")
    print(f"   per day             : ${daily:8.2f}   (Rs {daily * USD_TO_INR:8.2f})")
    print(f"   per month           : ${monthly:8.2f}   (Rs {monthly * USD_TO_INR:8.2f})")
    print("=" * 70)

    # --- budget verdict (the offline pass/fail) ---
    verdict = "PASS" if avg_cost <= COST_BUDGET_PER_QUERY_USD else "FAIL"
    print(f"BUDGET: cost/query <= ${COST_BUDGET_PER_QUERY_USD:.6f}  ->  "
          f"${avg_cost:.6f}   [{verdict}]")
    print("=" * 70)
    print("note: production caching of the (large, fixed) system prompt can push the")
    print("real bill BELOW this estimate -- watch the 'cached' count grow online.")

# ============================================================
# 7. ENTRYPOINT
# ============================================================
def main():
    rows = benchmark()
    report(rows)

if __name__ == "__main__":
    main()
