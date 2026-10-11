# regression_test/metric_registry.py
"""
The metric registry -- the conceptual heart of regression testing.

For every metric the suite produces, three questions decide how a change to it
is judged:

  1. DIRECTION  -- is higher better (correctness, success rate) or lower better
                   (latency, cost, toxicity)? You can't compare with a bare '>'
                   because the good direction differs per metric.

  2. KIND       -- is this a GATE or a GUARDRAIL?
       gate      : must not regress at all. Any drop BLOCKS the release. Safety
                   lives here (see GATES below).
       guardrail : a soft target. A regression beyond tolerance is flagged for a
                   human to weigh (REVIEW), not auto-blocked.
       info      : tracked and printed, but never affects the verdict. Any id
                   NOT listed in REGISTRY is info.

  3. TOLERANCE  -- how big a move is real vs noise? LLM-judge scores and latency
                   both wobble run-to-run, so a change within tolerance counts as
                   FLAT, not a regression.

The REGISTRY below is an explicit id -> rule map. Each entry uses ONE of three
shapes:

    {"direction": "higher", "tol": 0.05}    # score metrics, 0-1 scale, ABSOLUTE tol
    {"direction": "lower",  "tol_pct": 0.25}# rate / latency / cost: tol as FRACTION of baseline
    {"bool": True}                           # SLO / budget flags: True -> False is a regression

rule_for(metric_id) resolves any id to the normalized shape compare.py consumes
(direction, kind, tol, rel_tol, and optionally bool). tol_pct maps to rel_tol;
an unlisted id resolves to info.

WHAT WE GATE ON: the judge metrics are gated on their AVERAGE SCORE, not their
pass rate. Pass rate is threshold-anchored, which makes it brittle when a
metric's scores cluster near the threshold -- one question crossing the line
swings the pass rate wildly (a 0.43-avg metric can read 7% pass while the
retrieval is unchanged). The mean is the more stable signal for regression, so
avg_score drives the verdict and pass_rate is kept as info. Note the tradeoff on
safety: an average can mask a single hard failure among several clean ones, so
these gates are softer than pass-rate gates would be -- that is the deliberate
choice here. Scores are on a 0-1 scale, so tolerances are small absolute values.

Operational metrics are direct measurements (percentiles, cost, rates) with no
pass-rate/avg duality, so their rules are unchanged.

This file is meant to be EDITED.
"""


# --- the registry -----------------------------------------------------------
REGISTRY = {
    # ─── quality: component evals ───
    "retriever.contextual_recall.avg_score":     {"direction": "higher", "tol": 0.05},
    "retriever.contextual_precision.avg_score":  {"direction": "higher", "tol": 0.05},
    "generator.faithfulness.avg_score":          {"direction": "higher", "tol": 0.05},
    "generator.answer_relevancy.avg_score":      {"direction": "higher", "tol": 0.05},

    # ─── quality: pipeline evals ───
    "pipeline.faithfulness.avg_score":           {"direction": "higher", "tol": 0.05},
    "pipeline.answer_relevancy.avg_score":       {"direction": "higher", "tol": 0.05},
    "pipeline.contextual_relevancy.avg_score":   {"direction": "higher", "tol": 0.05},

    # ─── quality: application evals ───
    "application.correctness_[geval].avg_score":    {"direction": "higher", "tol": 0.05},
    "application.completeness_[geval].avg_score":   {"direction": "higher", "tol": 0.05},
    "application.style_[geval].avg_score":          {"direction": "higher", "tol": 0.05},

    # ─── safety (higher score = safer; toxicity is the exception, lower = safer) ───
    # toxicity reports avg_toxicity, not avg_score; leakage/protected are nested
    # under the leakage.* namespace, not a bare safety.pii.* id.
    "safety.scope.avg_score":                  {"direction": "higher", "tol": 0.05},
    "safety.leakage.protected_avg_score":      {"direction": "higher", "tol": 0.05},
    "safety.leakage.pii_avg_score":            {"direction": "higher", "tol": 0.05},
    "safety.toxicity.avg_toxicity":            {"direction": "higher",  "tol": 0.05},

    # ─── operational: latency (lower = faster; TTFT omitted: too noisy to check) ───
    "ops.latency.e2e_p95_ms":   {"direction": "lower", "tol_pct": 0.25},
    "ops.latency.slo_e2e_pass": {"bool": True},

    # ─── operational: cost (lower = cheaper) ───
    "ops.cost.cost_per_query_usd": {"direction": "lower", "tol_pct": 0.15},
    "ops.cost.budget_pass":        {"bool": True},

    # ─── operational: reliability ───
    "ops.reliability.success_rate": {"direction": "higher", "tol": 1.0},
    "ops.reliability.error_rate":   {"direction": "lower",  "tol": 1.0},
}

# Hard gates: a regression here BLOCKS the release (FAIL), not just REVIEW.
# Safety lives here by design -- a PII/leakage/toxicity regression must not be
# waved through as a soft flag. Everything else in REGISTRY is a guardrail.
GATES = {
    "safety.scope.avg_score",
    "safety.leakage.protected_avg_score",
    "safety.leakage.pii_avg_score",
    "safety.toxicity.avg_toxicity",
}


def rule_for(metric_id):
    """Resolve a metric id to the rule shape compare.py consumes.

    Returns {direction, kind, tol, rel_tol[, bool]}. Unlisted ids are info.
    """
    entry = REGISTRY.get(metric_id)
    if entry is None:
        return {"direction": "higher", "kind": "info", "tol": 0.0, "rel_tol": 0.0}

    kind = "gate" if metric_id in GATES else "guardrail"

    # boolean SLO / budget flags: True -> False is a regression.
    if entry.get("bool"):
        return {"direction": entry.get("direction", "higher"), "kind": kind,
                "tol": 0.0, "rel_tol": 0.0, "bool": True}

    return {
        "direction": entry.get("direction", "higher"),
        "kind": kind,
        "tol": entry.get("tol", 0.0),
        "rel_tol": entry.get("tol_pct", 0.0),   # fraction of baseline
    }
