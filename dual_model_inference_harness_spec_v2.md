# Dual-Model Local Inference Harness — v2 Spec

## 0. Thesis

> **How much task quality can we extract from small local models by spending hardware throughput on structured search instead of model size?**

Everything in this document serves that question. v2 tightens v1 by (a) making the thesis measurable, (b) reducing reliance on noisy LLM-as-judge signal, (c) treating hardware constraints (VRAM residency, KV prefix reuse) as first-class design inputs, and (d) splitting verifiable from non-verifiable tasks as separate control-flow paths.

Non-goals, explicitly:
- Chat / agent framework.
- Roleplay multi-agent system.
- Calling hosted frontier models at runtime. A hosted model may be used **only at dev-time** for judge calibration (see §10).

---

## 1. What Changed From v1

| Area | v1 | v2 |
|---|---|---|
| Judge weighting | cross-eval 0.65, self-eval 0.10, deterministic 0.25 | deterministic dominates when available; LLM weights collapse to tiebreak |
| Task classes | single control flow | two paths: **verifiable** and **non-verifiable** |
| Model residency | unspecified | explicit `coresident` vs `swapped` config |
| KV prefix reuse | not mentioned | mandatory backend capability; prompt format is prefix-first |
| Repair stage | always runs | gated on evaluation disagreement or specific flagged flaws |
| Tournament | always pairwise after repair | absolute scores first; pairwise only on close finalists |
| Baselines | none stated | explicit single-shot baseline required, reported per run |
| Quality targets | implicit | per-task-class uplift targets gate phase advancement |
| JSON reliability | assumed | constrained decoding required where backend supports it |
| Metrics | listed | each metric tied to an action or decision |

---

## 2. Core Principles (revised)

1. **Verify before you judge.** Deterministic checks always run first and dominate when they exist. LLM judgment is a last resort, not a default.
2. **Judge noise is the bottleneck.** Small-model judges have strong positional, length, and praise-collapse biases. The pipeline is designed to minimize total LLM-judgment calls and to calibrate the ones it does make.
3. **Hardware shapes the pipeline.** VRAM residency, KV prefix cache, and batch throughput are load-bearing design inputs, not ops details.
4. **Breadth cheap, depth expensive.** Spend tokens on many short independent candidates; spend tokens on long-form output only at synthesis.
5. **Measure uplift or stop spending.** Every stage must demonstrate measurable gain over the prior stage on at least one task class, or it is disabled for that class.
6. **Disagreement is a trigger, not a score.** Use disagreement to *route* (repair, extra candidate, escalate) rather than to bias the aggregate score.
7. **Config over code.** Search width, weights, and routing are YAML. Changing them should not require touching the pipeline.

---

## 3. Scope

### 3.1 In scope
Prompt orchestration, batched candidate generation, deterministic validators, rubric-driven cross-model scoring (gated), targeted repair (gated), pairwise finals (gated), single-pass synthesis, trace + metrics collection, backend abstraction, judge calibration at dev-time.

### 3.2 Out of scope (v1 scope carries forward)
Autonomous tool use, browser integration, distributed scheduling, cross-project memory, fine-tuning, RL.

### 3.3 Deliberately deferred to v3+
Learned routing, learned judge calibration, candidate reuse across similar tasks, speculative decoding, tree search beyond two rounds.

---

## 4. Architecture

```text
User Task
   |
   v
Task Normalizer -------> (task class: verifiable | non-verifiable)
   |
   v
Planner / Rubric Builder (+ verifier spec if verifiable)
   |
   v
Proposal Stage (A + B, batched, prefix-shared, JSON-constrained)
   |
   v
Deterministic Validation (always, if applicable)
   |
   v
Pruner #1 (drop hard failures)
   |
   v
[Branch]
   |---- verifiable path: tiebreak among passers via cheap pairwise
   |
   |---- non-verifiable path: cross-eval rubric score + calibrated
   |     aggregation + disagreement flag
   v
Pruner #2 (keep top K)
   |
   v
Adversarial Repair (only for candidates with specific flagged flaws)
   |
   v
Finalist Selection
   |     (single top candidate if clear; pairwise only if scores within ε)
   v
Final Synthesizer (interactive backend)
   |
   v
Final Output + Trace + Metrics
```

---

## 5. Task Classes

Every task is classified at normalization time. The pipeline branches from here.

### 5.1 Verifiable
Characterized by a cheap, deterministic correctness check:
- code that runs against tests
- math with a checkable answer
- structured output against a JSON Schema
- extraction with known ground truth
- transformations whose inverse is checkable

**Rule:** the verifier is the primary judge. LLM judgment only ranks among passing candidates, and only if >1 passes.

### 5.2 Non-verifiable
No cheap deterministic signal: open writing, analysis, planning, unconstrained summarization.

**Rule:** LLM judgment is unavoidable. Apply calibration (§10) and keep judge-call counts minimal.

### 5.3 Mixed
Some success criteria verifiable, others not. Treat verifiable criteria as hard filters; apply non-verifiable path only among passers.

---

## 6. Execution Model

### 6.1 Models
Minimum two: Model A, Model B. Roles (proposer / critic / verifier / synthesizer) are stage-specific.

### 6.2 Residency (explicit decision)

```yaml
residency:
  mode: coresident | swapped
```

- `coresident`: both models fit in available VRAM at chosen quantization. Stages can interleave freely.
- `swapped`: only one fits. Pipeline must be **model-batched**: all of A's work for a stage runs together, then B's. Swap cost is paid O(stages), not O(candidates).

The scheduler (§12) enforces this.

### 6.3 Runtime modes
- **Batch backend**: proposals, scoring, repair, pairwise. High concurrency, low-latency-unimportant.
- **Interactive backend**: final synthesis only. Often the same model, lower batch size, streaming.

### 6.4 Default search widths

Per task class. These are starting defaults, not prescriptions.

```yaml
verifiable:
  proposals_per_model: 4      # 8 total
  post_validation_keep: passers + top-2 failing  # never lose context
  finalists: 1 (or pairwise among ≤3 passers)

non_verifiable:
  proposals_per_model: 3      # 6 total — judge cost dominates
  post_score_keep: 3
  post_repair_keep: 2
  finalists: 1 (pairwise only if top-2 within 0.05 aggregate)
```

Raising these numbers must be justified by measured uplift.

---

## 7. Pipeline Stages

Unchanged stages from v1 are referenced but not repeated. Changes are called out.

### 7.1 Stage 0 — Normalization

Output additions over v1:
```json
{
  "task_class": "verifiable | non_verifiable | mixed",
  "verifier_spec": {
    "kind": "tests | json_schema | regex | function | none",
    "artifact": "string (path, schema, predicate ref)"
  } | null
}
```

Prefer deterministic parsing. Use a single quick LLM call only when classification is ambiguous.

### 7.2 Stage 1 — Rubric Builder

Rubric weights are still output, but are **only applied on the non-verifiable path**. For verifiable tasks the rubric is used only as a checklist for repair prompts, not for scoring.

### 7.3 Stage 2 — Proposal

Additions:
- **Strategy axis is explicit**, drawn from an enumerated set to enforce real diversity:
  ```
  ["baseline", "decompose-first", "edge-case-first",
   "brevity", "evidence-heavy", "inverted-assumptions"]
  ```
  The 8 proposals must cover at least 4 distinct strategies.
- **Prompt format is prefix-first**: `<system><rubric><task><strategy_delta>`. The first three blocks are identical across all 8 calls so the backend can share KV cache.
- **JSON output is constrained**, not requested. On backends that support grammar/schema-constrained decoding (llama.cpp grammars, vLLM guided decoding, SGLang regex), use it. On backends that do not, wrap with a repair-parse + one retry.

### 7.4 Stage 3a — Deterministic Validation (verifiable path)

Runs before any LLM judgment. Records for each candidate:
- pass/fail on verifier
- failure signature (compile error class, failed test name, schema violation path)

Candidates that fail verification are dropped unless <2 pass, in which case up to 2 near-miss failures are kept for repair.

### 7.5 Stage 3b — Cross-Model Scoring (non-verifiable path, and verifiable tiebreak only when ≥2 pass)

Changes from v1:
- **Aggregate formula reweighted**:
  ```
  combined_score =
      0.70 * deterministic_score_normalized  (1.0 if pass, 0.0 if fail, partial if spec allows)
    + 0.25 * cross_score_mean
    + 0.05 * self_score
    - penalty_terms
  ```
  When no deterministic signal exists, weights renormalize to `(cross=0.83, self=0.17)`.
- **Self-eval is optional** and disabled by default. Enable per-task-class only if it demonstrably helps on the eval set.
- **Disagreement does not change the score.** It only sets a flag:
  ```
  disagreement_flag = |score_A - score_B| > τ
  ```
  Flagged candidates go to repair regardless of rank; unflagged ones go to repair only if below-median.
- **Positional and length bias controls**: when A scores B's candidates, randomize candidate order within the prompt and include length in the token budget check, not as a score input.

### 7.6 Stage 4 — Pruning

Two-step:
1. After deterministic validation: drop hard failures (verifiable path) / schema-invalid outputs.
2. After LLM scoring: keep top K by aggregate; always include one flagged-disagreement candidate if it is above a minimum viability threshold.

### 7.7 Stage 5 — Adversarial Repair (gated)

Repair is **no longer blanket**. It runs when at least one of:
- a specific detected flaw exists (from validator or eval `fatal_issues` / `minor_issues`)
- `disagreement_flag` is set
- candidate is below-median but above viability threshold

This targets the v1 "shared blind spot" problem: blanket repair often produces no-ops because the attacker model shares the flaw. Metric `repair_noop_rate` (§11) tracks this; if it exceeds 0.5 for a task class, repair is auto-disabled for that class in subsequent runs (config flag).

Rule stays: A repairs B's, B repairs A's. Repair output must reference specific flaws, not freestyle rewrite.

### 7.8 Stage 6 — Finalist Selection

- If top candidate leads runner-up by more than `ε` (default 0.05): skip pairwise, use top.
- Otherwise, run pairwise among top-3 with both judges, randomized order and position-swap. Winner is argmax of confidence-weighted wins.
- Bradley-Terry is overkill at n≤3; deferred to v3.

### 7.9 Stage 7 — Final Synthesis

Unchanged from v1 intent. Clarifications:
- Runs on interactive backend.
- Receives the finalist content, unresolved risks, and any open uncertainties. Does not see raw candidate pool or eval scores.
- Must preserve verified content verbatim when task is verifiable (no silent edits to code that passed tests).

---

## 8. Prompting Contract

Unchanged families from v1, with additions:
- `validator_invoke_v1` (invokes the deterministic verifier, not a prompt)
- `repair_targeted_v1` (replaces v1's generic repair)
- `pairwise_ordered_v1` (with position-swap control)

Rules:
- Terse, schema-first.
- JSON constrained by grammar where possible.
- Explicit instruction to refuse to invent unsupported facts; say "insufficient evidence" instead.

---

## 9. Backend Requirements

### 9.1 Must
- OpenAI-compatible chat completions or native equivalent.
- Batched generation (multiple requests in one backend round trip) OR fast sequential with shared KV cache.
- KV prefix cache reuse within a batch. If the backend cannot do this, the pipeline falls back to sequential and reports a throughput-degraded flag.
- Stop sequences, temperature, top_p, seed.

### 9.2 Should
- Grammar / JSON-schema constrained decoding.
- Speculative decoding or draft models (for v3 latency work).
- Per-request `priority` hint for the scheduler.

### 9.3 v1 adapters
Same as v1 list (OpenAI-compatible, vLLM, llama.cpp server, optional SGLang), with capability negotiation at init:
```python
class BackendCapabilities:
    supports_prefix_cache: bool
    supports_constrained_decoding: bool
    max_batch_size: int
    coresident_compatible: bool
```

---

## 10. Judge Calibration (dev-time only)

The weakest link is LLM-as-judge quality from small local models. We address it once, at dev-time, using an external oracle — then ship local-only.

### 10.1 The pattern
1. Curate a small calibration set (30–100 items per task class) where ground truth is knowable: unit-tested code, math with answers, human-labeled pairwise preferences for writing.
2. For each item, generate candidates with the harness and have both local models score/compare them.
3. Optionally have a frontier model score the same items as an **oracle**. The oracle is never called at runtime; it only exists to answer: "how well does our local judge track truth?"
4. Compute judge agreement with oracle per rubric dimension, positional-bias rate, length-bias rate, praise-collapse rate.
5. Bake findings into weights and prompt tweaks (e.g., "Judge A is reliable on `correctness` but not on `clarity`, so down-weight its clarity score").

### 10.2 What this is not
- Not a runtime dependency. The shipped harness uses only local models.
- Not a training signal. No fine-tuning, no RL, no distillation from the oracle.
- Not silent. Every calibration run is logged and the resulting config changes are diffable.

### 10.3 Why this is consistent with the thesis
The thesis asks what can be extracted from local models, not how to discover that without help. Using a strong oracle once to learn that "Model B is a poor judge of clarity but fine on correctness" is the same class of activity as running a benchmark — it shapes the config, it does not run the task.

---

## 11. Metrics

Each metric exists to drive a decision. Metrics with no linked action are removed.

### 11.1 System
| Metric | Decision it drives |
|---|---|
| tokens/sec per backend | backend selection, batch size tuning |
| stage wall-clock | identify bottleneck stage |
| batch size distribution | scheduler policy |
| JSON validity rate | enable/disable constrained decoding |
| prefix cache hit rate | validate prompt-format design |

### 11.2 Quality
| Metric | Decision it drives |
|---|---|
| single-shot baseline score per task class | required reference — every run reports this alongside harness score |
| harness uplift over baseline | phase advancement gate |
| judge-oracle agreement (dev) | judge weight config |
| positional bias rate (dev) | prompt order randomization config |
| disagreement flag rate | tune `τ` |
| repair_noop_rate | auto-disable repair per class |
| pairwise stability (swap-order consistency) | disable pairwise if unstable |
| marginal gain per extra candidate | tune search width |

### 11.3 Targets (phase gates)

Phase 1 cannot ship until:
- end-to-end run completes on 3 task classes
- harness beats single-shot baseline by ≥5% on at least one class
- trace store is queryable

Phase 2 cannot ship until:
- repair stage demonstrates ≥3% uplift on at least one class where it runs
- `repair_noop_rate` measured and documented

Phase 3 ships when:
- ablation report for each stage exists
- cost-quality frontier plotted per task class

If a target is not met, the stage is disabled for that class rather than shipped with unclear value.

---

## 12. Scheduler

Additions over v1:
- Stage scheduling honors `residency.mode`. In `swapped` mode, the scheduler enforces strict model-batching: all Model-A work for a stage completes before swapping.
- Prefix-aware batching: requests are grouped by shared prefix hash so KV cache is actually reused.
- Backpressure: if a stage's queue exceeds N pending requests, upstream stages pause.

Retry policy stays minimal (1 retry on transient backend error; no retry on invalid JSON when constrained decoding is on).

---

## 13. Configuration

Single YAML file, loaded once per run, hashed into the trace.

```yaml
run:
  task_type_default: general_reasoning
  trace_dir: ./traces
  baseline_model: model_a    # used for single-shot baseline run

models:
  model_a:
    name: qwen
    backend: vllm
    quantization: q4_k_m
  model_b:
    name: gemma
    backend: vllm
    quantization: q4_k_m

residency:
  mode: coresident           # or: swapped

backends:
  vllm:
    endpoint: http://localhost:8000
    prefer_constrained_decoding: true

search:
  verifiable:
    proposals_per_model: 4
    finalists: 1
  non_verifiable:
    proposals_per_model: 3
    post_score_keep: 3
    post_repair_keep: 2

scoring:
  non_verifiable:
    deterministic_weight: 0.0     # no signal
    cross_eval_weight: 0.83
    self_eval_weight: 0.17        # disabled unless enabled explicitly
    self_eval_enabled: false
  verifiable:
    deterministic_weight: 0.70
    cross_eval_weight: 0.25
    self_eval_weight: 0.05
    self_eval_enabled: false

disagreement:
  tau: 0.25

repair:
  enabled: true
  gates:
    - below_median_and_viable
    - disagreement_flag_set
    - has_specific_flagged_flaw
  auto_disable_if_noop_rate_above: 0.5

pairwise:
  epsilon_to_skip: 0.05
  randomize_position: true
  swap_order_consistency_check: true

budgets:
  proposal_max_tokens: 400
  eval_max_tokens: 200
  repair_max_tokens: 300
  final_max_tokens: 1200

calibration:
  oracle_backend: null        # set only at dev-time
  calibration_set_path: null

routing:
  coding:        { preferred_finalizer: model_a }
  long_context:  { preferred_finalizer: model_b }
  default:       { preferred_finalizer: winner }
```

---

## 14. Data Model

Same core entities as v1 (`TaskRun`, `Candidate`, `Evaluation`, `TraceEvent`), plus:

### ValidatorResult
```json
{
  "validator_result_id": "uuid",
  "candidate_id": "uuid",
  "kind": "tests | json_schema | regex | function",
  "pass": true,
  "signals": {
    "failed_tests": ["string"],
    "schema_path": "string",
    "stdout": "string",
    "stderr": "string"
  },
  "duration_ms": 0
}
```

### ConfigSnapshot
```json
{
  "run_id": "uuid",
  "config_hash": "sha256",
  "config_yaml": "string"
}
```
(Ensures every trace is reproducible.)

---

## 15. Failure Handling

Same catalog as v1 with responses refined:
- Invalid JSON with constrained decoding on → bug in backend, log and fail the candidate (do not retry blindly).
- Invalid JSON without constrained decoding → repair parser → one retry → discard.
- Evaluator praise collapse → detected in calibration; apply stricter prompt and score normalization per judge.
- Score saturation → drop to pairwise earlier; if still saturated, widen strategy axes.
- Model collapse across candidates → increase temperature, rotate strategy axes.

---

## 16. Security and Safety

Same as v1. Plus: the calibration oracle, if configured, must never be called outside `calibration` flows. A runtime call to an external endpoint is a hard failure.

---

## 17. Phased Plan (with exit criteria)

### Phase 1 — Spine
- Normalizer (with task-class tagging), rubric builder, proposal stage, deterministic validation, cross-eval stage (non-verifiable path only), two-step pruner, synthesis.
- Backend: vLLM or llama.cpp.
- KV prefix reuse enabled.
- Trace store + baseline-comparison report.

**Exit:** harness beats single-shot baseline by ≥5% on one task class; trace queryable.

### Phase 2 — Gated repair + finalist pairwise
- Targeted repair (gated).
- Pairwise finals (gated by ε).
- Disagreement flagging.
- `repair_noop_rate` tracked.

**Exit:** repair shows ≥3% uplift on at least one class; pairwise shows swap-order stability ≥0.8.

### Phase 3 — Calibration + experiments
- Dev-time oracle calibration flow.
- Ablation runner (stage on/off).
- Cost-quality frontier report per task class.

**Exit:** calibrated config deployed; each stage has documented marginal-gain evidence or is disabled.

---

## 18. Suggested Code Layout

```text
harness/
  app.py
  config.py
  types.py
  pipeline/
    normalize.py
    rubric.py
    propose.py
    validate.py          # NEW: deterministic validators dispatcher
    evaluate.py
    prune.py
    repair.py            # gated logic lives here
    finalize.py          # pairwise + synthesis entry
  backends/
    base.py
    capabilities.py      # NEW: capability negotiation
    vllm.py
    llamacpp.py
    sglang.py
    openai_compat.py
  prompts/
    registry.py
    templates/
  validation/
    tests.py             # run unit tests
    json_schema.py
    regex.py
    function.py
  ranking/
    aggregate.py
    pairwise.py
  scheduler/
    queue.py
    residency.py         # NEW: swap vs coresident policies
  storage/
    traces.py
    sqlite_store.py
  metrics/
    collector.py
    reports.py
    baselines.py         # NEW: single-shot baseline runner
  calibration/           # NEW: dev-time only
    oracle_client.py
    calibration_set.py
    judge_bias.py
  experiments/
    benchmark.py
    ablation.py
```

---

## 19. v2 Non-Negotiables

- Every run reports a single-shot baseline alongside harness output.
- Deterministic validation runs before any LLM judgment when a verifier exists.
- Backend must negotiate capabilities; missing KV prefix cache is a warning, missing constrained decoding is a fallback.
- Repair and pairwise stages are gated, not blanket.
- Oracle is dev-time only; a runtime oracle call is a hard failure.
- Config is a single YAML; trace stores its hash.
- Stages with no measured uplift are disabled per task class.

---

## 20. On the External-Oracle Question

The user's instinct is right: calling a frontier model at runtime defeats the thesis. The design answer is a clean boundary.

- **Runtime:** local-only. No exceptions. This is what the thesis measures.
- **Dev-time:** a strong external model may be used to calibrate local judges, the way you'd use a reference measurement to calibrate a meter. The result is a config, not a dependency.

The question "how much intelligence can we squeeze out of small local models?" is not the same as "can we discover good local-model orchestration without ever consulting a better model?" The first is the product claim; the second is a self-imposed handicap that would slow learning without changing the product.

If you want the stricter version — no oracle ever, even for calibration — the harness still works. Calibration falls back to human-labeled preference pairs (slower, noisier, but still dev-time-only).

---

## 21. Open Questions (narrowed)

v1 left many questions open. v2 resolves most by making them configurable with defaults. Remaining genuine unknowns:

1. On which task classes does pairwise beat aggregate scoring? (Measure.)
2. Is there a better diversity-inducing axis than the enumerated six? (Iterate.)
3. At what VRAM budget does `coresident` stop being viable and `swapped` win on throughput? (Hardware-dependent; measure on target machine.)
4. Can the repair prompt be made class-aware enough to beat a blanket-disabled baseline, or is gating better? (Phase 2 exit measures this.)
5. Is the cost of 6–8 candidates justified versus 2–3 plus more repair rounds? (Ablation.)

---

## 22. Summary

v2 is v1 with the judge made honest, the hardware made explicit, and every stage required to prove its worth. The pipeline stays: normalize → propose → validate → score → prune → (repair if flawed) → (pairwise if close) → synthesize. What changed is that deterministic signal dominates when it exists, LLM judgment is minimized and calibrated, and each stage carries its own uplift receipt.

The thesis is measurable: a harness run always reports its single-shot baseline alongside its final answer. If the delta is zero, the harness is not doing its job.
