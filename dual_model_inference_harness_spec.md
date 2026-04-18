# Dual-Model Local Inference Harness Spec

## 1. Purpose

Build a local orchestration harness that uses two LLMs as competing proposal engines and cross-verifiers. The system should maximize answer quality on a constrained local machine by converting throughput into test-time search, verification, and repair.

The harness is not a chatbot framework and not a roleplay multi-agent system. It is a structured inference pipeline built around:

- parallel candidate generation
- cross-model evaluation
- targeted repair
- aggressive pruning
- final synthesis

Primary design goal:

> Spend high-throughput batch capacity on breadth and filtering, then spend low-latency serial capacity only on finalists.

---

## 2. Core Principles

### 2.1 Breadth before depth
Use batched generation to explore multiple candidate solutions in parallel. Do not allow long conversational loops between workers.

### 2.2 Structured artifacts over transcripts
Every stage must exchange compact, machine-readable outputs. JSON is the default interchange format.

### 2.3 Cross-model disagreement is signal
The harness should not assume either model is the permanent judge. It should explicitly measure disagreement and use it to trigger repair or escalation.

### 2.4 Hard token budgets
All stages except final synthesis must operate under strict output caps.

### 2.5 Prune early
Do not carry weak candidates forward. The system should reduce search width at each phase.

### 2.6 Task routing is dynamic
The harness may prefer one model over the other depending on task type, but permanent hardcoding should be avoided. Routing should be configurable and data-driven.

---

## 3. System Scope

### 3.1 In scope
- prompt orchestration
- batched candidate generation
- rubric-driven scoring
- cross-model critique
- pairwise tournament ranking
- candidate repair
- final synthesis
- metrics collection
- experiment configuration
- pluggable model backends

### 3.2 Out of scope for v1
- autonomous tool use
- browser integration
- distributed multi-machine scheduling
- long-term memory across projects
- fine-tuning
- reinforcement learning
- complex DAG optimizers

---

## 4. High-Level Architecture

```text
User Task
   |
   v
Task Normalizer
   |
   v
Planner / Rubric Builder
   |
   v
Proposal Stage (Model A + Model B, batched)
   |
   v
Initial Scoring Stage (cross-model + self-check)
   |
   v
Pruner
   |
   v
Adversarial Repair Stage
   |
   v
Tournament / Reranker
   |
   v
Final Synthesizer
   |
   v
Final Output + Metrics + Trace
```

Supporting modules:
- backend adapters
- scheduler
- prompt registry
- schema validator
- trace store
- metrics collector
- experiment config loader

---

## 5. Execution Model

### 5.1 Models
The harness must support at least two models:
- Model A (example: Qwen)
- Model B (example: Gemma)

Each model may act as:
- proposer
- critic
- verifier
- synthesizer

Roles are stage-specific, not permanent.

### 5.2 Runtime modes
The harness supports two execution modes:

#### Batch mode
Use a high-throughput backend for:
- proposal generation
- rubric scoring
- pairwise comparison
- repair generation

#### Interactive mode
Use a low-latency backend for:
- final synthesis
- human-facing answer formatting

### 5.3 Search width
Configurable per task class.

Suggested defaults:
- proposals per model: 4
- total initial candidates: 8
- post-score survivors: 4
- post-repair survivors: 2
- finalists: 1

---

## 6. Pipeline Stages

## 6.1 Stage 0: Task Normalization

### Goal
Convert raw user input into a normalized task object.

### Input
Raw prompt.

### Output schema
```json
{
  "task_id": "uuid",
  "raw_input": "string",
  "task_type": "coding|analysis|planning|writing|extraction|other",
  "success_criteria": ["string"],
  "constraints": ["string"],
  "deliverable_type": "code|text|json|plan|mixed",
  "risk_level": "low|medium|high"
}
```

### Notes
This stage may be heuristic or model-assisted. Prefer deterministic parsing when possible.

---

## 6.2 Stage 1: Planner / Rubric Builder

### Goal
Produce a compact rubric used by downstream scoring and repair.

### Inputs
Normalized task object.

### Outputs
```json
{
  "task_summary": "string",
  "evaluation_rubric": {
    "correctness": 0.40,
    "completeness": 0.20,
    "constraint_compliance": 0.15,
    "clarity": 0.10,
    "efficiency": 0.10,
    "risk": 0.05
  },
  "failure_modes": ["string"],
  "recommended_strategy_axes": ["string"],
  "max_candidate_tokens": 400
}
```

### Requirements
- Sum of rubric weights must equal 1.0
- Output must be JSON only
- Must identify likely failure modes explicitly

---

## 6.3 Stage 2: Proposal Generation

### Goal
Generate diverse candidate solutions from both models.

### Inputs
- normalized task
- rubric
- strategy axis
- seed / temperature config

### Diversity controls
Candidate diversity should be induced by varying:
- temperature
- seed
- strategy prompt
- decomposition style
- brevity constraint
- evidence emphasis

### Requirements
- No candidate should see sibling candidates
- Candidates must be independent
- Context must include only task + rubric + strategy instructions

### Candidate output schema
```json
{
  "candidate_id": "uuid",
  "model": "model_a|model_b",
  "strategy": "string",
  "solution": "string",
  "assumptions": ["string"],
  "known_risks": ["string"],
  "self_confidence": 0.0,
  "estimated_failure_probability": 0.0
}
```

### Storage
All candidates must be persisted with metadata:
- prompt hash
- seed
- temperature
- backend
- latency
- input tokens
- output tokens

---

## 6.4 Stage 3: Initial Evaluation

### Goal
Evaluate candidate quality using both cross-model scoring and lightweight self-checks.

### Evaluation modes

#### Cross-model rubric score
Model A scores Model B candidates.
Model B scores Model A candidates.

#### Optional self-check
Same model scores its own candidates using a stricter checklist.
This score should have lower weight than cross-model score.

#### Deterministic validators
Used when available:
- JSON schema validation
- code compiles
- tests pass
- output length within budget
- required fields present

### Scoring output schema
```json
{
  "candidate_id": "uuid",
  "evaluator_model": "model_a|model_b|deterministic",
  "score_breakdown": {
    "correctness": 0.0,
    "completeness": 0.0,
    "constraint_compliance": 0.0,
    "clarity": 0.0,
    "efficiency": 0.0,
    "risk": 0.0
  },
  "aggregate_score": 0.0,
  "fatal_issues": ["string"],
  "minor_issues": ["string"],
  "repair_suggestions": ["string"]
}
```

### Aggregation
Compute:
- cross_score_mean
- cross_score_variance
- self_score_mean
- deterministic_penalties
- disagreement_index

Suggested combined score:

```text
combined_score =
  0.65 * cross_score_mean +
  0.10 * self_score_mean +
  0.25 * deterministic_score -
  penalty_terms
```

Where penalty terms include:
- schema failure
- missing constraints
- hallucination markers
- excessive verbosity

### Disagreement index
```text
disagreement_index = abs(score_model_a - score_model_b)
```

High disagreement candidates should be flagged for special handling.

---

## 6.5 Stage 4: Pruning

### Goal
Reduce candidate pool aggressively.

### Default rule
Keep top N by combined score, but include at least one high-disagreement candidate if it is above a minimum viability threshold.

### Suggested default
- input: 8 candidates
- output: 4 candidates

### Rationale
Sometimes a polarizing candidate is novel but underappreciated by one evaluator. Preserve one such candidate when justified.

---

## 6.6 Stage 5: Adversarial Repair

### Goal
Use the opposite model to attack and improve each surviving candidate.

### Inputs
- candidate
- evaluation trace
- repair suggestions
- rubric

### Rule
Model A repairs Model B candidates.
Model B repairs Model A candidates.

### Repair output schema
```json
{
  "candidate_id": "uuid",
  "repair_model": "model_a|model_b",
  "repaired_solution": "string",
  "changes_made": ["string"],
  "issues_addressed": ["string"],
  "remaining_uncertainties": ["string"]
}
```

### Requirements
- Must preserve valid content where possible
- Must not rewrite blindly without referencing specific detected flaws
- Must remain under token budget

---

## 6.7 Stage 6: Tournament Ranking

### Goal
Select finalists using pairwise comparisons rather than only absolute scores.

### Method
For all repaired survivors, run pairwise comparisons.

Comparison prompt must ask evaluator to choose:
- winner
- why winner is better under rubric
- confidence in decision
- whether merge is superior to either standalone

### Pairwise output schema
```json
{
  "left_candidate_id": "uuid",
  "right_candidate_id": "uuid",
  "evaluator_model": "model_a|model_b",
  "winner": "left|right|merge",
  "confidence": 0.0,
  "reason": "string"
}
```

### Ranking algorithm
Recommended for v1:
- Bradley-Terry approximation or simple win count
- confidence-weighted wins

Suggested simple score:
```text
tournament_score = sum(confidence_weighted_pairwise_wins)
```

---

## 6.8 Stage 7: Final Synthesis

### Goal
Produce the final user-facing answer or artifact.

### Inputs
- top finalist or merged pair
- rubric
- key issues resolved
- unresolved risks

### Synthesizer routing
By default choose:
- winner of tournament
or
- preferred model for task class

### Output
Free-form final response or structured artifact depending on task.

### Requirements
- no internal scoring jargon
- no chain-of-thought exposure
- incorporate best repaired content
- explicitly mention unresolved uncertainty when material

---

## 7. Routing Policy

The harness must support pluggable routing rules.

### Example v1 routing
```json
{
  "coding": {
    "primary_generator": "model_a",
    "secondary_generator": "model_b",
    "preferred_finalizer": "model_a"
  },
  "long_context_analysis": {
    "primary_generator": "model_b",
    "secondary_generator": "model_a",
    "preferred_finalizer": "model_b"
  },
  "general_reasoning": {
    "primary_generator": "both",
    "secondary_generator": "both",
    "preferred_finalizer": "winner"
  }
}
```

Routing policy must be configurable by file, not hardcoded.

---

## 8. Prompting Contract

All prompts must be versioned and named.

Required prompt families:
- `task_normalizer_v1`
- `rubric_builder_v1`
- `proposal_v1`
- `cross_evaluator_v1`
- `self_check_v1`
- `repair_v1`
- `pairwise_ranker_v1`
- `final_synthesizer_v1`

Each prompt should specify:
- allowed inputs
- expected output schema
- max output tokens
- failure behavior

### Prompt rules
- Prefer terse instructions
- Demand JSON only in all non-final stages
- Forbid long explanations unless requested by schema
- Explicitly reject unsupported assumptions

---

## 9. Data Model

## 9.1 Core entities

### TaskRun
```json
{
  "run_id": "uuid",
  "task_id": "uuid",
  "started_at": "timestamp",
  "config_id": "string",
  "status": "running|completed|failed",
  "task_type": "string"
}
```

### Candidate
```json
{
  "candidate_id": "uuid",
  "run_id": "uuid",
  "origin_model": "string",
  "origin_stage": "proposal|repair|merge",
  "parent_candidate_id": "uuid|null",
  "content": "string",
  "metadata": {}
}
```

### Evaluation
```json
{
  "evaluation_id": "uuid",
  "candidate_id": "uuid",
  "evaluator": "string",
  "stage": "cross_eval|self_check|pairwise",
  "scores": {},
  "issues": {},
  "raw_output": {}
}
```

### TraceEvent
```json
{
  "event_id": "uuid",
  "run_id": "uuid",
  "stage": "string",
  "backend": "string",
  "model": "string",
  "latency_ms": 0,
  "input_tokens": 0,
  "output_tokens": 0,
  "status": "ok|error",
  "payload_ref": "string"
}
```

---

## 10. Backend Abstraction

The harness must support multiple inference backends via adapters.

### Required adapter interface
```python
class InferenceBackend:
    def generate(self, requests: list[GenerationRequest]) -> list[GenerationResponse]:
        ...

    def health(self) -> BackendHealth:
        ...

    def model_info(self) -> ModelInfo:
        ...
```

### GenerationRequest
```python
@dataclass
class GenerationRequest:
    request_id: str
    model: str
    prompt: str
    temperature: float
    top_p: float
    max_tokens: int
    stop: list[str]
    seed: int | None
    json_schema: dict | None
    metadata: dict
```

### GenerationResponse
```python
@dataclass
class GenerationResponse:
    request_id: str
    model: str
    text: str
    finish_reason: str
    input_tokens: int
    output_tokens: int
    latency_ms: int
    error: str | None
    metadata: dict
```

### v1 adapters
- OpenAI-compatible HTTP adapter
- vLLM adapter
- llama.cpp server adapter
- optional SGLang adapter

---

## 11. Scheduler

### Goals
- batch compatible requests together
- avoid mixing incompatible schemas in the same logical batch unit
- enforce stage concurrency limits
- collect latency and throughput metrics

### Features
- request queue per stage
- request coalescing
- timeout handling
- retry policy
- backpressure limits

### Default policies
- max retries: 1 for transient backend error
- no retry for invalid JSON unless repair parser is enabled
- stage timeout configurable

---

## 12. Validation Layer

### Required validators
- JSON parse validator
- JSON schema validator
- empty output validator
- forbidden field validator
- token budget validator
- task-specific deterministic validators where available

### Repair parser
Optional module that attempts:
- extraction of first valid JSON object
- bracket balancing
- trivial cleanup

If repair parser fails, request is marked invalid.

---

## 13. Metrics

The harness must record both system metrics and quality metrics.

### 13.1 System metrics
- wall clock latency per stage
- input tokens / output tokens
- tokens per second per backend
- batch size distribution
- failure rate by stage
- JSON validity rate
- backend timeout rate

### 13.2 Quality metrics
- final acceptance score
- task-specific pass/fail
- judge agreement rate
- disagreement index distribution
- repair uplift
- pairwise stability
- best-of-N gain over single-shot baseline

### 13.3 Derived analysis
- marginal gain per extra candidate
- gain per extra repair round
- cost-quality frontier
- model A vs model B win rates by task class

---

## 14. Configuration

Use file-based config, preferably YAML.

### Example
```yaml
run:
  task_type_default: general_reasoning
  trace_dir: ./traces

models:
  model_a:
    name: qwen
    backend: vllm
  model_b:
    name: gemma
    backend: vllm

search:
  proposals_per_model: 4
  survivors_after_prune: 4
  finalists: 2

sampling:
  temperatures: [0.2, 0.5, 0.8, 1.0]
  top_p: 0.95

budgets:
  proposal_max_tokens: 400
  eval_max_tokens: 250
  repair_max_tokens: 300
  final_max_tokens: 1200

scoring:
  cross_eval_weight: 0.65
  self_eval_weight: 0.10
  deterministic_weight: 0.25
  disagreement_keep_threshold: 0.25

routing:
  coding:
    preferred_finalizer: model_a
  long_context_analysis:
    preferred_finalizer: model_b
```

---

## 15. Failure Handling

### Common failures
- invalid JSON
- backend timeout
- empty response
- repeated hallucinated assumptions
- evaluator collapse into generic praise
- score saturation with poor discrimination

### Responses
- invalid JSON: validator fail, optional repair parse, else discard
- timeout: retry once if transient
- score saturation: inject pairwise comparison stage earlier
- evaluator verbosity: tighten prompt and lower max tokens
- model collapse: change strategy axis / temperature / seed

---

## 16. Security and Safety

### Requirements
- no execution of arbitrary code in v1 unless sandboxed separately
- prompt templates stored locally and versioned
- trace store may contain sensitive prompts; keep local-only by default
- final answer must not expose internal rubric or hidden deliberation unless explicitly requested

---

## 17. Reference Implementation Plan

## Phase 1: Minimal working harness
Implement:
- task normalizer
- rubric builder
- proposal stage
- cross-eval stage
- prune stage
- final synthesis

Skip initially:
- repair stage
- tournament ranking
- advanced routing

Success criterion:
- end-to-end run completes on a sample task
- structured traces saved
- candidates ranked successfully

## Phase 2: Add repair and tournament
Implement:
- adversarial repair
- pairwise ranker
- disagreement-aware pruning

Success criterion:
- repaired candidates outperform proposal baseline on at least one task subset

## Phase 3: Add metrics and experiments
Implement:
- benchmark harness
- ablation runner
- report generation

Success criterion:
- compare single-shot vs dual-model search quantitatively

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
    evaluate.py
    prune.py
    repair.py
    tournament.py
    finalize.py
  backends/
    base.py
    openai_compat.py
    vllm.py
    llamacpp.py
    sglang.py
  prompts/
    registry.py
    templates/
  validation/
    json_check.py
    schema_check.py
    deterministic.py
  ranking/
    aggregate.py
    pairwise.py
  storage/
    traces.py
    sqlite_store.py
  metrics/
    collector.py
    reports.py
  experiments/
    benchmark.py
    ablation.py
```

---

## 19. v1 Non-Negotiables

- JSON-only for non-final stages
- independent proposal generation
- cross-model scoring
- deterministic validation hooks
- aggressive pruning
- full traceability
- config-driven search widths and budgets
- backend abstraction from day one

---

## 20. Immediate Next Step

Implement the minimal v1 loop:

1. normalize task
2. build rubric
3. generate 4 proposals from each model
4. cross-score all proposals
5. prune to top 3 or 4
6. synthesize final answer from top candidate
7. store trace and metrics

Only after that works should repair and tournament logic be added.

---

## 21. Open Questions for Implementation

These should be left configurable, not assumed:

- exact candidate counts per task type
- score aggregation formula
- whether self-eval helps or hurts
- whether pairwise ranking beats direct scoring on your workloads
- whether both models should stay resident simultaneously
- whether final synthesis should always be routed to the tournament winner

---

## 22. Summary

This harness is a local test-time compute engine, not a conversational agent framework. It treats models as competing proposal generators and critics, uses disagreement as signal, and converts throughput into answer quality through structured search, repair, and selection.

