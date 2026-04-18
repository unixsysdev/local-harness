"""Pydantic models used across the harness."""
from __future__ import annotations
from typing import Any, Literal, Optional
from uuid import uuid4
from pydantic import BaseModel, Field


def _uuid() -> str:
    return uuid4().hex[:12]


TaskClass = Literal["verifiable", "non_verifiable", "mixed"]
DeliverableType = Literal["code", "text", "json", "plan", "mixed"]
RiskLevel = Literal["low", "medium", "high"]


class VerifierSpec(BaseModel):
    kind: Literal["tests", "json_schema", "regex", "function", "none"] = "none"
    artifact: Optional[str] = None        # path, schema ref, regex, function name
    inline: Optional[dict[str, Any]] = None  # inline schema or test code


class WordCountConstraint(BaseModel):
    min: int
    max: int


class SoftConstraints(BaseModel):
    """Deterministic, non-blocking checks applied to any task class.

    Unlike the primary verifier (which is pass/fail for the verifiable path),
    soft constraints yield a score in [0, 1] that penalizes the combined score
    so candidates violating stated constraints are deprioritized.
    """
    word_count: Optional[WordCountConstraint] = None
    must_contain: list[str] = []           # substrings that MUST appear
    must_not_contain: list[str] = []       # substrings that MUST NOT appear


class Task(BaseModel):
    task_id: str = Field(default_factory=_uuid)
    raw_input: str
    task_class: TaskClass = "non_verifiable"
    task_type: str = "other"
    deliverable_type: DeliverableType = "text"
    success_criteria: list[str] = []
    constraints: list[str] = []
    risk_level: RiskLevel = "low"
    verifier: VerifierSpec = VerifierSpec()
    soft_constraints: Optional[SoftConstraints] = None


class Rubric(BaseModel):
    task_summary: str
    weights: dict[str, float]
    failure_modes: list[str] = []
    recommended_strategy_axes: list[str] = []
    max_candidate_tokens: int = 800


class Candidate(BaseModel):
    candidate_id: str = Field(default_factory=_uuid)
    model: str                               # model_a | model_b
    strategy: str
    temperature: float
    seed: Optional[int] = None
    solution: str
    assumptions: list[str] = []
    known_risks: list[str] = []
    self_confidence: float = 0.5
    # Backend metadata
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    latency_ms: int = 0


class ValidatorResult(BaseModel):
    candidate_id: str
    kind: str
    passed: bool
    score: float = 0.0           # 1.0 pass, 0.0 hopeless, partial in between
    signals: dict[str, Any] = {}
    duration_ms: int = 0


class SoftScore(BaseModel):
    """Soft-constraint compliance score, applied to all task classes when present."""
    candidate_id: str
    score: float = 1.0           # 1.0 = all constraints satisfied, <1.0 = violations
    violations: list[str] = []
    signals: dict[str, Any] = {}


class ScoreBreakdown(BaseModel):
    correctness: float = 0.0
    completeness: float = 0.0
    constraint_compliance: float = 0.0
    clarity: float = 0.0
    efficiency: float = 0.0
    risk: float = 0.0


class Evaluation(BaseModel):
    evaluation_id: str = Field(default_factory=_uuid)
    candidate_id: str
    evaluator_model: str          # model_a | model_b | deterministic
    stage: str                    # cross_eval | self_check | deterministic
    scores: ScoreBreakdown = ScoreBreakdown()
    aggregate: float = 0.0
    fatal_issues: list[str] = []
    minor_issues: list[str] = []
    repair_suggestions: list[str] = []
    raw: Optional[str] = None


class RunResult(BaseModel):
    run_id: str = Field(default_factory=_uuid)
    task: Task
    rubric: Optional[Rubric] = None
    candidates: list[Candidate] = []
    evaluations: list[Evaluation] = []
    validator_results: list[ValidatorResult] = []
    finalist_id: Optional[str] = None
    final_answer: Optional[str] = None
    baseline_answer: Optional[str] = None
    metrics: dict[str, Any] = {}
