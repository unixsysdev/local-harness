"""Load & validate the single YAML config file."""
from __future__ import annotations
from pathlib import Path
import yaml
from pydantic import BaseModel


class ModelCfg(BaseModel):
    endpoint: str
    served_name: str
    thinking_default: bool = False


class BaselineCfg(BaseModel):
    model: str
    temperature: float = 0.7
    max_tokens: int = 1500
    thinking: bool = True


class SearchClassCfg(BaseModel):
    proposals_per_model: int = 3
    post_score_keep: int = 3


class SearchCfg(BaseModel):
    non_verifiable: SearchClassCfg = SearchClassCfg()
    verifiable: SearchClassCfg = SearchClassCfg()


class SamplingCfg(BaseModel):
    strategies: list[str]
    temperatures: list[float]
    top_p: float = 0.95
    proposal_max_tokens: int = 800
    eval_max_tokens: int = 400
    final_max_tokens: int = 1500


class ScoringClassCfg(BaseModel):
    deterministic_weight: float = 0.0
    cross_eval_weight: float = 0.83
    self_eval_weight: float = 0.17
    self_eval_enabled: bool = False


class ScoringCfg(BaseModel):
    non_verifiable: ScoringClassCfg = ScoringClassCfg()
    verifiable: ScoringClassCfg = ScoringClassCfg(
        deterministic_weight=0.70, cross_eval_weight=0.25, self_eval_weight=0.05
    )


class DisagreementCfg(BaseModel):
    tau: float = 0.25


class RepairCfg(BaseModel):
    enabled: bool = True
    max_to_repair: int = 4
    max_tokens: int = 700
    viability_floor: float = 0.1
    auto_disable_if_noop_rate_above: float = 0.5


class ShortCircuitCfg(BaseModel):
    """Skip the harness when baseline already passes the verifier. Off in eval mode."""
    enabled: bool = False


class JudgeCalibrationCfg(BaseModel):
    """Per-judge vote weighting. Values < 1.0 down-weight this judge."""
    weights: dict[str, float] = {}
    min_consistency_for_full_weight: float = 0.8
    non_unanimous_penalty: float = 0.5  # applied when judge was biased on prior runs


class SynthesisCfg(BaseModel):
    model: str = "model_a"
    thinking: bool = True
    temperature: float = 0.5


class RunCfg(BaseModel):
    name: str = "default"
    trace_dir: str = "traces"
    tasks_dir: str = "tasks"


class Config(BaseModel):
    run: RunCfg
    models: dict[str, ModelCfg]
    baseline: BaselineCfg
    search: SearchCfg
    sampling: SamplingCfg
    scoring: ScoringCfg
    disagreement: DisagreementCfg = DisagreementCfg()
    repair: RepairCfg = RepairCfg()
    short_circuit: ShortCircuitCfg = ShortCircuitCfg()
    judge_calibration: JudgeCalibrationCfg = JudgeCalibrationCfg()
    synthesis: SynthesisCfg

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        with open(path, "r") as fh:
            data = yaml.safe_load(fh)
        return cls.model_validate(data)
