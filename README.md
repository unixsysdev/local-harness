# local-harness

A local test-time-compute harness that runs two small-ish local LLMs as competing proposal generators and cross-evaluators, plus an evaluator that measures whether the harness actually beats a single-shot baseline on the same hardware, with judge-bias metrics exposed.

Not a chatbot framework, not an agent, not a roleplay. A structured inference pipeline with an attached measurement tool.

---

## TL;DR

> Batched local throughput is useful as a **variance reducer** on tasks where the baseline is stochastic, and as a **constraint enforcer** on tasks where the baseline routinely violates a checkable spec. It is *not* a capability amplifier: tasks outside the model's reach stay outside reach regardless of candidate count. At 24× baseline compute cost, selective application beats blanket use.

Three runs, 42 total evaluated tasks, one abandoned experiment. Results below.

---

## Setup

| | |
|---|---|
| Machine | AMD Strix Halo (Radeon 8060S iGPU, gfx1151) |
| Memory | 126 GB unified LPDDR5X (≈ 256 GB/s) |
| Runtime | ROCm 7.2, llama.cpp build 8638 |
| Model A | Qwen3.6-35B-A3B (Q4_K_XL, ≈ 3 B active, 21 GB) |
| Model B | Gemma-4-26B-A4B (Q4_K_M, ≈ 4 B active, 16 GB) |
| Backend | 2× `llama-server` instances on ports 8080 / 8081, `--parallel 6 --kv-unified` |
| Pipeline | normalize → rubric → 12 proposals (4 strategies × 3 temps × 2 models) → deterministic + soft-constraint validators → cross-model evaluation → prune → gated repair → synth |

Both models are MoE with small active-parameter counts. Per-stream decode is ≈ 45 tps solo and aggregate ≈ 150 tps at the n=6 parallel knee.

---

## Results

### Run 1 — 8 tasks (3 coding + 2 extraction + 3 writing)

| outcome | n | notes |
|---|---|---|
| harness_win | 3 | all writing tasks with word-count and must-contain constraints |
| baseline_win | 0 | — |
| both_pass / both_fail | 5 | verifiable tasks baseline already solved |

Harness 3 / baseline 0. Directional but thin.

### Run 2 — 31 tasks (gated repair + judge calibration enabled)

| class | n | harness_win | baseline_win | ties / both |
|---|---|---|---|---|
| coding (verifiable) | 10 | 0 | 0 | 9 both_pass, 1 both_fail |
| extraction (verifiable) | 11 | 0 | 0 | 11 both_pass |
| writing (non-verifiable) | 10 | 7 | 2 | 1 tie |
| **total** | **31** | **7** | **2** | **22** |

Harness 7 / baseline 2 / ties 22. Win rate 22.6 %, loss rate 6.5 %. Cost ratio 24.5× baseline wall time (108.6 s vs 4.4 s).

**Judge calibration corrected one Run-1 false positive.** `writing_decline_email` was `harness_win` in Run 1; after down-weighting `model_b` to 0.6 (observed 60 % positional bias toward label "A"), the same task became `tie`. Calibration cost a pseudo-win but delivered honest ones.

### Run 3 — 3 hard tasks, thinking on both sides (Phase 2 abandoned)

Triage of 13 candidate "hard" tasks with `thinking=true` found that **10 were already solved single-shot by both models**. Famous riddles (Monty Hall, bat-and-ball, lily-pond) and most classic LeetCode problems fell easily. Only 3 tasks survived triage.

**Phase 1 completed — baseline × 3 samples per (task, model), 18 runs total:**

| task | model_a | model_b |
|---|---|---|
| calculator | 1/3 = 33 % | 2/3 = **67 %** |
| valid_number | 1/3 = 33 % | 1/3 = 33 % |
| word_break | 0/3 = 0 % | 0/3 = 0 % |

- `calculator` and `valid_number` are **stochastic**: identical prompt, same model, same temperature yield both passing and failing runs. On `valid_number / model_a`, one sample converged in 8381 tokens; another spiraled to 16 384 tokens without finding the answer.
- `word_break` shows a **hard capability ceiling**: 0/6 across both models with unlimited thinking budget.

**Phase 2 (full harness on the 3 tasks) was killed.** The first Phase 2 run hit `Context size has been exceeded` because `proposal_max_tokens=32000 × 6 concurrent slots` exceeded the 49 152-token KV buffer. Configuration was corrected, but by then it was clear Phase 2 would mostly confirm what Phase 1 + probability already predicted:

> With 12 independent candidates and a 67 % per-shot success rate, P(≥ 1 passes) ≈ 99.99 %. With 0 % per-shot, no amount of search rescues it. Phase 2 would verify exactly this and add no new story worth the additional compute after a long sequence of infrastructure bugs.

---

## Where the harness earns its compute, and where it doesn't

| regime | evidence | defensible claim |
|---|---|---|
| Constrained non-verifiable writing (word count, must-contain, format rules) | Run 1 3/3, Run 2 7/10 | Strong selective fit; harness wins on constraint compliance |
| Verifiable tasks baseline solves single-shot | Run 2 0/21 wins | No help; wasted 24× compute. Use baseline-short-circuit in production. |
| Stochastic-failure tasks within model capability | Run 3 Phase 1 (no Phase 2) | **Probabilistic argument supports it**, no direct empirical confirmation in this repo |
| Tasks beyond model capability | Run 3: word_break 0/6 | Harness cannot help. Search is not a capability amplifier. |

---

## What Run 3 was meant to prove and didn't

- **Meant to prove:** that the harness empirically beats baseline on stochastic-failure tasks.
- **Actually proved:** baseline stochasticity exists (Phase 1 pass rates of 33–67 % on identical prompts), plus a hard capability ceiling (0 % on `word_break`).
- **Did not prove:** the harness beats baseline empirically on these tasks. Phase 2 was abandoned before producing that data.
- **Did not prove:** that search + filter is preferable to self-consistency N=3 at equal compute. Open question.

---

## Honest limits of this work

1. **Difficulty calibration by intuition was wrong.** 10 of 12 hand-crafted "hard" tasks turned out to be single-shot solvable with thinking on. Future difficulty should come from calibrated benchmarks (HumanEval+, MBPP+, LiveCodeBench), not from my guess.
2. **Sample sizes are small.** 31 tasks in Run 2 and 3 tasks in Run 3 are too few to support strong statistical claims. Win rates carry large uncertainty bars.
3. **Judge noise is real.** `model_b` shows measurable positional bias; `model_a` is cleaner but not noise-free. Calibration reduces this but doesn't eliminate it.
4. **No frontier-oracle calibration.** The v2 spec describes using a frontier model at dev-time to calibrate local judges per rubric dimension. That's not implemented here — calibration is purely data-driven from observed bias.
5. **Infrastructure bugs ate compute.** Client timeouts, max_tokens-vs-context, prompt spoilers, sequential triage — each a real defect found during runs rather than designed out. Time-to-signal on Run 3 would have been minutes instead of hours had these been anticipated.

---

## Conclusions

1. **Selective deployment beats blanket use.** For constrained writing, the harness is worth the 24× compute. For verifiable tasks the baseline already passes, short-circuit to baseline. For tasks past the model's capability, neither helps.
2. **Stochasticity is the harness's natural friend.** Run 3 Phase 1 confirmed baseline single-shot is not deterministic on hard-but-reachable tasks. The mathematics of independent samples predicts search should help in this regime; empirical verification remains future work.
3. **The evaluator is the more enduring artifact.** It caught a Run-1 false positive via calibration in Run 2. That self-measurement discipline is arguably more valuable than the harness itself.

---

## Reproduce

```bash
bash scripts/start_servers.sh                  # ROCm llama.cpp, both models
/usr/bin/python3 -m venv .venv && .venv/bin/pip install httpx pydantic pyyaml

# Single task end-to-end (baseline + harness):
PYTHONPATH=. .venv/bin/python -m harness.app \
  --task tasks/coding_is_prime.json --config config.yaml

# Full evaluator across a task directory:
PYTHONPATH=. .venv/bin/python -m harness.evaluator \
  --tasks tasks --config config.yaml --output eval_report.json

bash scripts/stop_servers.sh
```

---

## Repository layout

```text
harness/                  pipeline, backends, evaluator, soft-constraint validator
scripts/                  start/stop, benchmarks, triage, reruns, experiment drivers
tasks/                    31-task eval set (Runs 1+2)
tasks_r3/, tasks_r3_sweet/  Run 3 candidate + sweet-spot tasks
config.yaml               default Runs 1+2 config
config_r3.yaml            Run 3 config (thinking on, standard token caps)
config_r3_unlimited.yaml  Run 3 config (unlimited-ish caps; what should have been default for thinking)
traces/                   one JSONL per harness run
logs/                     server logs + experiment logs
baseline_triage.json      Run 3 per-task disposition
r3_baseline_n3.json       Run 3 Phase 1 data (18 baseline runs)
eval_report.json          Run 1 output
eval_report_v2.json       Run 2 output
```
