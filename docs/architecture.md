# ARGOS architecture

This document describes the ARGOS architecture and where each part lives in
the code. The references below name files and functions.

## Components

| Component | Role | Code |
|---|---|---|
| Request interface | Public HTTP API: submit requests, query request and cluster status, register nodes, start the loop | `src/argos/orchestrator/api.py` (`POST /submit-job`, `/cluster/status`, `/nodes/register`, `/orchestrator/start`) |
| Admission control | Validates a contract, computes the minimum node count required by the lower coverage bound, and admits, queues, or rejects | `src/argos/orchestrator/loop.py` (`OrchestrationLoop`) |
| Pending queue | Holds feasible requests until capacity frees up; ordered by priority, tenant fairness debt, and arrival | `src/argos/orchestrator/loop.py` |
| Per-request controller | One environment plus one policy per admitted request | `src/argos/orchestrator/environment.py`, `src/argos/orchestrator/rl/` |
| Runtime monitor | Polls `GET /metrics` on every node; aggregates CPU, memory, bandwidth, latency, and realized quality | `src/argos/orchestrator/loop.py` |
| Deployment manager | Maps target coverage to a node count, selects nodes by load, pushes node plans with `POST /configure` | `src/argos/orchestrator/loop.py`, `src/argos/orchestrator/coverage.py` |
| Runtime audit log | Append-only JSONL records of every lifecycle event, decision, configuration, violation, and error | `src/argos/orchestrator/persistence.py` |
| Control plane (optional) | Heartbeats, capacity-aware leader election, NATS events for multi-orchestrator deployments; not exercised in the evaluation | `src/argos/orchestrator/control_plane.py` |
| Node adapter | Worker-node API: applies node plans, runs the analytics, reports host and service telemetry | `src/argos/node.py` |
| Application runtime | Geographic heatmap over the Seville trajectories; computes spatial fidelity and hotspot recall | `src/argos/services/geo_heatmap.py` |

## Request lifecycle

1. The client submits a contract: accepted ranges for coverage, sample, and
   freshness, tenant, priority, CPU and memory caps, placement limits, and an
   optional cost budget.
2. The contract is validated and converted to a midpoint target configuration.
3. The orchestrator computes the minimum node count required by the lower
   coverage bound. With capacity available the request is placed; if it is
   feasible but capacity is exhausted it is queued; if its placement limits make
   the minimum impossible it is rejected with a reason.
4. Each admitted request gets its own environment and policy. When a queued
   request is admitted later, its controller is created at admission time; if a
   frozen policy was registered for it while it waited, that policy is loaded
   before the first decision (`OrchestrationLoop._initialize_rl_for_request`,
   `_apply_pending_policy_load`).
5. The decision loop adapts the request until it completes or expires; the
   request then releases its nodes and the queue is reconsidered.

## Decision model

**State.** Thirteen discretized dimensions: CPU, memory, and bandwidth
pressure; the position of coverage, sample, and freshness inside their accepted
ranges (normalized to [0, 1] and cut at 0.2, 0.4, 0.6, 0.8); service duty;
offered input load; active jobs; free capacity; pending demand; latency; and the
number of decision epochs with any violation in the last 20 epochs.
Q-learning uses the encoded tuple directly; DQN and PPO use its vector form.
Code: `src/argos/orchestrator/environment.py`.

**Actions.** Seven actions (`src/argos/orchestrator/rl/base.py`, `STANDARD_ACTIONS`):

| Action | Effect |
|---|---|
| `hold` | no change |
| `increase_coverage` / `decrease_coverage` | target coverage ± 0.05 |
| `increase_sample` / `decrease_sample` | target sample ± 0.05 |
| `increase_freshness` | update interval − 10 s (fresher, more work) |
| `decrease_freshness` | update interval + 10 s (staler, less work) |

Before each decision the environment masks actions that would have no effect
(the target is already at a contract bound) and actions that would add load
when placement capacity is exhausted or a node approaches its CPU or memory cap.
A final clamp keeps every target inside the accepted range.

**Reward.** A clipped linear score:

```
r = clip[-1, 1]( quality − resource − cost − range − overload − fairness − latency − capacity )
```

Quality is the mean of three normalized scores (coverage and sample are better
near the top of their range, freshness near the bottom of its interval range).
Resource pressure and cost share a unit penalty budget with weights 1.2 and 0.5;
overload contributes at most 0.5 each for CPU and memory; the three range checks
together at most 0.5; latency 0.35; capacity 0.30; fairness 0.20. Every
component is persisted with each decision (`reward_components` in
`rl_decisions.jsonl`).

**Controllers** (`src/argos/orchestrator/rl/`):

| File | Controller | Main parameters |
|---|---|---|
| `qlearning.py` | Q-learning | learning rate 0.1, discount 0.95, epsilon 0.15 → 0.01, optimistic init 3.0 |
| `dqn.py` | DQN | hidden size 64, replay 10,000, warm-up 32, target sync every 25 steps, Adam, smooth L1 loss |
| `ppo.py` | PPO | hidden size 64, 32-step rollouts, GAE λ 0.95, 4 epochs, minibatch 16, clip 0.2, value weight 0.5, entropy 0.01 |
| `static.py` | Static midpoint / Best-fixed | holds a fixed configuration (midpoint, or a selected candidate) |
| `threshold.py` | Threshold | reduces sample, then coverage, then freshness under pressure; restores after |
| `factory.py`, `selection.py` | | builds controllers by name; per-request selection |

## Placement and coverage

The deployment manager assigns ⌈coverage × N⌉ nodes (at least one, within the
request's placement limits), where N is the number of active nodes. Realized
coverage is the number of assigned nodes divided by N. With N = 3 it can only be
1/3, 2/3, or 1, so the realized value often differs from the continuous target.
Because the node count rounds up, realized coverage can exceed a contract
maximum (a target of 0.35 needs 2 nodes, realized 2/3, above a maximum of 0.45).
ARGOS records each such epoch as a coverage violation with `bound: "maximum"`.
All coverage violations recorded in the reference evaluation are of this kind;
none has `bound: "minimum"`.

## Runtimes: thread and process

A worker node executes the analytics in one of two modes, selected with the
environment variable `ARGOS_RUNTIME_EXECUTION_MODE` (`src/argos/node.py`):

- **thread**: the heatmap computation runs in threads of the node process.
- **process**: the computation is offloaded to a `ProcessPoolExecutor`. Worker
  processes are not limited by the Python global interpreter lock for
  CPU-bound work, but add process start-up and inter-process communication
  overhead.

The controlled evaluation runs every configuration in both modes. The live
evaluation uses the thread mode. `scripts/set_runtime_mode.py` switches
all nodes of a deployment over SSH and verifies the mode reported by each node.

## Frozen-policy guarantees

| Guarantee | Mechanism |
|---|---|
| No exploration | Controllers take a `training` flag in `select_action`; with `training=False`, DQN and Q-learning skip the epsilon branch and PPO uses masked argmax instead of sampling (`orchestrator/rl/dqn.py`, `orchestrator/rl/qlearning.py`, `orchestrator/rl/ppo.py`). The loop passes `training=self.config.rl_training`, which is false in evaluation mode (`src/argos/experiment.py`: `rl_training = policy_mode == "train"`). |
| No parameter updates | `agent.update(...)` and trajectory completion run only when `rl_training` is true (`OrchestrationLoop` in `orchestrator/loop.py`). |
| Same parameters before and after | `policy_fingerprint()` hashes the network or table parameters, excluding optimizer state (`orchestrator/rl/torch_common.py`, `torch_state_fingerprint`). Each evaluation records the fingerprint before and after; `assert_rl_policy_unchanged` fails the run on any change. |
| Correct artifact | The campaign index stores the SHA-256 of each trained artifact and its final fingerprint; the analyzers reject an evaluation whose loaded artifact hash or initial fingerprint differs (`src/argos/analysis/controlled.py`). The live runner re-hashes each artifact before a trial (`scripts/run_live_campaign.py`, `_policy_map`). |
| Correct controller, profile, and seed | Artifacts are resolved from the campaign index by profile, controller, and training seed; evaluation records carry `controller`, `profile`, `training_seed`, and `source_policy_fingerprint`. |
| No silent fallback | A learned request that ends without a loaded, unchanged policy is a hard failure of the trial (`src/argos/live_trial.py`), except requests admitted within the final polling window, which are reported as tail-censored. |

## Persisted records

Every run writes JSONL streams to its persistence directory
(`data/sessions/<session>/persistence/`):

| File | Content |
|---|---|
| `requests.jsonl` | submitted contracts |
| `request_lifecycle.jsonl` | admission, queueing, rejection, completion |
| `rl_decisions.jsonl` | state, action, reward and its components, exploration flag, algorithm |
| `config_changes.jsonl` | target configurations applied |
| `node_assignments.jsonl`, `node_plan_applied.jsonl` | placement |
| `request_metrics.jsonl` | target and realized quality, spatial fidelity, hotspot recall, service duty, contract bounds |
| `slo_violations.jsonl` | violated metric, measured value, limit, bound direction |
| `slo_decision_epochs.jsonl` | per-epoch violation flags |
| `tenant_fairness.jsonl`, `leader_election_events.jsonl`, `errors.jsonl` | fairness debt, control plane, errors |

Controlled runs also write `data/runs/<session>/` (CSV summaries,
`node_metrics.csv`, trained artifacts under `rl_models/`) and a provenance
manifest with the Git commit, clean-tree flag, code and dataset fingerprints,
package versions, and the exact command (`src/argos/provenance.py`).
