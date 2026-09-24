# ARGOS: Reinforcement Learning-Driven Multidimensional Elasticity for Service Orchestration in the Computing Continuum

ARGOS (Adaptive Reinforcement Learning-Driven Governance for Orchestrated
Services) is an orchestrator for analytics services that run on heterogeneous
nodes of the computing continuum. This repository contains its source code,
the configuration and scripts of its evaluation, the result tables, and the
code that generates every figure.

## Overview

Clients submit analytics requests whose quality is not a single value but an
accepted range on three dimensions: coverage, sample, and freshness. ARGOS
validates each request against the capacity of the cluster, admits it, queues
it while capacity is exhausted, or rejects it when its minimum requirements can
never be placed. Every admitted request then receives its own controller, which
repeatedly observes resource pressure and delivered quality and moves the
request's operating point inside its accepted ranges: lower quality when the
cluster is under pressure, higher quality when capacity is available. The
controller is pluggable. The same state, actions, action masking, and
configuration path serve three learned controllers (Q-learning, DQN, PPO) and
three non-learning comparators (static midpoint, threshold, best-fixed).

ARGOS has been evaluated on a heterogeneous three-node cluster, in a controlled
stage (frozen policies, five workload profiles, two runtime modes) and a live
stage (30-minute multi-tenant traffic trials). The results describe behavior
at that scale; see [Scope and limitations](#scope-and-limitations).

## Core concepts

| Term | Meaning |
|---|---|
| **Coverage** | Fraction of the spatial or user partitions of the dataset included in the analysis. Each assigned node processes one partition, so realized coverage is the assigned-node count divided by the number of active nodes. |
| **Sample** | Fraction of the available observations processed within the selected partitions. |
| **Freshness** | Interval between result updates, in seconds. A **lower** interval means a **fresher** result and more work. |
| **Analytics contract** | The accepted range for each of the three dimensions, plus tenant identity, priority (critical, standard, best-effort), per-request CPU and memory caps, placement limits, and an optional cost budget. |
| **Spatial fidelity** | Similarity between the heatmap computed from the sampled data and the full-sample reference heatmap of the same partitions: `F = 1 - 0.5 * sum_i |p_i - q_i|` over normalized cell frequencies (one minus the total variation distance). It measures the effect of sampling only; it does not include coverage or freshness. |
| **Service duty** | Processing time of the analytics as a percentage of the requested update interval. |

Action names refer to the quality of a dimension: `increase_freshness` makes
results fresher by **shortening** the update interval by 10 s, and
`decrease_freshness` lengthens it.

## Architecture

```
client ──► Request interface (orchestrator API)
             │ contract validation
             ▼
           Admission control ──► reject (infeasible) │ queue (no capacity yet) │ admit
             │ admitted
             ▼
           Per-request controller  (one per request)
             ├─ Environment: state encoding (13 dims), action masking, range clamp, reward
             └─ Policy: Static │ Threshold │ Best-fixed │ Q-learning │ DQN │ PPO
             │ target coverage / sample / freshness
             ▼
           Deployment manager ──► node plans ──► Node adapters (argos.node) ──► heatmap runtime
             ▲                                        │ (thread or process execution)
           Runtime monitor ◄── CPU, memory, bandwidth, latency, realized quality
             │
           Runtime audit log: requests, lifecycle, decisions, configurations, telemetry, violations
```

The decision loop runs at a fixed cadence (1 s in the evaluation). On each
iteration ARGOS polls node telemetry, encodes the state of each admitted
request, masks actions that would have no effect or would add load under
pressure, lets the request's policy choose among the remaining seven actions
(`hold` and one increase and one decrease per dimension), applies the change
within the contract bounds, pushes new node plans when the placement changes,
and records every decision with its reward components. Target coverage maps to
a node count of ⌈coverage × N⌉, so on a three-node cluster realized coverage
can only be 1/3, 2/3, or 1. [docs/architecture.md](docs/architecture.md)
describes the components, the state, the reward, and where each lives in the
code.

## Controllers

| Controller | Kind | Behavior |
|---|---|---|
| Q-learning | learned | Tabular values over the encoded state |
| DQN | learned | Neural value function with replay buffer and target network |
| PPO | learned | Actor-critic with clipped policy updates |
| Static | comparator | Holds the midpoint of every accepted range |
| Threshold | comparator | Reduces quality when CPU, memory, duty, queueing, or recent violations reach warning levels; restores it when pressure subsides |
| Best-fixed | comparator | A static configuration selected offline for each workload profile from 9 predeclared candidates (the 8 vertices of the contract box and its joint midpoint) on separate tuning seeds, then held fixed during evaluation |

Best-fixed is a tuned static reference, not an oracle: it is selected with
knowledge of the profile but cannot adapt within a run.

## Repository structure

| Path | Contents |
|---|---|
| `src/argos/orchestrator/` | Orchestrator API, control loop, admission, placement, persistence, decision environment (MDP), and controllers (`rl/`) |
| `src/argos/node.py` | Worker-node API and analytics runtimes (thread and process execution) |
| `src/argos/services/` | Geographic heatmap service and spatial fidelity |
| `src/argos/domain/` | Requests, contracts, cost model, resource monitoring |
| `src/argos/experiment.py` | Runs one controller on one request against a set of nodes (training or frozen evaluation) |
| `src/argos/live_trial.py` | Live traffic client used by the live campaign |
| `src/argos/analysis/` | Campaign validation, statistics, and result tables |
| `scripts/` | The five commands of the workflow ([Workflow](#workflow)) |
| `config.yaml` | Configuration: RL parameters, workload profiles, orchestration, cost model, cluster |
| `data/seville_bus/` | Input dataset: 60 simulated per-user movement trajectories over Seville |
| `examples/quickstart/` | A one-minute local run |
| `tests/` | Test suite |
| `results/` | Result tables and figures ([results/README.md](results/README.md)) |
| `docs/` | Architecture, evaluation protocols, reproducibility guide |
| `deploy/aws/` | Deployment guide and service units for a three-node cluster |

## Requirements

- Linux (tested on x86_64).
- Python 3.9 to 3.12 (the test suite is run on 3.9 and 3.12).
- Git: every run records the commit of the code it ran, so ARGOS runs from a
  Git checkout.
- The Python packages in [requirements.txt](requirements.txt): FastAPI,
  Uvicorn, httpx, Pydantic, psutil, PyYAML, nats-py, PyTorch (the CPU build is
  enough), NumPy, pandas, SciPy, Matplotlib.
- NATS is optional. It carries control-plane events only when
  `orchestration.control_plane.nats_enabled` is set for a multi-orchestrator
  deployment. The evaluation campaigns, the quickstart, the tests, and the
  figures do not use it.

## Installation

```bash
git clone https://github.com/jamateosb/ARGOS.git
cd ARGOS
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt -r requirements-dev.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu
```

The extra index installs the CPU-only PyTorch build (about 200 MB instead of
several GB of CUDA libraries). Omit it to install the default PyTorch wheel.

If you work from a downloaded archive instead of a clone, turn the extracted
folder into a Git repository once: `git init && git add -A && git commit -m "ARGOS"`.

## Quick start

```bash
python examples/quickstart/run.py
```

In about one minute, and without any cloud resources, this starts three worker
nodes on localhost ports 8010 to 8012, trains a DQN controller for 24 decisions
on one request, reloads the trained policy for 8 frozen decisions, prints every
decision (coverage, sample, freshness, action, reward), and checks that the
evaluated policy is the trained one and did not change. It uses the same entry
point and arguments as the controlled evaluation with far fewer decisions, so
its numbers are a functional check only. See
[examples/quickstart/README.md](examples/quickstart/README.md) for the expected
output.

## Running ARGOS

A deployment has one orchestrator and one or more worker nodes. Commands run
from the repository root.

```bash
# worker node (one per machine; NODE_ID selects the dataset partition)
cd src && NODE_ID=1 uvicorn argos.node:app --host 0.0.0.0 --port 8000

# orchestrator API (reads config.yaml; accepts requests at POST /submit-job)
cd src && uvicorn argos.orchestrator.api:app --host 0.0.0.0 --port 8001

# one controller on one request, as in the controlled evaluation
PYTHONPATH=src python -m argos.experiment --mode benchmark \
    --nodes http://<node-1>:8000,http://<node-2>:8000,http://<node-3>:8000 \
    --algorithm dqn --seeds 11 --iterations 64 --poll-interval 1 \
    --coverage-range 0.25,0.45 --sample-range 0.20,0.45 --freshness-range 90,180 --policy-mode train
```

Each run writes its decision log, request metrics, violations, lifecycle
events, and a run manifest under `data/sessions/<session>/` and
`data/runs/<session>/`, and saves trained policies under
`data/runs/<session>/rl_models/`. [docs/architecture.md](docs/architecture.md#persisted-records)
lists the files, and [deploy/aws/README.md](deploy/aws/README.md) describes a
three-node deployment.

## Workflow

The full evaluation uses five commands, in this order:

| Step | Command | What it does |
|---|---|---|
| 1 | `python scripts/set_runtime_mode.py thread` | Sets every worker node to the thread (or `process`) runtime |
| 2 | `python scripts/run_controlled_campaign.py ...` | Trains the learned controllers, selects Best-fixed, and evaluates every controller with frozen policies |
| 3 | `python scripts/run_live_campaign.py ...` | Runs the paired live trials with the frozen DQN and PPO policies |
| 4 | `python scripts/analyze_results.py ...` | Validates the campaigns, computes the statistics, and writes the result tables |
| 5 | `python scripts/generate_figures.py` | Generates every figure from the result tables |

Steps 1 to 4 need a three-machine cluster and several days; the exact commands
are in [docs/reproducibility.md](docs/reproducibility.md#3-new-campaigns). Step 5
works directly on the tables included in this repository.

## Controlled evaluation

Each learned controller is trained for **2048** decisions at a 1 s cadence
under the input-load schedule `2,16,64,4` (16 decisions per phase, 32 complete
cycles), with training seeds 11, 22, 33, 44, 55. The saved policy is then
evaluated **frozen** for 64 decisions under the different schedule `3,12,48,6`
(16 decisions per phase) on held-out evaluation seeds 111, 222, 333, 444, 555.
A frozen policy still chooses actions from the observed state, but without
exploration and without parameter updates, and its parameter fingerprint is
checked before and after. The six controllers are compared on five workload
profiles and on both runtime modes (thread and process). Best-fixed is chosen
on tuning seeds 101, 202, 303. The design yields 300 frozen evaluations and
19,200 frozen decisions. Details: [docs/controlled_evaluation.md](docs/controlled_evaluation.md).

## Live evaluation

The live stage runs the static midpoint and the frozen DQN and PPO policies
under realistic arrivals (below saturation) and concurrency arrivals
(overload), on the thread runtime, with five live seeds (1001 to 1005). Each
seed produces one 30-minute trial per variant with the same arrival trace, for
30 trials in total. Admission, queueing, and placement stay active in every
variant. The trials cover 58,285 decision epochs and 369 evaluated requests.
Details: [docs/live_evaluation.md](docs/live_evaluation.md).

## Results

[results/](results/README.md) contains the tables behind every reported number
and figure, and the figures themselves.

## Figures

```bash
python scripts/generate_figures.py          # all 16 figures
python scripts/generate_figures.py --main   # the 8 main panels only
```

Figures are written to `results/figures/pdf/` and `results/figures/png/` from
the result tables alone. [results/README.md](results/README.md) maps each
figure to its source table.

## Tests

```bash
python -m pytest
```

The suite (425 tests, 2 of which are skipped when PyTorch is installed) runs in
about 10 seconds and needs no network or running nodes.

## Reproducibility levels

| Level | What it reproduces | Needs | Time |
|---|---|---|---|
| Quick check | ARGOS runs end to end: nodes, admission, controller decisions, frozen evaluation, persistence | this repository | about 1 minute |
| Figures and numbers | every figure and reported number from the result tables | this repository | about 1 minute |
| New campaigns | a full controlled and live evaluation on your own cluster | three machines, several days of runtime | see guide |

[docs/reproducibility.md](docs/reproducibility.md) gives the commands for each level.

## Scope and limitations

- The evaluation uses a cluster of three heterogeneous nodes, and the results
  describe ARGOS at that scale. The architecture is designed for larger
  continuum deployments, but larger topologies have not been evaluated yet.
- With three nodes, realized coverage can only be 1/3, 2/3, or 1. Every
  coverage violation recorded in the evaluation is an **upper-bound**
  violation: the node count ⌈coverage × N⌉ rounds the target up, so realized
  coverage exceeds the contract maximum (for example 2/3 against a maximum of
  0.45). No coverage event fell below a contract minimum.
- The workload is one urban-mobility heatmap service over simulated
  trajectories, with five workload profiles.
- New campaigns depend on the machines and their real telemetry, so their
  numbers will differ in detail from the reference results. The figures and
  numbers in this repository are reproduced exactly from the result tables.

## Citation

If you use ARGOS, please cite it using the metadata in
[CITATION.cff](CITATION.cff) (GitHub shows it under "Cite this repository").

## License

ARGOS is released under the [MIT License](LICENSE). Copyright (c) 2026 Javier
Mateos-Bravo.
