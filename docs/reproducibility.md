# Reproducibility guide

ARGOS supports three levels of reproduction. The first two need only this
repository and a Linux machine; the third needs a three-machine cluster.

```
campaigns (scripts/run_controlled_campaign.py, scripts/run_live_campaign.py)
      │  campaign indexes and per-run logs under data/
      ▼
scripts/analyze_results.py
      │  validation and statistics (data/analysis/), result tables (results/*/tables/)
      ▼
scripts/generate_figures.py
      │
      ▼
results/figures/{pdf,png}/
```

All commands run from the repository root inside the environment created in
the [README](../README.md#installation).

## 1. Quick check

Checks that ARGOS runs end to end on one machine, in about one minute, without
cloud resources.

```bash
python examples/quickstart/run.py
python -m pytest
```

The quickstart starts three worker nodes on localhost, trains a DQN controller
for 24 decisions, evaluates the frozen policy for 8 decisions, and checks the
policy fingerprint. See [examples/quickstart/README.md](../examples/quickstart/README.md).

## 2. Figures and numbers

Regenerates every figure from the result tables included in the repository.

```bash
rm -rf results/figures
python scripts/generate_figures.py
```

With the package versions listed below, the regenerated PDF and PNG files are
identical to the ones in the repository. Every reported number can be
recomputed from the same tables; for example, the controlled mean reward per
controller and runtime:

```python
import pandas as pd
runs = pd.read_csv("results/controlled/tables/controlled_runs.csv")
seed_means = runs.groupby(["runtime", "controller", "seed"]).reward_per_step.mean()
print(seed_means.groupby(["runtime", "controller"]).mean().round(3))
```

[results/README.md](../results/README.md) describes every table.

## 3. New campaigns

Runs the full evaluation on your own cluster. The reference evaluation used
three worker nodes of different sizes:

| Node | Role | vCPUs | Memory |
|---|---|---:|---:|
| Large | orchestrator and worker | 16 | 32 GB |
| Medium | worker | 2 | 4 GB |
| Small | worker | 1 | 2 GB |

Steps:

1. Deploy the worker-node service on the three machines:
   [deploy/aws/README.md](../deploy/aws/README.md).
2. Fill in the `<...>` placeholders of `config.yaml` (node addresses, SSH user
   and key) and set the API tokens described in the deployment guide.
3. Commit any local change: the campaign runners refuse to start from a Git
   tree with uncommitted changes, because every run records its commit.
4. Controlled stage: [controlled_evaluation.md](controlled_evaluation.md#running-a-campaign)
   (about 49 hours per runtime).
5. Live stage: [live_evaluation.md](live_evaluation.md#running-a-campaign)
   (at least 15 hours).
6. Analysis, tables, and figures (below).

Results of new campaigns will differ in detail from the reference results:
decisions depend on the CPU, memory, and timing measured on the machines, so
trajectories diverge between runs even with identical seeds. The seeds fix the
random streams of the controllers and of the workload, not the physical
behavior of the machines.

### Analysis, tables, and figures

```bash
python scripts/analyze_results.py \
    --controlled data/campaigns/thread_primary/campaign_index.json \
    --controlled data/campaigns/thread_generalization/campaign_index.json \
    --controlled data/campaigns/process_primary/campaign_index.json \
    --controlled data/campaigns/process_generalization/campaign_index.json \
    --live data/live_campaigns/live_thread/campaign_index.json

python scripts/generate_figures.py
```

`analyze_results.py` validates every campaign, compares the learned
controllers against each comparator on each runtime, analyzes the live
campaign, writes the per-run analyses to `data/analysis/`, and overwrites the
result tables in `results/` with those of your campaigns. Campaign records
store absolute paths; if you move the `data/` directory, add
`--evidence-root <directory containing data/>`.

## Tested versions

The test suite, the quickstart, and the figure generation are tested on Python
3.9.16 and 3.12.14. The reference results were produced with:

| Package | Version |
|---|---|
| Python | 3.9.16 |
| torch | 2.8.0+cpu |
| numpy | 2.0.2 |
| pandas | 2.3.3 |
| scipy | 1.13.1 |
| matplotlib | 3.9.4 |
| fastapi | 0.128.8 |
| pydantic | 2.13.3 |
| uvicorn | 0.39.0 |
| httpx | 0.28.1 |
| PyYAML | 6.0.3 |
| psutil | 7.2.2 |

## Scripts

| Script | Purpose |
|---|---|
| `scripts/set_runtime_mode.py` | switch all worker nodes between thread and process mode over SSH, and verify |
| `scripts/run_controlled_campaign.py` | train, tune Best-fixed, and evaluate frozen policies for a set of profiles on one runtime |
| `scripts/run_live_campaign.py` | run the paired live trials with frozen policies |
| `scripts/analyze_results.py` | validate the campaigns, compute the statistics, and build the result tables |
| `scripts/generate_figures.py` | generate every figure from the result tables |
| `examples/quickstart/run.py` | one-minute local run |

The scripts are thin command-line entry points; the logic lives in the
`argos` package (`argos.experiment` runs one controller on one request,
`argos.live_trial` drives one live trial, and `argos.analysis` holds the
validation, statistics, and table code).
