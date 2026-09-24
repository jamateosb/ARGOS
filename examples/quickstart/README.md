# Quickstart

A one-minute, single-machine run of ARGOS: three worker nodes, one analytics
request, a DQN controller trained for 24 decisions and then evaluated frozen for
8 decisions.

```bash
python examples/quickstart/run.py
```

Options: `--profile <name>` picks another workload profile from `config.yaml`
(default `lax-background`); `--base-port <port>` moves the three nodes off ports
8010 to 8012.

## What it does

1. Starts three worker nodes (`src/argos/node.py`) on localhost. Each node loads
   its share of the 60 Seville trajectories in `data/seville_bus/`.
2. Runs `src/argos/experiment.py` in training mode with the same arguments the
   controlled campaigns use, but 24 decisions instead of 2048. The orchestrator
   admits one request with the profile's accepted ranges, places it on the
   nodes, and lets DQN adjust coverage, sample, and freshness once per second.
3. Reloads the saved policy and runs 8 frozen decisions: no exploration, no
   parameter updates.
4. Checks that the evaluated policy has the fingerprint the training run
   produced and that it did not change during evaluation, then stops the nodes.

Only the number of decisions is reduced; the controller, environment, reward,
and orchestration code are the same as in the full evaluation. With 24
training decisions the policy has barely started learning, so its rewards say
nothing about the reference results. The run demonstrates the mechanics.

## Expected output

Exact numbers vary from run to run: rewards depend on the measured CPU and
memory of your machine. The structure and the two final checks do not.

```
[1/4] Starting three worker nodes on ports 8010-8012
[2/4] Training DQN for 24 decisions on one lax-background request (1 s per decision)
  target configuration at each decision, the action chosen, and its reward:
  step  coverage  sample  freshness  action                reward  explored
     1     0.350   0.325     135.0s  increase_freshness     0.278  yes
     2     0.350   0.325     125.0s  decrease_sample        0.254  no
     3     0.350   0.275     125.0s  increase_sample        0.303  yes
   ...
    24     0.250   0.200     180.0s  hold                  -0.163  no
  accepted ranges: coverage [0.25, 0.45], sample [0.2, 0.45], freshness [90, 180] s
[3/4] Frozen evaluation of the trained policy for 8 decisions (no exploration, no updates)
  target configuration at each decision, the action chosen, and its reward:
  step  coverage  sample  freshness  action                reward  explored
     1     0.350   0.325     135.0s  decrease_sample        0.202  no
   ...
     8     0.350   0.200     175.0s  decrease_freshness    -0.041  no
  accepted ranges: coverage [0.25, 0.45], sample [0.2, 0.45], freshness [90, 180] s
[4/4] Summary
  training reward per decision:          0.048
  frozen evaluation reward per decision: 0.087
  evaluated policy is the trained policy: yes
  policy unchanged during evaluation:     yes
  trained policy artifact: data/runs/quickstart_<timestamp>_train_dqn_seed11_<timestamp>/rl_models/agent_<id>_dqn.pt
  decision logs:           data/sessions/quickstart_<timestamp>_evaluate_dqn_s11/persistence/rl_decisions.jsonl
ARGOS quickstart completed successfully.
```

Reading the table: every value stays inside the accepted ranges; `explored`
is `yes` only during training (DQN's epsilon-greedy exploration); in the frozen
phase every action is the policy's greedy choice. `increase_freshness`
shortens the update interval (fresher results) and `decrease_freshness`
lengthens it.

## Where the output goes

| Path | Content |
|---|---|
| `data/sessions/quickstart_<timestamp>_<phase>_dqn_s11/persistence/` | JSONL audit streams: decisions with reward components, requests, lifecycle, configurations, placement, metrics, violations |
| `data/runs/quickstart_<timestamp>_<phase>_dqn_seed11_<timestamp>/` | CSV summaries, node metrics, provenance manifest, trained policy (`rl_models/`) |
| `data/sessions/quickstart_<timestamp>_<phase>/benchmark/benchmark_summary.json` | run summary, including policy fingerprints |
| `data/quickstart/<timestamp>/` | node logs |

`data/` is ignored by Git, so the quickstart leaves the checkout clean.

## Troubleshooting

- *must run from a Git checkout*: ARGOS records the commit of every run. In an
  extracted archive run `git init && git add -A && git commit -m "ARGOS"`.
- *Worker node did not become healthy*: another process uses ports 8010 to
  8012; pass `--base-port 8110`.
