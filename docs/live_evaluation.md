# Live evaluation

The live stage tests whether the frozen deep controllers remain deployable when
requests arrive and depart over time and compete for the cluster. Admission,
queueing, and placement stay active in every variant; only the per-request
controller differs.

## Protocol

| Item | Value |
|---|---|
| Variants | Static midpoint, frozen DQN, frozen PPO |
| Runtime | thread |
| Arrival regimes | realistic (below saturation), concurrency (overload) |
| Live seeds | 1001, 1002, 1003, 1004, 1005 |
| Trials | 2 regimes × 5 seeds × 3 variants = 30 |
| Trial length | 30 minutes |
| Polling interval of the traffic client | 10 s |
| Policy source | policies trained in the two controlled thread campaigns; each live seed is paired with one training seed (1001 ↔ 11, …, 1005 ↔ 55) |

Q-learning is not included in the live stage because it does not improve on
the static midpoint in the controlled evaluation.

**Arrival regimes** (`SCENARIO_PARAMETERS` in `scripts/run_live_campaign.py`):

| Regime | Inter-arrival time | Request duration |
|---|---|---|
| realistic | uniform 2 to 4 min | uniform 10 to 15 min |
| concurrency | uniform 0.5 to 1 min | until the end of the trial |

Each submitted request takes the next workload profile in a fixed rotation over
the five profiles of `config.yaml`.

**Pairing.** The traffic generator is seeded with the live seed, so the static,
DQN, and PPO trials of one seed receive the identical arrival trace. The order
in which the three variants run is rotated across seeds to spread any drift of
the shared infrastructure. Each trial starts its own orchestrator with a fresh
persistence directory.

**Frozen policies.** Before the campaign, the runner looks up, for every
algorithm, training seed, and profile, the trained policy recorded by the
controlled campaigns and checks that the file has not changed since training.
In a learned trial, every admitted request loads the policy of its profile
before its first decision, including requests that were queued first;
exploration and parameter updates are disabled, and the policy fingerprint must
be unchanged when the request ends. A request admitted within the final polling
window of a trial may end before completing one decision; such requests are
reported as tail-censored rather than as evaluated.

## Reference results

| Item | Value |
|---|---:|
| Trials | 30 |
| Decision epochs | 62,976 |
| Requests submitted | 692 |
| Evaluated requests (at least one frozen decision) | 393 |
| Tail-censored requests | 2 |
| Violation events | 21 |
| of which freshness | 21 |
| of which coverage, sample, CPU, memory | 0 |

| Regime | Static | DQN | PPO |
|---|---:|---:|---:|
| Realistic: mean decision reward | 0.237 | 0.458 | 0.400 |
| Realistic: delta vs Static, 95 % interval (wins of 5) | | +0.221 [0.112, 0.330] (5) | +0.163 [0.110, 0.216] (5) |
| Concurrency: mean decision reward | −0.016 | 0.036 | 0.094 |
| Concurrency: delta vs Static, 95 % interval (wins of 5) | | +0.052 [−0.067, 0.172] (3) | +0.110 [0.039, 0.180] (5) |

**Coverage.** With three nodes, realized coverage is 1/3, 2/3, or 1. The
node count is ⌈coverage × 3⌉ restricted to the counts whose realized coverage
lies inside the contract range, so no coverage violation is recorded. The 21
freshness events occur in the concurrency regime (3 under DQN, 18 under PPO).

**Admission.** In the realistic regime every request is admitted on arrival
and at most six are active. In the concurrency regime about 46 of the 180
requests per variant are admitted on arrival, the rest wait in the queue, and
the active set saturates at nine to ten requests. The last scheduled arrival of
live seed 1004 (realistic) falls about 4 s before the end of the trial, so it is
submitted in the DQN and PPO trials (and tail-censored) but not in the static
trial; the evaluated requests are the same in all three.

## Running a campaign

```bash
python scripts/run_live_campaign.py --campaign-id live_thread \
    --campaign-index data/campaigns/thread_primary/campaign_index.json \
    --campaign-index data/campaigns/thread_generalization/campaign_index.json \
    --runtime thread
```

The runner needs the three worker nodes of `config.yaml` running in thread mode
and a Git tree without uncommitted changes. The defaults match the protocol
above (variants static, DQN, and PPO; live seeds 1001 to 1005; training seeds
11 to 55; both regimes; 30-minute trials; 10 s client polling). The 30 trials
take at least 15 hours. `--resume` continues an interrupted campaign.

## Analysis

The analyzer checks that every trial record is intact, that the paired design
is complete, and that the frozen-policy checks passed. It writes run-level
metrics, per-seed paired deltas (learned minus static), and a summary with
Student-t 95 % intervals over the five paired seeds. The commands are in
[reproducibility.md](reproducibility.md#analysis-tables-and-figures).
