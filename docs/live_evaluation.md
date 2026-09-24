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
| Decision epochs | 58,285 |
| Requests submitted | 693 |
| Evaluated requests (at least one frozen decision) | 369 |
| Tail-censored requests | 2 |
| Violation events | 27,145 |
| of which coverage (all above the contract maximum) | 27,130 |
| of which freshness | 15 |
| of which sample, CPU, memory | 0 |

**Coverage violations.** All 27,130 coverage events are above the contract
maximum; none is below the contract minimum. With three nodes, realized
coverage is 1/3, 2/3, or 1, and the node count ⌈coverage × 3⌉ rounds the
controller's target up: a lax-background request with a target inside
[0.25, 0.45] that receives two nodes realizes 2/3. In every event the assigned
node count equals the desired node count, so the events come from placement
granularity, not from a shortage of nodes. Controllers that hold higher
coverage targets cross into the next node count more often and therefore
record more events.

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
