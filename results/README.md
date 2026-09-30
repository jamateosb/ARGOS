# Results

This directory contains the result tables of the reference evaluation of ARGOS
(four controlled campaigns and one live campaign) and the figures generated
from them. Only frozen evaluation runs and accepted live trials are included;
training runs and Best-fixed tuning runs are inputs to the evaluation, not
results.

```
results/
├── controlled/tables/     controlled-evaluation tables
├── live/tables/           live-evaluation tables
└── figures/{pdf,png}/     figures (regenerate with scripts/generate_figures.py)
```

## Definitions

| Quantity | Definition |
|---|---|
| Reward per decision | Mean of the per-decision reward (clipped to [-1, 1]; the clip never binds in the reference evaluation, see the reward-clipping tables) over the decisions of a run. In the controlled evaluation the experimental unit is the evaluation seed: rewards are averaged over the five profiles within each seed. |
| Paired delta | Controller minus comparator, computed per evaluation seed (controlled) or per live seed (live), on identical seeds and traces. |
| 95 % interval | Two-sided Student-t interval over the five paired seed values (4 degrees of freedom). No confirmatory p-values are reported: with five pairs the exact Wilcoxon signed-rank test cannot give p < 0.0625. |
| Spatial fidelity | `1 - 0.5 * sum_i |p_i - q_i|` between the normalized cell frequencies of the sampled heatmap (p) and of the full-sample reference heatmap (q) of the same partitions. It reflects sampling only, not coverage or freshness. A value of 0.94 means the sampled spatial distribution is 94 % similar to the full-sample one in total-variation terms; it does not mean that 94 % of the city is covered. |
| Service duty | Processing time as a percentage of the requested update interval. |
| CPU / memory p95 | 95th percentile of the cluster utilization samples recorded during a run. |
| Violation event | One recorded breach of a contract or resource bound in one decision epoch. `bound` = `maximum` means the realized value was above the upper limit. |
| Violated-epoch ratio | Fraction of a trial's decision epochs with at least one violation; reported as the mean of the five per-trial ratios. |
| Evaluated request (live) | An admitted request that completed at least one frozen decision epoch. Requests admitted too late to do so are tail-censored. |

## Tables

### Controlled (`controlled/tables/`)

| File | One row per | Content |
|---|---|---|
| `controlled_runs.csv` | frozen evaluation run (300) | reward, spatial fidelity, hotspot recall, service duty, violation counts by metric, CPU and memory p95 and maximum, bandwidth p95 |
| `controlled_decisions.csv` | decision (19,200) | action, reward, cumulative reward, position of coverage, sample, and freshness inside their range, CPU, memory, input multiplier |
| `controlled_action_counts.csv` | run × action | count and share of each action |
| `controlled_reward_summary.csv` | runtime × controller | mean reward per decision and 95 % interval over the five seed means |
| `controlled_profile_reward_summary.csv` | runtime × profile × controller | mean reward and standard deviation over five seeds |
| `controlled_paired_reward_deltas.csv` | runtime × controller × comparator × seed | paired seed-level deltas against Static, Threshold, and Best-fixed |
| `controlled_paired_reward_summary.csv` | runtime × controller × comparator | mean paired delta, 95 % interval, wins, ties, losses |
| `controlled_violation_bounds.csv` | run × metric × bound | violation events by bound direction; `bound_source` is `recorded` when the event stores its bound and `derived` (measured value against limit) for CPU and memory caps |
| `controlled_reward_clipping.csv` | frozen evaluation run (300) | decisions, decisions whose unclipped reward (rebuilt from the persisted reward components) falls below −1 or above 1, unclipped minimum and maximum, clipped and unclipped reward per decision |
| `controlled_reward_clipping_sensitivity.csv` | runtime × controller × comparator | paired delta, 95 % interval, and wins computed with the clipped and with the unclipped reward; `conclusion_changed` flags a change of sign, of interval significance, or of wins |

### Live (`live/tables/`)

| File | One row per | Content |
|---|---|---|
| `live_runs.csv` | trial (30) | submitted, admitted on arrival, queued on arrival, evaluated and tail-censored requests, decision epochs, reward, violated epochs, violation counts by metric, spatial fidelity, hotspot recall |
| `live_summary.csv` | regime × variant | mean and standard deviation over five trials |
| `live_paired_deltas.csv` | regime × learned variant × seed | learned, static, and delta values of reward, evaluated requests, violation ratios, fidelity, recall |
| `live_load_timeline.csv` | trial × half-minute | active and queued requests |
| `live_reward_timeline.csv` | trial × half-minute | cumulative decision reward |
| `live_action_counts.csv` | trial × action | count and share of each action |
| `live_violation_bounds.csv` | trial × profile × metric × bound | violation events by workload profile and bound direction |
| `live_reward_clipping.csv` | trial (30) | as `controlled_reward_clipping.csv` |
| `live_reward_clipping_sensitivity.csv` | regime × learned variant | as `controlled_reward_clipping_sensitivity.csv`, against Static |

## Figures

All figures are generated by `scripts/generate_figures.py` from the tables
above; the PDF (vector) and PNG (300 dpi) versions share the same name. Colors,
markers, and hatching identify each controller consistently across figures;
Best-fixed is hatched to mark it as an offline-selected reference.

### Main figures

| Figure | Description | Source table |
|---|---|---|
| `controlled_mean_reward_thread`, `controlled_mean_reward_process` | Mean frozen reward per decision by controller; error bars are Student-t 95 % intervals over the five evaluation-seed means | `controlled_reward_summary.csv` |
| `controlled_reward_by_profile_thread`, `controlled_reward_by_profile_process` | Mean frozen reward per decision by workload profile and controller, averaged over five seeds | `controlled_profile_reward_summary.csv` |
| `live_request_load_realistic`, `live_request_load_concurrency` | Active (solid) and queued (dashed) requests over trial time, mean of five paired seeds; the three variants share the same arrival trace, so their curves largely coincide (line width decreases from Static to PPO to keep all three visible) | `live_load_timeline.csv` |
| `live_reward_by_seed_realistic`, `live_reward_by_seed_concurrency` | Mean decision reward of each trial by live seed; both panels share one y axis | `live_runs.csv` |

### Additional figures

| Figure | Description | Source table |
|---|---|---|
| `controlled_paired_reward_deltas` | Paired reward delta of Q-learning, DQN, and PPO against Static, Threshold, and Best-fixed; bars are means, error bars Student-t 95 % intervals, dots the five seed pairs | `controlled_paired_reward_deltas.csv` |
| `controlled_fidelity_vs_duty` | Mean spatial fidelity against mean service duty per controller (25 runs each); the y axis starts at 0.90 to separate the controllers | `controlled_runs.csv` |
| `controlled_resource_pressure` | Mean over 25 runs of the per-run cluster CPU and memory p95, thread (plain) and process (dotted) | `controlled_runs.csv` |
| `controlled_action_frequency` | Share of frozen decisions per action, pooled over profiles and seeds | `controlled_action_counts.csv` |
| `coverage_violations_by_profile` | Coverage violation events above the contract maximum by profile, controlled (both runtimes) and live (both regimes); the titles report the number of events below a contract minimum. With contract-feasible placement the reference evaluation records no coverage event in either direction | `controlled_violation_bounds.csv`, `live_violation_bounds.csv` |
| `live_paired_fidelity_delta` | Spatial-fidelity delta of DQN and PPO against Static per live seed, with the mean and its Student-t 95 % interval | `live_paired_deltas.csv` |
| `live_admission_outcomes` | Requests submitted, admitted on arrival, queued on arrival, and evaluated, summed over the five trials of each variant | `live_runs.csv` |
| `live_cumulative_reward` | Cumulative decision reward over trial time, mean of five paired seeds | `live_reward_timeline.csv` |

## Key numbers

All values can be recomputed from the tables above.

| Controlled | Thread | Process |
|---|---:|---:|
| Mean reward per decision: Static / Threshold / Q-learning | 0.229 / 0.456 / 0.205 | 0.227 / 0.461 / 0.189 |
| Mean reward per decision: DQN / PPO / Best-fixed | 0.376 / 0.390 / 0.544 | 0.368 / 0.398 / 0.549 |
| DQN − Static (wins of 5) | +0.148 (5) | +0.141 (5) |
| PPO − Static (wins of 5) | +0.161 (5) | +0.170 (5) |
| DQN − Threshold (wins of 5) | −0.079 (0) | −0.093 (1) |
| PPO − Threshold (wins of 5) | −0.066 (1) | −0.063 (1) |
| DQN − Best-fixed (wins of 5) | −0.168 (0) | −0.181 (0) |
| PPO − Best-fixed (wins of 5) | −0.154 (0) | −0.152 (0) |
| Aggressive-incident only: Threshold / DQN / PPO | 0.006 / 0.320 / 0.255 | 0.035 / 0.330 / 0.268 |
| Decisions with a clipped reward | 0 of 9,600 | 0 of 9,600 |

| Live | Realistic | Concurrency |
|---|---:|---:|
| Mean decision reward: Static / DQN / PPO | 0.237 / 0.458 / 0.400 | −0.016 / 0.036 / 0.094 |
| DQN − Static, 95 % interval (wins) | +0.221 [0.112, 0.330] (5) | +0.052 [−0.067, 0.172] (3) |
| PPO − Static, 95 % interval (wins) | +0.163 [0.110, 0.216] (5) | +0.110 [0.039, 0.180] (5) |
| Coverage violations: Static / DQN / PPO | 0 / 0 / 0 | 0 / 0 / 0 |
| Freshness violations: Static / DQN / PPO | 0 / 0 / 0 | 0 / 3 / 18 |
| Decisions with a clipped reward | 0 of 18,011 | 0 of 44,965 |
