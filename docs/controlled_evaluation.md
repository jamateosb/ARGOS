# Controlled evaluation

The controlled stage compares six controllers on one analytics request at a
time, under a fixed input-load schedule, on the three-node cluster. It separates
training from frozen evaluation so that every reported number comes from a
policy that no longer learns.

## Protocol

| Item | Value |
|---|---|
| Decision cadence | 1 s |
| Training | 2048 decisions per learned controller, input-load schedule `2,16,64,4`, 16 decisions per phase, 32 complete cycles |
| Best-fixed tuning | 64 decisions per candidate, 9 candidates, tuning seeds 101, 202, 303 |
| Frozen evaluation | 64 decisions, input-load schedule `3,12,48,6`, 16 decisions per phase, no exploration, no parameter updates |
| Training seeds | 11, 22, 33, 44, 55 |
| Evaluation seeds | 111, 222, 333, 444, 555 |
| Profiles | lax-background, short-burst, aggressive-incident, standard-operations, cost-sensitive |
| Runtimes | thread, process |
| Learned controllers | Q-learning, DQN, PPO |
| Comparators | Static midpoint, Threshold, Best-fixed |

The input-load schedule multiplies the trajectory volume processed in each
decision phase. Training and evaluation use different schedules and different
seeds, so evaluation never replays the training trace. The accepted ranges of
each profile are defined in `config.yaml` (`live_trial.profiles`).

**Training versus frozen evaluation.** During training the controller explores
and updates its parameters after every decision; at the end its policy is saved
together with a parameter fingerprint. In frozen evaluation the saved policy is
loaded before the first decision. It still selects actions from the observed
state, but greedily (no exploration), and its parameters are never updated; the
fingerprint is recorded before and after the run and must match the one saved
at the end of training. The mechanisms are listed in
[architecture.md](architecture.md#frozen-policy-guarantees).

**Best-fixed.** For each profile, nine static configurations are defined in
advance: the eight vertices of the contract box (each dimension at its minimum
or maximum) and the joint midpoint. Each is run for 64 decisions on the three
tuning seeds; the one with the highest mean tuning reward becomes the profile's
Best-fixed configuration and is then evaluated like any other controller. It is
a tuned static reference, not an oracle. In the reference evaluation the
selected candidate was the same for every profile and runtime: coverage and
sample at their maximum, update interval at its minimum.

**Pairing and statistics.** Controllers are paired on runtime, profile, and
evaluation seed. Reward per decision is averaged over the five profiles within
each evaluation seed, giving five paired seed-level values per controller and
runtime. The tables report the mean of the five paired deltas, a two-sided
Student-t 95 % interval over those five values, and win counts. With five pairs
the exact two-sided Wilcoxon signed-rank test cannot give p < 0.0625, so no
confirmatory p-values are reported.

## Design size

| Role | Per runtime | Formula |
|---|---:|---|
| Training | 75 | 5 profiles × 5 seeds × 3 learners |
| Best-fixed tuning | 135 | 5 profiles × 3 seeds × 9 candidates |
| Frozen evaluation | 150 | 5 profiles × 5 seeds × 6 controllers |
| Total | 360 | 720 over both runtimes |

The 300 frozen evaluations contain 19,200 decisions. All 150 learned-policy
evaluations kept the fingerprint of their trained policy.

## Running a campaign

The campaign runner trains, tunes, and evaluates in one pass, validates every
run as it completes, and records it in `data/campaigns/<campaign-id>/campaign_index.json`.
It refuses to start from a Git tree with uncommitted changes, because each run
records the commit it ran.

Each runtime is run as two campaigns: a primary campaign with the
aggressive-incident profile and a generalization campaign with the other four
profiles. The node endpoints are read from `config.yaml` by
`set_runtime_mode.py`; pass the same endpoints to the runner.

```bash
NODES=http://<node-large>:8000,http://<node-medium>:8000,http://<node-small>:8000

python scripts/set_runtime_mode.py thread
python scripts/run_controlled_campaign.py --campaign-id thread_primary \
    --nodes "$NODES" --runtime thread --profiles aggressive-incident
python scripts/run_controlled_campaign.py --campaign-id thread_generalization \
    --nodes "$NODES" --runtime thread \
    --profiles lax-background,standard-operations,cost-sensitive,short-burst

python scripts/set_runtime_mode.py process
python scripts/run_controlled_campaign.py --campaign-id process_primary \
    --nodes "$NODES" --runtime process --profiles aggressive-incident
python scripts/run_controlled_campaign.py --campaign-id process_generalization \
    --nodes "$NODES" --runtime process \
    --profiles lax-background,standard-operations,cost-sensitive,short-burst
```

The remaining parameters default to the protocol above: 2048 training
decisions, 64 tuning and evaluation decisions, training, evaluation, and tuning
seeds, 1 s cadence, schedules `2,16,64,4` and `3,12,48,6`, and 16 decisions per
phase. `--resume` continues an interrupted campaign; `--dry-run` prints the
commands without running them. One runtime takes about 49 hours (primary about
10 hours, generalization about 39 hours).

## Analysis

The analyzer reads only frozen evaluation runs. It rejects any run with
uncommitted code changes, a policy that does not match its training record, or
a decision log that does not reproduce the recorded reward, and writes per-run
metrics, paired deltas, and descriptive statistics. The commands are in
[reproducibility.md](reproducibility.md#analysis-tables-and-figures).
