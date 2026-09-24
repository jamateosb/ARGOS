#!/usr/bin/env python3
"""ARGOS quickstart: a complete train-then-freeze cycle on one machine.

The script starts three ARGOS worker nodes on localhost, then runs the same
experiment entry point as the controlled evaluation (argos.experiment, with
the arguments used by scripts/run_controlled_campaign.py) at a much smaller
scale:

1. training: a DQN controller adapts coverage, sample, and freshness for one
   analytics request of the lax-background profile over 24 one-second decisions;
2. frozen evaluation: the trained policy is reloaded and evaluated over 8
   decisions without exploration or parameter updates, and its fingerprint is
   checked before and after.

Nothing here changes the algorithms; only the number of decisions is reduced
(the full evaluation uses 2048 training and 64 evaluation decisions). The
results are therefore a functional check, not a reproduction of the reference
results.

    python examples/quickstart/run.py

Runtime is about one minute. Outputs are written under data/ in the repository.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config.yaml"


def _profile(name: str) -> dict:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    for profile in config["live_trial"]["profiles"]:
        if profile["name"] == name:
            return profile["payload"]
    raise SystemExit(f"Unknown profile {name!r}; see live_trial.profiles in config.yaml")


def _require_git_checkout() -> None:
    probe = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    if probe.returncode != 0:
        raise SystemExit(
            "ARGOS records the Git commit of every run and must run from a Git checkout.\n"
            "Clone the repository, or inside an extracted archive run:\n"
            "    git init && git add -A && git commit -m 'ARGOS release'"
        )


def _wait_healthy(url: str, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=1.0) as response:
                if response.status == 200:
                    return
        except OSError:
            time.sleep(0.3)
    raise SystemExit(f"Worker node at {url} did not become healthy")


def _start_nodes(base_port: int, log_dir: Path) -> tuple[list[subprocess.Popen], list[str]]:
    processes, urls = [], []
    for index in range(3):
        port = base_port + index
        env = {**os.environ, "NODE_ID": str(index + 1), "ARGOS_MACHINE_ID": f"quickstart-node-{index + 1}"}
        log = (log_dir / f"node{index + 1}.log").open("w", encoding="utf-8")
        processes.append(
            subprocess.Popen(
                [sys.executable, "-m", "uvicorn", "argos.node:app", "--host", "127.0.0.1", "--port", str(port)],
                cwd=ROOT / "src",
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        )
        urls.append(f"http://127.0.0.1:{port}")
    for url in urls:
        _wait_healthy(url)
    return processes, urls


def _run(session_id: str, nodes: list[str], payload: dict, profile: str, *, mode: str, iterations: int,
         schedule: str, segment: int, exploration_steps: int, policy_path: str = "") -> dict:
    limits = payload.get("resource_limits") or {}
    command = [
        sys.executable, "-m", "argos.experiment",
        "--config", str(CONFIG),
        "--mode", "benchmark",
        "--nodes", ",".join(nodes),
        "--session-id", session_id,
        "--experiment-name", session_id,
        "--iterations", str(iterations),
        "--poll-interval", "1.0",
        "--algorithm", "dqn",
        "--seeds", "11",
        "--coverage-range", f"{payload['coverage_min']},{payload['coverage_max']}",
        "--sample-range", f"{payload['sample_min']},{payload['sample_max']}",
        "--freshness-range", f"{payload['freshness_min']},{payload['freshness_max']}",
        "--input-multiplier", "4",
        "--input-schedule", schedule,
        "--schedule-segment-iterations", str(segment),
        "--initial-quantiles", "0.5,0.5,0.5",
        "--profile-name", profile,
        "--tenant-id", f"quickstart-{profile}",
        "--priority", str(payload.get("priority", "standard")),
        "--cpu-max", str(limits.get("cpu_max_percent", 80.0)),
        "--memory-max", str(limits.get("memory_max_percent", 85.0)),
        "--rl-params-source", "default",
        "--exploration-decay-steps", str(exploration_steps),
        "--policy-mode", mode,
    ]
    if policy_path:
        command += ["--policy-path", policy_path]
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    subprocess.run(command, cwd=ROOT, env=env, check=True, stdout=subprocess.DEVNULL)
    summary_path = ROOT / "data" / "sessions" / session_id / "benchmark" / "benchmark_summary.json"
    return json.loads(summary_path.read_text(encoding="utf-8"))[0]


def _repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _print_decisions(summary: dict, contract: dict) -> None:
    decisions_path = _repo_path(summary["persistence_dir"]) / "rl_decisions.jsonl"
    rows = [json.loads(line) for line in decisions_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    print("  target configuration at each decision, the action chosen, and its reward:")
    print(f"  {'step':>4}  {'coverage':>8}  {'sample':>6}  {'freshness':>9}  {'action':<20} {'reward':>7}  explored")
    for row in rows:
        state = row["state"]
        print(
            f"  {row['iteration']:>4}  {state['coverage']:>8.3f}  {state['sample']:>6.3f}  "
            f"{state['freshness']:>8.1f}s  {row['action']:<20} {row['reward']:>7.3f}  "
            f"{'yes' if row.get('was_exploration') else 'no'}"
        )
    print(
        f"  accepted ranges: coverage [{contract['coverage_min']}, {contract['coverage_max']}], "
        f"sample [{contract['sample_min']}, {contract['sample_max']}], "
        f"freshness [{contract['freshness_min']}, {contract['freshness_max']}] s"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile", default="lax-background", help="Workload profile from config.yaml")
    parser.add_argument("--base-port", type=int, default=8010, help="First of three local node ports")
    args = parser.parse_args()

    _require_git_checkout()
    payload = _profile(args.profile)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = ROOT / "data" / "quickstart" / stamp
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Starting three worker nodes on ports {args.base_port}-{args.base_port + 2}")
    processes, nodes = _start_nodes(args.base_port, log_dir)
    try:
        print(f"[2/4] Training DQN for 24 decisions on one {args.profile} request (1 s per decision)")
        train = _run(f"quickstart_{stamp}_train", nodes, payload, args.profile, mode="train", iterations=24,
                     schedule="2,16,64,4", segment=6, exploration_steps=24)
        _print_decisions(train, payload)
        artifact = sorted(_repo_path(train["model_dir"]).glob("*_dqn.*"))
        if len(artifact) != 1:
            raise SystemExit(f"Expected one trained policy in {train['model_dir']}")

        print("[3/4] Frozen evaluation of the trained policy for 8 decisions (no exploration, no updates)")
        frozen = _run(f"quickstart_{stamp}_evaluate", nodes, payload, args.profile, mode="evaluate", iterations=8,
                      schedule="3,12", segment=4, exploration_steps=24, policy_path=str(artifact[0]))
        _print_decisions(frozen, payload)
    finally:
        for process in processes:
            process.terminate()
        for process in processes:
            process.wait(timeout=10)

    unchanged = frozen.get("policy_unchanged") is True
    same_policy = frozen.get("policy_fingerprint_before") == train.get("policy_fingerprint_after")
    print("[4/4] Summary")
    print(f"  training reward per decision:          {train['total_reward'] / train['steps']:.3f}")
    print(f"  frozen evaluation reward per decision: {frozen['total_reward'] / frozen['steps']:.3f}")
    print(f"  evaluated policy is the trained policy: {'yes' if same_policy else 'NO'}")
    print(f"  policy unchanged during evaluation:     {'yes' if unchanged else 'NO'}")
    print(f"  trained policy artifact: {artifact[0].relative_to(ROOT)}")
    print(f"  decision logs:           {_repo_path(frozen['persistence_dir']).relative_to(ROOT)}/rl_decisions.jsonl")
    if not (unchanged and same_policy):
        raise SystemExit("Frozen-policy check failed")
    print("ARGOS quickstart completed successfully.")


if __name__ == "__main__":
    main()
