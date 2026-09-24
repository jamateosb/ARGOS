#!/usr/bin/env python3
"""Analyze completed campaigns and build the result tables in one step.

    python scripts/analyze_results.py \
        --controlled data/campaigns/thread_primary/campaign_index.json \
        --controlled data/campaigns/thread_generalization/campaign_index.json \
        --controlled data/campaigns/process_primary/campaign_index.json \
        --controlled data/campaigns/process_generalization/campaign_index.json \
        --live data/live_campaigns/live_thread/campaign_index.json

For every runtime found in the controlled campaigns, the learned controllers
are compared against each comparator (Static, Threshold, Best-fixed); the live
campaign is analyzed as paired seeds; then the result tables under results/
are rebuilt. Per-run analysis files are written to data/analysis/.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from argos.analysis import controlled, live, tables  # noqa: E402

COMPARATORS = ("static", "threshold", "best_fixed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--controlled", type=Path, action="append", required=True,
                        help="campaign_index.json of a controlled campaign; repeat for every campaign")
    parser.add_argument("--live", type=Path, required=True, help="campaign_index.json of the live campaign")
    parser.add_argument("--analysis-dir", type=Path, default=ROOT / "data" / "analysis")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--evidence-root", type=Path, default=None,
                        help="Directory containing the campaigns' data/ tree, if it was moved")
    args = parser.parse_args()
    root = ["--evidence-root", str(args.evidence_root)] if args.evidence_root else []

    by_runtime: dict[str, list[Path]] = defaultdict(list)
    for index_path in args.controlled:
        runtime = json.loads(index_path.read_text(encoding="utf-8"))["runtime"]
        by_runtime[runtime].append(index_path)

    for runtime, indexes in sorted(by_runtime.items()):
        for comparator in COMPARATORS:
            print(f"controlled analysis: {runtime} runtime, comparator {comparator}")
            controlled.main(
                [arg for path in indexes for arg in ("--campaign-index", str(path))]
                + ["--primary-baseline", comparator,
                   "--output-dir", str(args.analysis_dir / f"{runtime}_{comparator}")]
                + root
            )

    print("live analysis")
    live.main(["--campaign-index", str(args.live), "--output-dir", str(args.analysis_dir / "live")] + root)

    print("result tables")
    tables.main(
        [arg for path in args.controlled for arg in ("--controlled-index", str(path))]
        + ["--live-index", str(args.live),
           "--live-analysis", str(args.analysis_dir / "live" / "live_run_metrics.csv"),
           "--output-dir", str(args.output_dir)]
        + root
    )


if __name__ == "__main__":
    main()
