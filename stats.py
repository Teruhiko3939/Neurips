"""stats.py — Calculate mean ± SD per meeting across numbered run folders (1, 2, 3, ...).

Usage:
  python stats.py                   # ami_single_4, saves to results/ami_single_4/stats.json
  python stats.py -f single_4       # saves to results/single_4/stats.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

_TS = r"\d{8}_\d{6}"
RAGAS_PAT = re.compile(rf"eval_RAGAS_(.+)_{_TS}\.jsonl$")
EVAL_PAT  = re.compile(rf"eval_(.+)_{_TS}\.jsonl$")

EVAL_SECTIONS  = ("score_in_out", "score_ref_out", "score_arg_div", "score_opp", "score_faith")
RAGAS_SECTIONS = ("score_ragas_in_out", "score_ragas_ref_out")


def _read_jsonl_first(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                return json.loads(line)
    return {}


def _flatten(d: dict[str, Any], prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        elif isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v):
            out[key] = float(v)
    return out


def load_runs(
    folder_path: Path,
) -> dict[str, list[dict[str, float]]]:
    """Return {meeting_id: [metrics_run1, metrics_run2, ...]} across all numbered run dirs."""

    run_dirs = sorted(
        [d for d in folder_path.iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda d: int(d.name),
    )
    if not run_dirs:
        print(f"[ERROR] No numbered run subdirectories in {folder_path}", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Found {len(run_dirs)} run dir(s): {[d.name for d in run_dirs]}", file=sys.stderr)

    per_meeting: dict[str, list[dict[str, float]]] = defaultdict(list)

    for run_dir in run_dirs:
        eval_map:  dict[str, Path] = {}
        ragas_map: dict[str, Path] = {}

        for f in run_dir.glob("*.jsonl"):
            if m := RAGAS_PAT.fullmatch(f.name):
                ragas_map[m.group(1)] = f
            elif m := EVAL_PAT.fullmatch(f.name):
                eval_map[m.group(1)] = f

        all_ids = set(eval_map) | set(ragas_map)

        for mid in sorted(all_ids):
            metrics: dict[str, float] = {}

            if mid in eval_map:
                rec = _read_jsonl_first(eval_map[mid])
                for section in EVAL_SECTIONS:
                    sec = rec.get(section)
                    if isinstance(sec, dict):
                        metrics.update(_flatten(sec, section))

            if mid in ragas_map:
                rec = _read_jsonl_first(ragas_map[mid])
                for section in RAGAS_SECTIONS:
                    sec = rec.get(section)
                    if isinstance(sec, dict):
                        metrics.update(_flatten(sec, section))

            if metrics:
                per_meeting[mid].append(metrics)

    return per_meeting


def build_output(per_meeting: dict[str, list[dict[str, float]]]) -> dict:
    """Build {meeting_id: {metric: {mean, sd, n}}} output dict."""
    all_metrics: set[str] = set()
    for runs in per_meeting.values():
        for run in runs:
            all_metrics.update(run.keys())
    all_metrics_sorted = sorted(all_metrics)

    output: dict = {}
    for mid in sorted(per_meeting.keys()):
        runs = per_meeting[mid]
        n = len(runs)
        metrics_out: dict = {}
        for metric in all_metrics_sorted:
            vals = [r[metric] for r in runs if metric in r]
            if len(vals) > 1:
                mu = mean(vals)
                sd = pstdev(vals)
                metrics_out[metric] = {"mean": round(mu, 6), "sd": round(sd, 6), "n": len(vals)}
        output[mid] = {"n_runs": n, "metrics": metrics_out}
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate mean ± SD per meeting across run sub-folders.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "-f", "--folder", default="ami_single_4",
        help="Sub-folder under results/ that contains numbered run dirs (default: ami_single_4).",
    )
    args = parser.parse_args()

    base = Path(__file__).resolve().parent / "results"
    folder_path = base / args.folder
    if not folder_path.is_dir():
        print(f"[ERROR] Folder not found: {folder_path}", file=sys.stderr)
        sys.exit(1)

    per_meeting = load_runs(folder_path=folder_path)

    if not per_meeting:
        print("[INFO] No evaluation data found.", file=sys.stderr)
        sys.exit(0)

    print(f"[INFO] {len(per_meeting)} meeting(s) loaded.", file=sys.stderr)

    result = build_output(per_meeting)
    json_text = json.dumps(result, ensure_ascii=False, indent=2)

    out_path = folder_path / "stats.json"
    out_path.write_text(json_text, encoding="utf-8")
    print(f"[INFO] Saved → {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
