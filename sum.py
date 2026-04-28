from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from scipy import stats


def _flatten(d: dict[str, Any], prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        elif isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v):
            out[key] = float(v)
    return out


def _iter_jsonl_records(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _load_folder(
    base: Path,
    folder: str,
    filename_init: str,
    score_sections: tuple[str, ...],
) -> dict[str, dict[str, float]]:
    """Return {meeting_id: {metric: value}} for all eval files in *folder*."""
    folder_path = base / folder
    files = sorted(folder_path.glob(f"{filename_init}*.jsonl"))
    if not files:
        print(f"[WARN] No {filename_init}*.jsonl in {folder_path}", file=sys.stderr)
        return {}
    data: dict[str, dict[str, float]] = {}
    for fp in files:
        for rec in _iter_jsonl_records(fp):
            meeting_id = rec.get("meeting_id", fp.stem)
            metrics: dict[str, float] = {}
            for section in score_sections:
                sec = rec.get(section)
                if isinstance(sec, dict):
                    metrics.update(_flatten(sec, section))
            if metrics:
                data[meeting_id] = metrics
    return data


def _load_stats_folder(base: Path, folder: str) -> dict[str, dict[str, float]]:
    """Return {meeting_id: {metric: mean_value}} from stats.json in *folder*.

    Only metrics that have a recorded mean are included.
    """
    stats_path = base / folder / "stats.json"
    if not stats_path.exists():
        print(f"[WARN] stats.json not found: {stats_path}", file=sys.stderr)
        return {}
    with stats_path.open(encoding="utf-8") as f:
        raw: dict = json.load(f)
    data: dict[str, dict[str, float]] = {}
    for meeting_id, body in raw.items():
        metrics_block = body.get("metrics", {})
        metrics: dict[str, float] = {}
        for metric, vals in metrics_block.items():
            if isinstance(vals, dict) and "mean" in vals:
                v = vals["mean"]
                if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v):
                    metrics[metric] = float(v)
        if metrics:
            data[meeting_id] = metrics
    return data


def _load_stats_folder_with_sd(base: Path, folder: str) -> dict[str, dict[str, dict]]:
    """Return {meeting_id: {metric: {mean, sd, n}}} from stats.json in *folder*."""
    stats_path = base / folder / "stats.json"
    if not stats_path.exists():
        print(f"[WARN] stats.json not found: {stats_path}", file=sys.stderr)
        return {}
    with stats_path.open(encoding="utf-8") as f:
        raw: dict = json.load(f)
    data: dict[str, dict[str, dict]] = {}
    for meeting_id, body in raw.items():
        metrics_block = body.get("metrics", {})
        metrics: dict[str, dict] = {}
        for metric, vals in metrics_block.items():
            if not isinstance(vals, dict) or "mean" not in vals:
                continue
            mu = vals["mean"]
            sd = vals.get("sd", 0.0)
            n  = vals.get("n", 1)
            if not (isinstance(mu, (int, float)) and not isinstance(mu, bool) and math.isfinite(mu)):
                continue
            metrics[metric] = {
                "mean": float(mu),
                "sd":   float(sd) if isinstance(sd, (int, float)) and math.isfinite(sd) else 0.0,
                "n":    int(n) if isinstance(n, int) else 1,
            }
        if metrics:
            data[meeting_id] = metrics
    return data


def main() -> None:
    base = Path(__file__).resolve().parent / "results"

    # ---- settings --------------------------------------------------------
    RAGAS_eval = True
    # Use stats.json (mean per meeting across runs) instead of raw eval files.
    # Requires stats.py to have been run beforehand for each folder.
    use_stats = True
    # Folders to compare (for per-sample / Wilcoxon output).
    # Set to None to fall back to aggregate summary of a single folder.
    folders: list[str] | None = ["multi_noaf", "multi_af"]
    # ["ami_multi_noaf", "ami_multi_af"]  # e.g. None, ["single_4", "multi_af"]
    # Single-folder aggregate mode (used when folders is None)
    single_folder = "multi_af"
    # ----------------------------------------------------------------------

    filename_init = "eval_RAGAS_" if RAGAS_eval else "eval_"
    if RAGAS_eval:
        score_sections: tuple[str, ...] = ("score_ragas_in_out", "score_ragas_ref_out")
    else:
        score_sections = ("score_in_out", "score_ref_out", "score_arg_div", "score_opp", "score_faith")

    # ---- per-sample output (Wilcoxon mode) --------------------------------
    if folders is not None:
        if use_stats:
            all_data: dict[str, dict[str, dict[str, float]]] = {
                f: _load_stats_folder(base, f) for f in folders
            }
        else:
            all_data = {
                f: _load_folder(base, f, filename_init, score_sections) for f in folders
            }

        all_metrics: set[str] = set()
        for fd in all_data.values():
            for m in fd.values():
                all_metrics.update(m.keys())
        all_metrics_sorted = sorted(all_metrics)

        all_meetings: set[str] = set()
        for fd in all_data.values():
            all_meetings.update(fd.keys())
        all_meetings_sorted = sorted(all_meetings)

        # print("meeting_id,folder," + ",".join(all_metrics_sorted))
        for meeting in all_meetings_sorted:
            for folder in folders:
                fd = all_data.get(folder, {})
                if meeting not in fd:
                    continue
                values = [str(fd[meeting].get(m, "")) for m in all_metrics_sorted]
                # print(f"{meeting},{folder}," + ",".join(values))

        # ---- Wilcoxon signed-rank test (multi_af vs others) ---------------
        # Use single_folder as REF if it is in folders, otherwise fall back to
        # the first folder containing "multi_af".
        if single_folder in folders:
            REF = single_folder
        else:
            REF = next((f for f in folders if "multi_af" in f), None)
        if REF and len(folders) > 1:
            ref_data = all_data[REF]
            other_folders = [f for f in folders if f != REF]
            print()
            col_w = 50
            col_n = 8
            header = f"{'metric':{col_w}} {'folder':{col_w}} {'n':>{col_n}} {'stat':>10} {'p-value':>10} {'sig':>5}"
            print(header)
            print("-" * len(header))
            for metric in all_metrics_sorted:
                for other in other_folders:
                    other_data = all_data[other]
                    common = sorted(set(ref_data) & set(other_data))
                    xs = [ref_data[m][metric] for m in common if metric in ref_data[m] and metric in other_data[m]]
                    ys = [other_data[m][metric] for m in common if metric in ref_data[m] and metric in other_data[m]]
                    n = len(xs)
                    if n < 2:
                        continue
                    try:
                        result = stats.wilcoxon(xs, ys, alternative="two-sided")
                        stat, p = result.statistic, result.pvalue
                        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                        print(f"{metric:{col_w}} {other:{col_w}} {n:>{col_n}} {stat:>10.3f} {p:>10.4f} {sig:>5}")
                    except ValueError as e:
                        print(f"{metric:{col_w}} {other:{col_w}} {n:>{col_n}} {'(skip: ' + str(e) + ')':>27}")

        # ---- Fligner-Killeen test (variance) for selected metrics ------------------
        FLIGNER_KILLEEN_METRICS = {
            "score_in_out.N_view",
            "score_in_out.Lexical_Diversity",
            "score_opp.Oppositionality",
        }
        fligner_killeen_targets = [m for m in all_metrics_sorted if m in FLIGNER_KILLEEN_METRICS]
        if fligner_killeen_targets and REF:
            print()
            print("[Fligner-Killeen test — variance comparison]")
            header_lv = f"{'metric':{col_w}} {'folder':{col_w}} {'n':>{col_n}} {'stat':>10} {'p-value':>10} {'sig':>5}"
            print(header_lv)
            print("-" * len(header_lv))
            for metric in fligner_killeen_targets:
                for other in other_folders:
                    other_data = all_data[other]
                    common = sorted(set(ref_data) & set(other_data))
                    xs = [ref_data[m][metric] for m in common if metric in ref_data[m] and metric in other_data[m]]
                    ys = [other_data[m][metric] for m in common if metric in ref_data[m] and metric in other_data[m]]
                    n = len(xs)
                    if n < 2:
                        continue
                    try:
                        result = stats.fligner(xs, ys)
                        stat, p = result.statistic, result.pvalue
                        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                        print(f"{metric:{col_w}} {other:{col_w}} {n:>{col_n}} {stat:>10.3f} {p:>10.4f} {sig:>5}")
                    except ValueError as e:
                        print(f"{metric:{col_w}} {other:{col_w}} {n:>{col_n}} {'(skip: ' + str(e) + ')':>27}")
        return

    # ---- aggregate summary (original mode) --------------------------------
    if use_stats:
        # Load per-meeting means AND within-meeting SD/n from stats.json
        folder_data_full = _load_stats_folder_with_sd(base, single_folder)
        if not folder_data_full:
            print(f"[INFO] No stats.json data found: {base / single_folder}")
            return
        # {metric: list of per-meeting {mean, sd, n}}
        agg_full: dict[str, list[dict]] = defaultdict(list)
        for meeting_metrics in folder_data_full.values():
            for metric, vals in meeting_metrics.items():
                agg_full[metric].append(vals)
        print(f"[INFO] meetings={len(folder_data_full)}, source=stats.json (method C: propagated SE)")
        print("-" * 100)
        print(f"{'metric':60} {'grand_mean':>10} {'SE_grand':>10} {'95%CI':>18}")
        print("-" * 100)
        for name in sorted(agg_full.keys()):
            entries = agg_full[name]
            M = len(entries)
            grand_mean = mean(e["mean"] for e in entries) if M else float("nan")
            # SE_grand = (1/M) * sqrt( sum( (sd_i / sqrt(n_i))^2 ) )
            se_grand = (1.0 / M) * math.sqrt(
                sum((e["sd"] / math.sqrt(e["n"])) ** 2 for e in entries)
            ) if M else float("nan")
            ci_lo = grand_mean - 1.96 * se_grand
            ci_hi = grand_mean + 1.96 * se_grand
            print(f"{name:60} {grand_mean:>10.3f} {se_grand:>10.4f} [{ci_lo:.3f}, {ci_hi:.3f}]")
        return
    else:
        files = sorted((base / single_folder).glob(f"{filename_init}*.jsonl"))
        if not files:
            print(f"[INFO] No {filename_init}*.jsonl files found: {base / single_folder}")
            return
        agg: dict[str, list[float]] = defaultdict(list)
        record_count = 0
        for fp in files:
            for rec in _iter_jsonl_records(fp):
                record_count += 1
                for section in score_sections:
                    sec = rec.get(section)
                    if isinstance(sec, dict):
                        for k, v in _flatten(sec, section).items():
                            agg[k].append(v)
        print(f"[INFO] files={len(files)}, records={record_count}")
    print("-" * 92)
    print(f"{'metric':60} {'mean ± SD':>24}")
    print("-" * 92)

    for name in sorted(agg.keys()):
        vals = agg[name]
        n = len(vals)
        mu = mean(vals) if n else float("nan")
        sd = pstdev(vals) if n > 1 else 0.0
        print(f"{name:60} {mu:.3f} ± {sd:.4f}")


if __name__ == "__main__":
    main()