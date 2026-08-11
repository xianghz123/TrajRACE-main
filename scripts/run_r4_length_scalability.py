#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R4-W8 parent-trajectory-length scalability experiment for TrajRACE.

Goal
----
Measure how TrajRACE's *post-map-matching* modeling/reporting/recovery cost
changes as parent mapped road-segment trajectories become longer, while
keeping the actual mechanism fixed (especially L_max=30 and B1p0).

Why this design
---------------
* We do NOT vary L_max.  Changing L_max would change SBS definition, count
  domain, and scheduler admissibility, confounding "length scalability" with
  mechanism sensitivity.
* We start from the existing map-matched parent segment-sequence JSONL files.
  Thus map matching is excluded from this timing experiment, matching the R1
  rebuttal's post-map-matching timing boundary.
* We form length strata, sample the same number of parent trajectories per
  stratum, write isolated fixed dataset roots, and run the existing canonical
  scripts unchanged:
      build_events.py -> compute_transition_risk.py ->
      privatize_riskaware.py -> recover_statistics.py --skip_joint
* One warm-up plus repeated measured runs are supported to reduce cache/startup
  noise.  Only aggregate timing/volume files should be published.

The script never publishes selected trajectory records; they remain under the
local rebuttal_runs work directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from trajrace.versioning_utils import apply_versioning


def resolve_path(value: str | Path) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    return (PROJECT_ROOT / p).resolve()


def load_yaml(path: Path) -> Dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return obj


def save_yaml(path: Path, obj: Mapping[str, Any]) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(dict(obj), f, allow_unicode=True, sort_keys=False)


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for row in rows:
        for k in row.keys():
            if k not in fields:
                fields.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as exc:
                raise ValueError(f"Invalid JSONL at {path}:{lineno}: {exc}") from exc
            if isinstance(obj, dict):
                yield obj


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def segment_length(rec: Mapping[str, Any]) -> int:
    for key in ("segments", "segment_seq", "road_segments", "segment_sequence"):
        x = rec.get(key)
        if isinstance(x, list):
            return len(x)
    # Last-resort support for records carrying an explicit length.
    for key in ("num_segments", "traj_len", "length"):
        try:
            return int(rec[key])
        except Exception:
            pass
    raise KeyError("Cannot infer parent road-segment length; expected a 'segments' list.")


def quantile_edges(lengths: Sequence[int], num_bins: int) -> List[int]:
    if num_bins < 2:
        return []
    xs = sorted(int(x) for x in lengths)
    if not xs:
        return []
    edges: List[int] = []
    for j in range(1, num_bins):
        q = j / num_bins
        idx = min(len(xs) - 1, max(0, int(math.ceil(q * len(xs))) - 1))
        edges.append(xs[idx])
    # Non-decreasing is okay; later labels show actual min/max.  Collapse exact
    # duplicate edges to avoid empty nominal intervals.
    out: List[int] = []
    for e in edges:
        if not out or e > out[-1]:
            out.append(e)
    return out


def assign_bin(length: int, edges: Sequence[int]) -> int:
    for i, e in enumerate(edges):
        if length <= e:
            return i
    return len(edges)


def bin_label(idx: int, edges: Sequence[int]) -> str:
    if not edges:
        return "all"
    if idx == 0:
        return f"le{edges[0]}"
    if idx == len(edges):
        return f"gt{edges[-1]}"
    return f"{edges[idx-1]+1}_{edges[idx]}"


class ReservoirRecords:
    def __init__(self, capacity: int, rng: random.Random):
        self.capacity = int(capacity)
        self.rng = rng
        self.items: List[Dict[str, Any]] = []
        self.seen = 0

    def add(self, rec: Dict[str, Any]) -> None:
        self.seen += 1
        if len(self.items) < self.capacity:
            self.items.append(rec)
            return
        j = self.rng.randrange(self.seen)
        if j < self.capacity:
            self.items[j] = rec


def split_records(records: Sequence[Dict[str, Any]], seed: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    xs = list(records)
    random.Random(seed).shuffle(xs)
    n = len(xs)
    n_train = int(round(0.8 * n))
    n_valid = int(round(0.1 * n))
    if n_train + n_valid > n:
        n_valid = max(0, n - n_train)
    return xs[:n_train], xs[n_train:n_train+n_valid], xs[n_train+n_valid:]


def run_stage(cmd: Sequence[str], log_path: Path, verbose: bool) -> float:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            list(cmd),
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            log.write(line)
            if verbose:
                print(line, end="")
        rc = proc.wait()
    elapsed = time.perf_counter() - t0
    if rc != 0:
        raise RuntimeError(f"Stage failed rc={rc}: {' '.join(cmd)}\nSee {log_path}")
    return elapsed


def count_event_volume(root: Path) -> Tuple[int, int]:
    event_dir = root / "processed" / "events"
    n_sbs = 0
    n_trans = 0
    for split in ("train", "valid", "test"):
        p = event_dir / f"{split}_events.jsonl"
        if not p.exists():
            raise FileNotFoundError(p)
        for rec in iter_jsonl(p):
            n_sbs += 1
            tr = rec.get("transition_events", [])
            if isinstance(tr, list):
                n_trans += len(tr)
    return n_sbs, n_trans


def mean_std(xs: Sequence[float]) -> Tuple[Optional[float], Optional[float]]:
    if not xs:
        return None, None
    mean = statistics.fmean(xs)
    std = statistics.stdev(xs) if len(xs) >= 2 else 0.0
    return mean, std


def linear_fit_r2(xs: Sequence[float], ys: Sequence[float]) -> Dict[str, Optional[float]]:
    if len(xs) < 2 or len(xs) != len(ys):
        return {"slope": None, "intercept": None, "r2": None}
    mx = statistics.fmean(xs)
    my = statistics.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return {"slope": None, "intercept": my, "r2": None}
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    intercept = my - slope * mx
    pred = [intercept + slope * x for x in xs]
    sse = sum((y - p) ** 2 for y, p in zip(ys, pred))
    sst = sum((y - my) ** 2 for y in ys)
    r2 = 1.0 - sse / sst if sst > 0 else 1.0
    return {"slope": slope, "intercept": intercept, "r2": r2}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="R4 length-stratified post-map-matching TrajRACE scalability experiment.")
    p.add_argument("--dataset_config", default="configs/dataset.yaml")
    p.add_argument("--exp_config", default="configs/exp_main.yaml")
    p.add_argument("--profile", default="B1p0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run_root", default="rebuttal_runs/r4/length_scalability")
    p.add_argument("--per_bin", type=int, default=20000,
                   help="Maximum equal parent-trajectory sample size per length bin.")
    p.add_argument("--num_bins", type=int, default=4,
                   help="Number of quantile-like length bins when --edges is omitted.")
    p.add_argument("--edges", nargs="*", type=int, default=None,
                   help="Optional explicit inclusive upper edges, e.g. --edges 15 30 60.")
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--dry_run", action="store_true",
                   help="Only inspect parent-length distribution and proposed bins; run no TrajRACE stages.")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--plot", action="store_true")
    return p.parse_args()


def main() -> None:
    a = parse_args()
    if a.per_bin <= 0 or a.repeats <= 0 or a.warmup < 0:
        raise ValueError("--per_bin and --repeats must be >0; --warmup must be >=0")

    dpath = resolve_path(a.dataset_config)
    epath = resolve_path(a.exp_config)
    raw_d = load_yaml(dpath)
    raw_e = load_yaml(epath)

    # Resolve the exact canonical already-map-matched parent segment-sequence files.
    dcfg, _ = apply_versioning(raw_d, raw_e)
    seq_paths = [resolve_path(dcfg[k]) for k in ("segment_seq_train", "segment_seq_valid", "segment_seq_test")]
    missing = [str(p) for p in seq_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Canonical mapped segment-sequence files are missing. Run preprocessing first, or use the exact configs from the R1 run.\n"
            + "\n".join(missing)
        )

    dataset = str(dcfg.get("dataset_name", raw_d.get("dataset_name", "dataset")))
    L_max = int(raw_e.get("L_max", 30))
    if L_max != 30:
        print(f"[Warning] L_max={L_max}. For the paper-matched R4 experiment, use the canonical L_max=30.")

    # Pass 1: mapped parent length distribution.
    lengths: List[int] = []
    for p in seq_paths:
        for rec in iter_jsonl(p):
            lengths.append(segment_length(rec))
    if not lengths:
        raise RuntimeError("No mapped parent trajectories found.")

    if a.edges:
        edges = sorted(set(int(x) for x in a.edges))
    else:
        edges = quantile_edges(lengths, a.num_bins)
    num_bins = len(edges) + 1

    avail = Counter(assign_bin(x, edges) for x in lengths)
    print(f"[Info] dataset={dataset}, mapped parents={len(lengths)}, min/median/max={min(lengths)}/{statistics.median(lengths)}/{max(lengths)}")
    print(f"[Info] edges={edges}")
    for b in range(num_bins):
        print(f"[Info] bin {b} {bin_label(b, edges)} available={avail[b]}")

    if a.dry_run:
        out = {
            "dataset": dataset,
            "num_parent_trajectories": len(lengths),
            "min_length": min(lengths),
            "median_length": statistics.median(lengths),
            "max_length": max(lengths),
            "edges": edges,
            "available_per_bin": {bin_label(b, edges): int(avail[b]) for b in range(num_bins)},
        }
        root = resolve_path(a.run_root) / dataset
        save_json(root / "length_distribution_dry_run.json", out)
        print(json.dumps(out, indent=2))
        print(f"DRY RUN DONE -> {root}")
        return

    min_available = min(avail[b] for b in range(num_bins))
    equal_n = min(int(a.per_bin), int(min_available))
    if equal_n < 100:
        raise RuntimeError(f"Smallest length bin has only {min_available} records; adjust --edges/--num_bins.")
    if equal_n < a.per_bin:
        print(f"[Info] Using equal_n={equal_n} per bin because smallest bin has {min_available} records.")

    # Pass 2: deterministic reservoir sample equal_n parent records per bin.
    reservoirs = {
        b: ReservoirRecords(equal_n, random.Random(a.seed + 1009 * b))
        for b in range(num_bins)
    }
    for p in seq_paths:
        for rec in iter_jsonl(p):
            b = assign_bin(segment_length(rec), edges)
            reservoirs[b].add(rec)

    root = resolve_path(a.run_root) / dataset
    root.mkdir(parents=True, exist_ok=True)
    result_rows: List[Dict[str, Any]] = []
    per_repeat_rows: List[Dict[str, Any]] = []

    for b in range(num_bins):
        label = bin_label(b, edges)
        selected = list(reservoirs[b].items)
        if len(selected) != equal_n:
            raise RuntimeError(f"Bin {label}: expected {equal_n} sampled records, got {len(selected)}")

        parent_lengths = [segment_length(x) for x in selected]
        train, valid, test = split_records(selected, a.seed + b)

        bin_root = root / label
        work_root = (bin_root / "work").resolve()
        seq_dir = work_root / "intermediate" / "segment_sequences"
        write_jsonl(seq_dir / "train.jsonl", train)
        write_jsonl(seq_dir / "valid.jsonl", valid)
        write_jsonl(seq_dir / "test.jsonl", test)

        # Dataset config remains the canonical public-resource config.  The fixed
        # root below redirects only versioned private/intermediate outputs.
        dgen = bin_root / "configs" / "dataset.generated.yaml"
        save_yaml(dgen, raw_d)

        egen_obj = dict(raw_e)
        egen_obj["methods"] = ["riskaware"]
        egen_obj["active_budget_profile"] = str(a.profile)
        egen_obj["default_budget_profile"] = str(a.profile)
        egen_obj["random_seed"] = int(a.seed)
        egen_obj["enable_recovery_dependence_diagnostic"] = False
        egen_obj["save_debug_files"] = False
        egen_obj["fixed_dataset_variant"] = f"r4len_{dataset}_{label}_seed{a.seed}"
        egen_obj["fixed_dataset_root"] = str(work_root)
        egen = bin_root / "configs" / "exp.generated.yaml"
        save_yaml(egen, egen_obj)

        stage_cmds = {
            "events": [a.python, "scripts/build_events.py", "--dataset_config", str(dgen), "--exp_config", str(egen)],
            "risk": [a.python, "scripts/compute_transition_risk.py", "--dataset_config", str(dgen), "--exp_config", str(egen)],
            "riskaware": [a.python, "scripts/privatize_riskaware.py", "--dataset_config", str(dgen), "--exp_config", str(egen)],
            "recovery": [a.python, "scripts/recover_statistics.py", "--dataset_config", str(dgen), "--exp_config", str(egen), "--skip_joint"],
        }

        total_rounds = a.warmup + a.repeats
        measured: Dict[str, List[float]] = {k: [] for k in stage_cmds}
        measured["total"] = []

        for r in range(total_rounds):
            warm = r < a.warmup
            tag = f"warmup{r+1}" if warm else f"repeat{r-a.warmup+1}"
            print(f"\n[{dataset}/{label}] {tag}")
            round_total = 0.0
            for stage, cmd in stage_cmds.items():
                sec = run_stage(cmd, bin_root / "logs" / f"{tag}_{stage}.log", a.verbose)
                print(f"  {stage}: {sec:.3f}s")
                round_total += sec
                if not warm:
                    measured[stage].append(sec)
            if not warm:
                measured["total"].append(round_total)
                per_repeat_rows.append({
                    "dataset": dataset,
                    "length_bin": label,
                    "repeat": r - a.warmup + 1,
                    "n_parent": equal_n,
                    "mean_parent_segments": statistics.fmean(parent_lengths),
                    "events_sec": measured["events"][-1],
                    "risk_sec": measured["risk"][-1],
                    "riskaware_sec": measured["riskaware"][-1],
                    "recovery_sec": measured["recovery"][-1],
                    "total_sec": round_total,
                })

        n_sbs, n_trans = count_event_volume(work_root)
        row: Dict[str, Any] = {
            "dataset": dataset,
            "length_bin": label,
            "edge_lower_exclusive": None if b == 0 else edges[b-1],
            "edge_upper_inclusive": None if b == len(edges) else edges[b],
            "n_parent": equal_n,
            "min_parent_segments": min(parent_lengths),
            "mean_parent_segments": statistics.fmean(parent_lengths),
            "median_parent_segments": statistics.median(parent_lengths),
            "max_parent_segments": max(parent_lengths),
            "total_parent_segments": sum(parent_lengths),
            "num_sbs": n_sbs,
            "mean_sbs_per_parent": n_sbs / equal_n,
            "num_transition_reports": n_trans,
            "mean_transition_reports_per_parent": n_trans / equal_n,
            "warmup_runs": a.warmup,
            "measured_repeats": a.repeats,
        }
        for stage in ("events", "risk", "riskaware", "recovery", "total"):
            m, s = mean_std(measured[stage])
            row[f"{stage}_sec_mean"] = m
            row[f"{stage}_sec_std"] = s
        row["total_sec_per_transition"] = (
            row["total_sec_mean"] / n_trans if n_trans > 0 and row["total_sec_mean"] is not None else None
        )
        result_rows.append(row)
        save_csv(root / "length_scalability.csv", result_rows)
        save_csv(root / "length_scalability_repeats.csv", per_repeat_rows)
        save_json(root / "length_scalability.json", result_rows)

    # Fit aggregate timing against actual transition-report volume and mean parent length.
    fit = {
        "dataset": dataset,
        "fit_total_time_vs_transition_reports": linear_fit_r2(
            [float(r["num_transition_reports"]) for r in result_rows],
            [float(r["total_sec_mean"]) for r in result_rows],
        ),
        "fit_total_time_vs_mean_parent_segments": linear_fit_r2(
            [float(r["mean_parent_segments"]) for r in result_rows],
            [float(r["total_sec_mean"]) for r in result_rows],
        ),
        "note": (
            "Timing excludes raw preprocessing/map matching and fixed-size synthesis. "
            "Mechanism parameters, including L_max and privacy profile, are fixed across bins."
        ),
    }
    save_json(root / "length_scalability_fit.json", fit)

    if a.plot:
        try:
            import matplotlib.pyplot as plt

            xs = [r["mean_parent_segments"] for r in result_rows]
            ys = [r["total_sec_mean"] for r in result_rows]
            plt.figure(figsize=(6.2, 4.2))
            plt.plot(xs, ys, marker="o")
            plt.xlabel("Mean parent trajectory length (road segments)")
            plt.ylabel("Post-map-matching time (s)")
            plt.title(f"TrajRACE length scalability: {dataset}")
            plt.tight_layout()
            plt.savefig(root / "length_scalability.png", dpi=180)
            plt.close()
        except Exception as exc:
            print(f"[Warning] plot skipped: {exc}")

    print(json.dumps(result_rows, ensure_ascii=False, indent=2))
    print(json.dumps(fit, ensure_ascii=False, indent=2))
    print(f"DONE -> {root}")


if __name__ == "__main__":
    main()
