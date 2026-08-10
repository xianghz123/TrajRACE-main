#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Invalid JSON root: {path}")
    return obj


def total_parent_count(obj: Dict[str, Any]) -> int:
    total = 0
    for stat in obj.get("splits", {}).values():
        if isinstance(stat, dict):
            total += int(stat.get("num_parent_trajectories", 0))
    return total


def find_summary(
    repo_root: Path,
    dataset: str,
    expected_lmax: int,
) -> Path:
    candidates: List[tuple[int, Path]] = []

    root = repo_root / "data" / "variants"
    if not root.exists():
        raise FileNotFoundError(f"Variant root not found: {root}")

    for path in root.rglob("event_summary.json"):
        try:
            obj = load_json(path)
        except Exception:
            continue

        if str(obj.get("dataset_name", "")).lower() != dataset.lower():
            continue

        if int(obj.get("L_max", -1)) != expected_lmax:
            continue

        if obj.get("sbs_partition_mode") != "transition_preserving_overlap":
            continue

        n = total_parent_count(obj)
        if n > 0:
            candidates.append((n, path))

    if not candidates:
        raise FileNotFoundError(
            f"No canonical event_summary.json found for "
            f"dataset={dataset}, L_max={expected_lmax}"
        )

    # Prefer the largest paper-scale result rather than a smoke run.
    candidates.sort(key=lambda x: (x[0], str(x[1])), reverse=True)

    best_n, best_path = candidates[0]

    print(
        f"[Select] dataset={dataset}, parents={best_n}, "
        f"summary={best_path}"
    )

    if len(candidates) > 1:
        print("[Info] Other matching summaries:")
        for n, p in candidates[1:5]:
            print(f"       parents={n}: {p}")

    return best_path


def aggregate_summary(path: Path) -> Dict[str, Any]:
    obj = load_json(path)

    dataset = str(obj.get("dataset_name", "unknown"))
    lmax = int(obj.get("L_max", -1))
    partition_mode = str(obj.get("sbs_partition_mode", "unknown"))

    splits = obj.get("splits", {})
    if not isinstance(splits, dict):
        raise ValueError(f"No valid 'splits' in {path}")

    num_parent = 0
    num_sbs = 0
    num_single = 0
    num_multi = 0
    weighted_parent_segments = 0.0
    max_q = 0
    min_coverage = 1.0
    max_sbs_segments = 0

    per_split = {}

    for split_name, stat in splits.items():
        if not isinstance(stat, dict):
            continue

        n = int(stat.get("num_parent_trajectories", 0))
        if n <= 0:
            continue

        sbs = int(stat.get("num_output_sbs", 0))
        single = int(stat.get("num_single_sbs_trajectories", 0))
        multi = int(stat.get("num_multi_sbs_trajectories", 0))

        if single + multi != n:
            raise AssertionError(
                f"{dataset}/{split_name}: "
                f"single({single}) + multi({multi}) != parents({n})"
            )

        coverage = float(stat.get("transition_coverage_ratio", 0.0))
        split_max_sbs_segments = int(stat.get("max_sbs_segments", 0))
        split_max_q = int(stat.get("max_sbs_per_trajectory", 0))
        avg_parent_segments = float(stat.get("avg_parent_segments", 0.0))

        num_parent += n
        num_sbs += sbs
        num_single += single
        num_multi += multi

        weighted_parent_segments += n * avg_parent_segments
        max_q = max(max_q, split_max_q)
        min_coverage = min(min_coverage, coverage)
        max_sbs_segments = max(max_sbs_segments, split_max_sbs_segments)

        per_split[split_name] = {
            "num_parent_trajectories": n,
            "num_output_sbs": sbs,
            "fraction_multi_sbs": multi / n,
            "avg_sbs_per_trajectory": sbs / n,
            "max_sbs_per_trajectory": split_max_q,
            "avg_parent_segments": avg_parent_segments,
            "transition_coverage_ratio": coverage,
        }

    if num_parent == 0:
        raise ValueError(f"No trajectories found in {path}")

    result = {
        "dataset": dataset,
        "L_max": lmax,
        "sbs_partition_mode": partition_mode,
        "num_parent_trajectories": num_parent,
        "num_output_sbs": num_sbs,
        "num_single_sbs_trajectories": num_single,
        "num_multi_sbs_trajectories": num_multi,
        "fraction_multi_sbs": num_multi / num_parent,
        "avg_sbs_per_trajectory": num_sbs / num_parent,
        "max_sbs_per_trajectory": max_q,
        "avg_parent_segments": weighted_parent_segments / num_parent,
        "max_sbs_segments": max_sbs_segments,
        "min_transition_coverage_ratio": min_coverage,
        "source": str(path),
        "per_split": per_split,
    }

    return result


def validate(result: Dict[str, Any], expected_lmax: int) -> None:
    if result["L_max"] != expected_lmax:
        raise AssertionError(
            f'L_max={result["L_max"]}, expected {expected_lmax}'
        )

    if result["sbs_partition_mode"] != "transition_preserving_overlap":
        raise AssertionError(
            f'Unexpected SBS mode: {result["sbs_partition_mode"]}'
        )

    if result["max_sbs_segments"] > expected_lmax:
        raise AssertionError(
            f'max_sbs_segments={result["max_sbs_segments"]} '
            f'> L_max={expected_lmax}'
        )

    if abs(result["min_transition_coverage_ratio"] - 1.0) > 1e-12:
        raise AssertionError(
            "Transition coverage is not 1.0: "
            f'{result["min_transition_coverage_ratio"]}'
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Report paper-scale multi-SBS statistics from canonical "
            "TrajRACE event_summary.json files."
        )
    )

    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("/home/hzxiang/code/submit/TrajRACE-main"),
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["porto"],
        help="e.g. --datasets porto oldenburg chengdu",
    )

    parser.add_argument(
        "--lmax",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional CSV output path.",
    )

    args = parser.parse_args()

    repo_root = args.repo_root.resolve()

    results = []

    for dataset in args.datasets:
        summary = find_summary(
            repo_root=repo_root,
            dataset=dataset,
            expected_lmax=args.lmax,
        )

        result = aggregate_summary(summary)
        validate(result, args.lmax)
        results.append(result)

    print()
    print("=" * 100)
    print("Multi-SBS statistics")
    print("=" * 100)

    header = (
        "Dataset",
        "N(parent)",
        "#SBS",
        "P(q>1)",
        "E[q]",
        "max(q)",
        "AvgParentSeg",
        "Coverage",
    )
    print(
        f"{header[0]:12s} {header[1]:>10s} {header[2]:>10s} "
        f"{header[3]:>10s} {header[4]:>10s} "
        f"{header[5]:>10s} {header[6]:>14s} {header[7]:>10s}"
    )

    for r in results:
        print(
            f'{r["dataset"]:12s} '
            f'{r["num_parent_trajectories"]:10d} '
            f'{r["num_output_sbs"]:10d} '
            f'{r["fraction_multi_sbs"]:10.4f} '
            f'{r["avg_sbs_per_trajectory"]:10.4f} '
            f'{r["max_sbs_per_trajectory"]:10d} '
            f'{r["avg_parent_segments"]:14.2f} '
            f'{r["min_transition_coverage_ratio"]:10.6f}'
        )

    print()
    print("Rebuttal-ready text:")
    for r in results:
        print(
            f'{r["dataset"]}: '
            f'{100.0 * r["fraction_multi_sbs"]:.2f}% of parent trajectories '
            f'are partitioned into multiple SBSs under L_m={r["L_max"]} '
            f'(E[q]={r["avg_sbs_per_trajectory"]:.2f}, '
            f'max q={r["max_sbs_per_trajectory"]}).'
        )

    if args.out is not None:
        out_path = args.out.resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "dataset",
            "L_max",
            "num_parent_trajectories",
            "num_output_sbs",
            "fraction_multi_sbs",
            "avg_sbs_per_trajectory",
            "max_sbs_per_trajectory",
            "avg_parent_segments",
            "max_sbs_segments",
            "min_transition_coverage_ratio",
            "source",
        ]

        with out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for r in results:
                writer.writerow({k: r[k] for k in fieldnames})

        print(f"\n[Saved] {out_path}")


if __name__ == "__main__":
    main()