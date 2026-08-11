#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R4-W6 risk-prevalence postprocessor for TrajRACE.

This script DOES NOT recompute or perturb any private data.  It reads the
already-generated Phase-3 risk JSONL files and produces compact aggregate
statistics suitable for the rebuttal/artifact.

Recommended use
---------------
1) Make sure the canonical events/risk stages already exist.
2) Run this script once per dataset using the SAME dataset/experiment configs
   used by the paper/rebuttal.
3) Publish only the compact JSON/CSV outputs, not the local risk JSONL files.

The script reports both:
  * all transition risk items; and
  * branching transitions only (candidate_size > 1), which are the
    non-degenerate next-hop contexts relevant to conditional successor privacy.

It is deliberately tolerant to minor historical field-name differences, but
it treats the risk files' b_t as authoritative when it is valid.  If b_t is
missing/invalid, it falls back to the fixed public theta_list in exp_main.yaml.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from trajrace.versioning_utils import apply_versioning, pretty_version_summary


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


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
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


def first_present(row: Mapping[str, Any], names: Sequence[str], default: Any = None) -> Any:
    for name in names:
        if name in row and row[name] is not None:
            return row[name]
    return default


def maybe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None


def maybe_int(x: Any) -> Optional[int]:
    try:
        return int(float(x))
    except Exception:
        return None


def get_risk_score(row: Mapping[str, Any]) -> float:
    x = first_present(row, ["R_t", "risk_score", "risk", "r_i"], 0.0)
    return float(maybe_float(x) or 0.0)


def get_candidate_size(row: Mapping[str, Any]) -> Optional[int]:
    x = first_present(row, ["candidate_size", "domain_size", "successor_domain_size", "m_t"], None)
    v = maybe_int(x)
    return v if v is None or v >= 0 else None


def get_component(row: Mapping[str, Any], kind: str) -> float:
    aliases = {
        "endpoint": ["phi_endpoint", "r_endpoint", "r_e", "endpoint_risk"],
        "stay": ["phi_stay", "r_stay", "r_s", "stay_risk"],
        "degree": ["phi_deg", "r_degree", "r_d", "degree_risk"],
    }
    return float(maybe_float(first_present(row, aliases[kind], 0.0)) or 0.0)


def assign_fixed_bucket(score: float, theta_list: Sequence[float], K: int) -> int:
    """Definition-12 style fixed public thresholds, bucket 1 = strongest/highest risk."""
    if K < 2:
        return 1
    theta = [float(x) for x in theta_list]
    if len(theta) != K - 1:
        raise ValueError(f"theta_list must have K-1={K-1} values, got {theta}")
    # Expect decreasing thresholds theta_1 > theta_2 > ...
    for idx, th in enumerate(theta, start=1):
        if score >= th:
            return idx
    return K


def get_target_bucket(row: Mapping[str, Any], theta_list: Sequence[float], K: int) -> int:
    raw = first_present(
        row,
        ["b_t", "target_bucket", "target_bucket_b_t", "bucket_target", "target_k"],
        None,
    )
    b = maybe_int(raw)
    if b is not None:
        # Accept canonical 1-based buckets; also tolerate 0-based historical data.
        if 1 <= b <= K:
            return b
        if 0 <= b < K:
            return b + 1
    return assign_fixed_bucket(get_risk_score(row), theta_list, K)


class Reservoir:
    def __init__(self, capacity: int, seed: int):
        self.capacity = max(1, int(capacity))
        self.rng = random.Random(seed)
        self.items: List[float] = []
        self.seen = 0

    def add(self, value: float) -> None:
        self.seen += 1
        if len(self.items) < self.capacity:
            self.items.append(float(value))
            return
        j = self.rng.randrange(self.seen)
        if j < self.capacity:
            self.items[j] = float(value)


def percentile(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None
    xs = sorted(float(x) for x in values)
    if len(xs) == 1:
        return xs[0]
    pos = min(max(float(q), 0.0), 1.0) * (len(xs) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    a = pos - lo
    return xs[lo] * (1.0 - a) + xs[hi] * a


class ScopeAccumulator:
    def __init__(
        self,
        scope: str,
        K: int,
        lambdas: Mapping[str, float],
        stay_threshold: float,
        sample_size: int,
        seed: int,
    ):
        self.scope = scope
        self.K = K
        self.lambdas = dict(lambdas)
        self.stay_threshold = float(stay_threshold)
        self.n = 0
        self.bucket = Counter()
        self.candidate = Counter()
        self.dominant_high = Counter()
        self.sum_risk = 0.0
        self.sum_components = defaultdict(float)
        self.long_gap = 0
        self.near_endpoint = 0
        self.high_count = 0
        self.risk_sample = Reservoir(sample_size, seed)

    def add(self, row: Mapping[str, Any], bucket: int) -> None:
        self.n += 1
        self.bucket[int(bucket)] += 1

        risk = get_risk_score(row)
        self.sum_risk += risk
        self.risk_sample.add(risk)

        vals = {
            "endpoint": get_component(row, "endpoint"),
            "stay": get_component(row, "stay"),
            "degree": get_component(row, "degree"),
        }
        for k, v in vals.items():
            self.sum_components[k] += v

        cand = get_candidate_size(row)
        if cand is not None:
            if cand >= 4:
                self.candidate["4+"] += 1
            else:
                self.candidate[str(cand)] += 1

        delta_t = maybe_float(first_present(row, ["delta_t", "time_gap", "gap_sec"], 0.0)) or 0.0
        if delta_t >= self.stay_threshold:
            self.long_gap += 1

        ns = maybe_int(first_present(row, ["near_start_flag", "near_start"], 0)) or 0
        ne = maybe_int(first_present(row, ["near_end_flag", "near_end"], 0)) or 0
        if ns or ne:
            self.near_endpoint += 1

        if int(bucket) == 1:
            self.high_count += 1
            weighted = {
                "endpoint": self.lambdas["endpoint"] * vals["endpoint"],
                "stay": self.lambdas["stay"] * vals["stay"],
                "degree": self.lambdas["degree"] * vals["degree"],
            }
            m = max(weighted.values())
            winners = [k for k, v in weighted.items() if abs(v - m) <= 1e-12]
            if len(winners) == 1:
                self.dominant_high[winners[0]] += 1
            else:
                self.dominant_high["tie"] += 1

    def summary(self) -> Dict[str, Any]:
        n = max(self.n, 1)
        sample = self.risk_sample.items
        return {
            "scope": self.scope,
            "num_transitions": self.n,
            "mean_risk": self.sum_risk / n if self.n else None,
            "risk_q10": percentile(sample, 0.10),
            "risk_q25": percentile(sample, 0.25),
            "risk_q50": percentile(sample, 0.50),
            "risk_q75": percentile(sample, 0.75),
            "risk_q90": percentile(sample, 0.90),
            "mean_phi_endpoint": self.sum_components["endpoint"] / n if self.n else None,
            "mean_phi_stay": self.sum_components["stay"] / n if self.n else None,
            "mean_phi_degree": self.sum_components["degree"] / n if self.n else None,
            "long_gap_share": self.long_gap / n if self.n else None,
            "near_endpoint_share": self.near_endpoint / n if self.n else None,
            "bucket_counts": {str(k): int(self.bucket.get(k, 0)) for k in range(1, self.K + 1)},
            "bucket_shares": {str(k): self.bucket.get(k, 0) / n if self.n else None for k in range(1, self.K + 1)},
            "candidate_size_counts": dict(self.candidate),
            "candidate_size_shares": {k: v / n if self.n else None for k, v in sorted(self.candidate.items())},
            "high_risk_count": self.high_count,
            "high_risk_dominant_component_counts": dict(self.dominant_high),
            "high_risk_dominant_component_shares": {
                k: v / self.high_count if self.high_count else None
                for k, v in sorted(self.dominant_high.items())
            },
            "risk_quantiles_sampled": len(sample),
        }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate R4 risk-prevalence statistics from TrajRACE risk JSONL outputs.")
    p.add_argument("--dataset_config", default="configs/dataset.yaml")
    p.add_argument("--exp_config", default="configs/exp_main.yaml")
    p.add_argument(
        "--risk_files",
        nargs="*",
        default=None,
        help="Optional explicit risk JSONL files. Default: resolved train/valid/test risk files from configs.",
    )
    p.add_argument("--output_dir", default="rebuttal_runs/r4/risk_prevalence")
    p.add_argument("--quantile_sample_size", type=int, default=200000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    a = parse_args()
    dpath = resolve_path(a.dataset_config)
    epath = resolve_path(a.exp_config)
    if not dpath.exists():
        raise FileNotFoundError(dpath)
    if not epath.exists():
        raise FileNotFoundError(epath)

    raw_d = load_yaml(dpath)
    raw_e = load_yaml(epath)
    dcfg, ecfg = apply_versioning(raw_d, raw_e)

    dataset = str(dcfg.get("dataset_name", "dataset"))
    K = int(ecfg.get("K", raw_e.get("K", 3)))
    theta = list(ecfg.get("theta_list", raw_e.get("theta_list", [])))
    if len(theta) != K - 1:
        raise ValueError(f"Expected fixed theta_list of length {K-1}; got {theta}")

    lambdas = {
        "endpoint": float(ecfg.get("lambda_e", raw_e.get("lambda_e", 1.0 / 3.0))),
        "stay": float(ecfg.get("lambda_s", raw_e.get("lambda_s", 1.0 / 3.0))),
        "degree": float(ecfg.get("lambda_d", raw_e.get("lambda_d", 1.0 / 3.0))),
    }
    T_stay = float(
        ecfg.get(
            "T_stay",
            ecfg.get("delta_st_sec", raw_e.get("T_stay", raw_e.get("delta_st_sec", 300.0))),
        )
    )

    if a.risk_files:
        risk_paths = [resolve_path(x) for x in a.risk_files]
    else:
        risk_paths = [resolve_path(dcfg[k]) for k in ("risk_train", "risk_valid", "risk_test")]

    missing = [str(p) for p in risk_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing risk JSONL files. Run scripts/compute_transition_risk.py first, or pass --risk_files.\n"
            + "\n".join(missing)
        )

    acc_all = ScopeAccumulator("all", K, lambdas, T_stay, a.quantile_sample_size, a.seed)
    acc_branch = ScopeAccumulator("branching", K, lambdas, T_stay, a.quantile_sample_size, a.seed + 1)

    split_counts: Dict[str, int] = {}
    invalid_bucket_rows = 0
    total_rows = 0

    for path in risk_paths:
        split = path.stem.replace("_risk", "")
        n_split = 0
        for row in iter_jsonl(path):
            total_rows += 1
            n_split += 1
            raw_b = first_present(row, ["b_t", "target_bucket", "target_bucket_b_t"], None)
            b0 = maybe_int(raw_b)
            if b0 is None or not (1 <= b0 <= K):
                invalid_bucket_rows += 1
            b = get_target_bucket(row, theta, K)
            acc_all.add(row, b)
            cand = get_candidate_size(row)
            if cand is not None and cand > 1:
                acc_branch.add(row, b)
        split_counts[split] = n_split

    all_s = acc_all.summary()
    branch_s = acc_branch.summary()

    out_dir = resolve_path(a.output_dir) / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "dataset": dataset,
        "dataset_config": str(dpath),
        "exp_config": str(epath),
        "dataset_variant": dcfg.get("dataset_variant"),
        "dataset_root": dcfg.get("dataset_root"),
        "version_info": pretty_version_summary(raw_d, raw_e),
        "K": K,
        "theta_list": [float(x) for x in theta],
        "lambdas": lambdas,
        "T_stay": T_stay,
        "risk_files": [str(x) for x in risk_paths],
        "split_counts": split_counts,
        "num_rows": total_rows,
        "num_rows_with_missing_or_invalid_stored_bucket": invalid_bucket_rows,
        "note": (
            "Stored b_t is used when valid; otherwise fixed public theta_list is used. "
            "Branching means candidate_size>1. Dominant component uses lambda*phi among bucket-1 transitions."
        ),
        "scopes": {
            "all": all_s,
            "branching": branch_s,
        },
    }
    save_json(out_dir / "risk_prevalence_summary.json", payload)

    bucket_rows: List[Dict[str, Any]] = []
    for s in (all_s, branch_s):
        n = int(s["num_transitions"])
        for b in range(1, K + 1):
            bucket_rows.append(
                {
                    "dataset": dataset,
                    "scope": s["scope"],
                    "bucket": b,
                    "bucket_label": "high" if b == 1 else ("low" if b == K else "medium"),
                    "count": int(s["bucket_counts"].get(str(b), 0)),
                    "share": s["bucket_shares"].get(str(b)),
                    "num_transitions": n,
                }
            )
    save_csv(out_dir / "risk_bucket_prevalence.csv", bucket_rows)

    characteristic_rows: List[Dict[str, Any]] = []
    for s in (all_s, branch_s):
        base = {
            "dataset": dataset,
            "scope": s["scope"],
            "num_transitions": s["num_transitions"],
            "mean_risk": s["mean_risk"],
            "risk_q10": s["risk_q10"],
            "risk_q25": s["risk_q25"],
            "risk_q50": s["risk_q50"],
            "risk_q75": s["risk_q75"],
            "risk_q90": s["risk_q90"],
            "long_gap_share": s["long_gap_share"],
            "near_endpoint_share": s["near_endpoint_share"],
        }
        characteristic_rows.append(base)
    save_csv(out_dir / "risk_characteristics.csv", characteristic_rows)

    dominant_rows: List[Dict[str, Any]] = []
    for s in (all_s, branch_s):
        shares = s["high_risk_dominant_component_shares"]
        counts = s["high_risk_dominant_component_counts"]
        for comp in ("endpoint", "stay", "degree", "tie"):
            dominant_rows.append(
                {
                    "dataset": dataset,
                    "scope": s["scope"],
                    "component": comp,
                    "count": int(counts.get(comp, 0)),
                    "share_among_bucket1": shares.get(comp),
                    "high_risk_count": s["high_risk_count"],
                }
            )
    save_csv(out_dir / "high_risk_dominant_components.csv", dominant_rows)

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"DONE -> {out_dir}")


if __name__ == "__main__":
    main()
