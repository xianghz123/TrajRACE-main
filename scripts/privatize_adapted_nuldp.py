#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""Adapted non-uniform LDP comparator for TrajRACE rebuttal (R1-W3/D3).

Method name: adapted_nuldp

Purpose
-------
The cited non-uniform/heterogeneous LDP works are not end-to-end road-network
trajectory synthesis systems.  This script implements a transparent adapted
comparator on TrajRACE's *same categorical next-hop reporting task*:

  * same SBS/event records;
  * same fixed public successor domain N(u);
  * same start/count privacy parameters;
  * same K ordered transition epsilon levels and epsilon_bucket;
  * same SBS schedule cap B and future-minimum-cost feasibility rule;
  * NO private endpoint/stay/composite risk score is used for allocation.

The target protection level is determined only by PUBLIC current-context
out-degree |N(u)|.  For K=3 the default public rule is:
    |N(u)| <= 2 -> bucket 1 (strongest)
    |N(u)| == 3 -> bucket 2
    |N(u)| >= 4 -> bucket 3 (weakest)
For K != 3, a monotone public degree rule is used; custom public thresholds can
be passed in exp_main.yaml as adapted_nuldp_degree_thresholds.

This is an *adapted non-uniform comparator*, not a claim of exact reproduction
of AAA or R. Du et al.  Its role is to test whether TrajRACE gains merely from
having heterogeneous epsilon levels, versus assigning them from private local
risk context.

Server-visible schema is identical to TrajRACE:
  start      {event_type, x_noisy}
  count      {event_type, x_noisy}
  transition {event_type, u, y, k_noisy}

A private debug trace can be emitted with --save_debug for attack evaluation.
"""

import argparse
import heapq
import json
import math
import os
import random
import shutil
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from trajrace.versioning_utils import apply_versioning, pretty_version_summary
except Exception as exc:  # pragma: no cover
    raise RuntimeError("Place this script under TrajRACE-main/scripts/.") from exc

DEFAULT_DATASET_CONFIG = "configs/dataset.yaml"
DEFAULT_EXP_CONFIG = "configs/exp_main.yaml"
METHOD = "adapted_nuldp"


def resolve_path(path: Optional[str]) -> Optional[Path]:
    if path is None:
        return None
    p = Path(path)
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def load_yaml(path: Path) -> Dict[str, Any]:
    import yaml
    with path.open("r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"YAML root must be mapping: {path}")
    return obj


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON: {path}:{line_no}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Expected JSON object: {path}:{line_no}")
            yield obj


def cfg_alias(exp: Mapping[str, Any], names: Sequence[str], required: bool = True, default: Any = None) -> Any:
    for name in names:
        if name in exp and exp[name] is not None:
            return exp[name]
    if required:
        raise KeyError(f"Missing config key; expected one of {list(names)}")
    return default


def extract_cfg(exp: Mapping[str, Any]) -> Dict[str, Any]:
    B = float(cfg_alias(exp, ["B_total", "B"]))
    eps_start = float(cfg_alias(exp, ["eps_start", "epsilon_s"]))
    eps_count = float(cfg_alias(exp, ["eps_count", "eps_len", "epsilon_c"]))
    eps_bucket = float(cfg_alias(exp, ["eps_bucket", "eps_buck", "epsilon_b"]))
    eps_event_list = [float(x) for x in cfg_alias(exp, ["eps_event_list", "eps_evt_list", "epsilon_e_list"])]
    K = int(cfg_alias(exp, ["K"]))
    L_max = int(cfg_alias(exp, ["L_max", "L_m"]))
    seed = int(cfg_alias(exp, ["random_seed"], required=False, default=42))
    if len(eps_event_list) != K:
        raise ValueError("eps_event_list length must equal K")
    if sorted(eps_event_list) != eps_event_list:
        raise ValueError("eps_event_list must be nondecreasing; smaller epsilon = stronger bucket")
    if K < 2 or L_max < 2:
        raise ValueError("Require K>=2 and L_max>=2")
    min_transition_cost = eps_event_list[0] + eps_bucket
    public_sufficient = eps_start + eps_count + (L_max - 1) * min_transition_cost
    if public_sufficient > B + 1e-10:
        raise ValueError(
            "Canonical schedule feasibility fails even for bucket 1: "
            f"{public_sufficient:.12f} > B={B:.12f}"
        )
    thresholds = exp.get("adapted_nuldp_degree_thresholds")
    if thresholds is not None:
        thresholds = [int(x) for x in thresholds]
        if len(thresholds) != K - 1:
            raise ValueError("adapted_nuldp_degree_thresholds must have K-1 integers")
        if thresholds != sorted(thresholds):
            raise ValueError("adapted_nuldp_degree_thresholds must be increasing")
    return {
        "B_total": B,
        "eps_start": eps_start,
        "eps_count": eps_count,
        "eps_bucket": eps_bucket,
        "eps_event_list": eps_event_list,
        "K": K,
        "L_max": L_max,
        "random_seed": seed,
        "public_degree_thresholds": thresholds,
        "min_transition_cost": min_transition_cost,
        "public_minimum_schedule_spend": public_sufficient,
    }


def rr_sample(true_value: Any, domain: Sequence[Any], epsilon: float, rng: random.Random) -> Any:
    domain = tuple(domain)
    d = len(domain)
    if d <= 0:
        raise ValueError("GRR domain empty")
    if true_value not in domain:
        raise ValueError(f"True value {true_value!r} is outside fixed public domain")
    if d == 1:
        return true_value
    ee = math.exp(float(epsilon))
    p = ee / (ee + d - 1)
    if rng.random() < p:
        return true_value
    while True:
        z = domain[rng.randrange(d)]
        if z != true_value:
            return z


def public_degree_bucket(out_degree: int, K: int, thresholds: Optional[Sequence[int]]) -> int:
    """Public monotone non-uniform assignment; smaller k => stronger protection."""
    m = max(1, int(out_degree))
    if thresholds:
        for k, upper in enumerate(thresholds, 1):
            if m <= int(upper):
                return k
        return K
    if K == 3:
        if m <= 2:
            return 1
        if m == 3:
            return 2
        return 3
    # General monotone fallback: bucket grows with public branching factor.
    return max(1, min(K, m - 1))


def feasible_buckets(C_before: float, remaining_after: int, cfg: Mapping[str, Any]) -> List[int]:
    eps_list = cfg["eps_event_list"]
    eb = float(cfg["eps_bucket"])
    future_min = max(0, int(remaining_after)) * (float(eps_list[0]) + eb)
    out = []
    for k, ee in enumerate(eps_list, 1):
        if float(ee) + eb + future_min <= float(C_before) + 1e-12:
            out.append(k)
    return out


def choose_exec_bucket(target: int, feasible: Sequence[int]) -> int:
    if not feasible:
        raise RuntimeError("Feasible bucket set is empty")
    if target in feasible:
        return int(target)
    stronger_or_equal = [k for k in feasible if k <= target]
    if not stronger_or_equal:
        raise RuntimeError("No feasible bucket no-weaker-than target; bucket 1 should be feasible")
    return max(stronger_or_equal)


def event_start(rec: Mapping[str, Any]) -> str:
    e = rec.get("start_event", {})
    for key in ("st", "e1", "start", "x"):
        if key in e:
            return str(e[key])
    raise KeyError("Cannot find start value in start_event")


def event_count(rec: Mapping[str, Any]) -> int:
    e = rec.get("count_event", rec.get("length_event", {}))
    for key in ("cnt", "L", "count", "length"):
        if key in e:
            return int(e[key])
    # Safe canonical fallback: exact segment count from transition list + 1.
    if "transition_events" in rec:
        return len(rec["transition_events"]) + 1
    raise KeyError("Cannot find exact segment count")


class ExternalShuffleWriter:
    def __init__(self, output_path: Path, rng: random.Random, chunk_size: int) -> None:
        self.output_path = output_path
        self.rng = rng
        self.chunk_size = max(1000, int(chunk_size))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.temp_dir = Path(tempfile.mkdtemp(prefix=".adapted_nuldp_shuffle_", dir=str(output_path.parent)))
        self.buffer: List[Tuple[int, int, str]] = []
        self.chunks: List[Path] = []
        self.seq = 0
        self.count = 0

    def add(self, report: Mapping[str, Any]) -> None:
        key = self.rng.getrandbits(128)
        payload = json.dumps(dict(report), ensure_ascii=False, separators=(",", ":"))
        self.buffer.append((key, self.seq, payload))
        self.seq += 1
        self.count += 1
        if len(self.buffer) >= self.chunk_size:
            self._flush()

    def _flush(self) -> None:
        if not self.buffer:
            return
        self.buffer.sort(key=lambda x: (x[0], x[1]))
        p = self.temp_dir / f"chunk_{len(self.chunks):06d}.txt"
        with p.open("w", encoding="utf-8") as f:
            for key, seq, payload in self.buffer:
                f.write(f"{key:032x}\t{seq:020d}\t{payload}\n")
        self.chunks.append(p)
        self.buffer = []

    @staticmethod
    def _iter_chunk(path: Path) -> Iterator[Tuple[int, int, str]]:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                a, b, payload = line.rstrip("\n").split("\t", 2)
                yield int(a, 16), int(b), payload

    def finalize(self) -> int:
        self._flush()
        with self.output_path.open("w", encoding="utf-8") as out:
            its = [self._iter_chunk(p) for p in self.chunks]
            for _, _, payload in heapq.merge(*its):
                out.write(payload + "\n")
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        return self.count


class DebugWriter:
    def __init__(self, path: Optional[Path]) -> None:
        self.f = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.f = path.open("w", encoding="utf-8")

    def write(self, obj: Mapping[str, Any]) -> None:
        if self.f is not None:
            self.f.write(json.dumps(dict(obj), ensure_ascii=False) + "\n")

    def close(self) -> None:
        if self.f is not None:
            self.f.close()
            self.f = None


def privatize_split(
    split: str,
    event_path: Path,
    report_path: Path,
    debug_path: Optional[Path],
    successor_cache: Mapping[str, Sequence[str]],
    cfg: Mapping[str, Any],
    mechanism_rng: random.Random,
    shuffle_rng: random.Random,
    shuffle_chunk_size: int,
    progress_every: int,
) -> Dict[str, Any]:
    start_domain = tuple(str(x) for x in successor_cache.keys())
    count_domain = tuple(range(1, int(cfg["L_max"]) + 1))
    bucket_domain = tuple(range(1, int(cfg["K"]) + 1))
    writer = ExternalShuffleWriter(report_path, shuffle_rng, shuffle_chunk_size)
    debug = DebugWriter(debug_path)
    n_sbs = n_trans = 0
    target_counts = Counter()
    exec_counts = Counter()
    fallback = 0
    keep_by_target = Counter()
    total_by_target = Counter()
    max_spend = 0.0
    min_margin = float("inf")

    try:
        for rec_idx, rec in enumerate(iter_jsonl(event_path), 1):
            sid = str(rec.get("sbs_id", f"{split}_{rec_idx}"))
            true_start = event_start(rec)
            true_count = event_count(rec)
            transitions = list(rec.get("transition_events", []))
            R = len(transitions)
            if true_start not in successor_cache:
                raise RuntimeError(f"Start outside fixed public road domain: {true_start}")
            if true_count not in count_domain:
                raise RuntimeError(f"Count outside 1..L_max: {true_count}")

            y_start = rr_sample(true_start, start_domain, float(cfg["eps_start"]), mechanism_rng)
            y_count = rr_sample(true_count, count_domain, float(cfg["eps_count"]), mechanism_rng)
            writer.add({"event_type": "start", "x_noisy": y_start})
            writer.add({"event_type": "count", "x_noisy": int(y_count)})
            debug.write({"event_type": "start", "sbs_id": sid, "true_x": true_start,
                         "noisy_x": y_start, "epsilon_used": float(cfg["eps_start"])})
            debug.write({"event_type": "count", "sbs_id": sid, "true_x": true_count,
                         "noisy_x": int(y_count), "epsilon_used": float(cfg["eps_count"])})

            C = float(cfg["B_total"]) - float(cfg["eps_start"]) - float(cfg["eps_count"])
            transition_spend = 0.0
            for t, tr in enumerate(transitions):
                u, v = str(tr["u"]), str(tr["v"])
                domain = tuple(str(x) for x in successor_cache.get(u, []))
                if not domain or v not in domain:
                    raise RuntimeError(f"True successor outside fixed public N(u): {u}->{v}")
                target = public_degree_bucket(len(domain), int(cfg["K"]), cfg.get("public_degree_thresholds"))
                feasible = feasible_buckets(C, R - t - 1, cfg)
                k = choose_exec_bucket(target, feasible)
                if k != target:
                    fallback += 1
                eps_e = float(cfg["eps_event_list"][k - 1])
                y = rr_sample(v, domain, eps_e, mechanism_rng)
                k_noisy = int(rr_sample(k, bucket_domain, float(cfg["eps_bucket"]), mechanism_rng))
                writer.add({"event_type": "transition", "u": u, "y": y, "k_noisy": k_noisy})
                keep = int(y == v)
                debug.write({
                    "event_type": "transition", "sbs_id": sid, "t": tr.get("t", t + 1),
                    "u": u, "true_v": v, "y": y,
                    "public_target_bucket": target, "exec_bucket_k_t": k,
                    "k_noisy": k_noisy, "epsilon_event": eps_e,
                    "epsilon_bucket": float(cfg["eps_bucket"]),
                    "candidate_size": len(domain), "keep_flag": keep,
                    "uses_private_risk_score": False,
                })
                cost = eps_e + float(cfg["eps_bucket"])
                C -= cost
                transition_spend += cost
                n_trans += 1
                target_counts[target] += 1
                exec_counts[k] += 1
                total_by_target[target] += 1
                keep_by_target[target] += keep

            spend = float(cfg["eps_start"]) + float(cfg["eps_count"]) + transition_spend
            if spend > float(cfg["B_total"]) + 1e-10:
                raise RuntimeError(f"Adapted NULDP SBS schedule exceeds B: {spend}")
            max_spend = max(max_spend, spend)
            min_margin = min(min_margin, float(cfg["B_total"]) - spend)
            n_sbs += 1
            if progress_every > 0 and rec_idx % progress_every == 0:
                print(f"[{METHOD}][{split}] SBSs={rec_idx:,}, transitions={n_trans:,}")
    finally:
        debug.close()

    n_reports = writer.finalize()
    expected = 2 * n_sbs + n_trans
    if n_reports != expected:
        raise RuntimeError(f"Report count mismatch: {n_reports} != {expected}")
    keep_by_bucket = {
        str(k): (keep_by_target[k] / total_by_target[k] if total_by_target[k] else 0.0)
        for k in range(1, int(cfg["K"]) + 1)
    }
    return {
        "split": split,
        "num_sbs": n_sbs,
        "num_transition_events": n_trans,
        "num_server_reports": n_reports,
        "expected_server_reports": expected,
        "target_bucket_counts": {str(k): int(target_counts[k]) for k in range(1, int(cfg["K"]) + 1)},
        "exec_bucket_counts": {str(k): int(exec_counts[k]) for k in range(1, int(cfg["K"]) + 1)},
        "keep_rate_by_target_bucket": keep_by_bucket,
        "keep_rate_overall": (sum(keep_by_target.values()) / n_trans if n_trans else 0.0),
        "scheduler_fallback_rate": (fallback / n_trans if n_trans else 0.0),
        "max_sbs_schedule_spend": max_spend,
        "minimum_schedule_margin": 0.0 if min_margin == float("inf") else min_margin,
        "server_report_path": str(report_path),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Adapted public-context non-uniform LDP comparator")
    p.add_argument("--dataset_config", default=DEFAULT_DATASET_CONFIG)
    p.add_argument("--exp_config", default=DEFAULT_EXP_CONFIG)
    p.add_argument("--progress_every", type=int, default=250)
    p.add_argument("--shuffle_chunk_size", type=int, default=200000)
    p.add_argument("--save_debug", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dpath = resolve_path(args.dataset_config)
    epath = resolve_path(args.exp_config)
    assert dpath is not None and epath is not None
    raw_d, raw_e = load_yaml(dpath), load_yaml(epath)
    dcfg, ecfg = apply_versioning(raw_d, raw_e)
    cfg = extract_cfg(ecfg)

    succ_path = resolve_path(str(dcfg["successor_cache_path"]))
    priv_dir = resolve_path(str(ecfg["privatized_dir"]))
    assert succ_path is not None and priv_dir is not None
    priv_dir.mkdir(parents=True, exist_ok=True)
    successor_cache = load_json(succ_path)
    save_debug = bool(args.save_debug or ecfg.get("save_debug_files", False))

    print("=" * 100)
    print(f"[{METHOD}] adapted non-uniform comparator")
    try:
        print(json.dumps(pretty_version_summary(raw_d, raw_e), ensure_ascii=False, indent=2))
    except Exception:
        pass
    print(json.dumps(cfg, ensure_ascii=False, indent=2))
    print("Allocation uses PUBLIC out-degree only; private TrajRACE risk score is not used.")
    print("=" * 100)

    offsets = {"train": 5101, "valid": 5202, "test": 5303}
    summaries: Dict[str, Any] = {}
    for split in ("train", "valid", "test"):
        event_path = resolve_path(str(dcfg[f"event_{split}"]))
        assert event_path is not None
        if not event_path.exists():
            raise FileNotFoundError(event_path)
        report_path = priv_dir / f"{METHOD}_{split}_reports.jsonl"
        debug_path = priv_dir / f"{METHOD}_{split}_debug.jsonl" if save_debug else None
        seed = int(cfg["random_seed"]) * 10000 + offsets[split]
        t0 = time.perf_counter()
        summaries[split] = privatize_split(
            split=split,
            event_path=event_path,
            report_path=report_path,
            debug_path=debug_path,
            successor_cache=successor_cache,
            cfg=cfg,
            mechanism_rng=random.Random(seed),
            shuffle_rng=random.Random(seed + 5000003),
            shuffle_chunk_size=args.shuffle_chunk_size,
            progress_every=args.progress_every,
        )
        print(
            f"[{METHOD}][{split}] DONE elapsed={time.perf_counter()-t0:.1f}s "
            f"reports={summaries[split]['num_server_reports']:,} "
            f"fallback={summaries[split]['scheduler_fallback_rate']:.6f}"
        )

    meta = {
        "dataset_name": dcfg.get("dataset_name"),
        "dataset_variant": dcfg.get("dataset_variant"),
        "exp_tag": ecfg.get("exp_tag"),
        "method": METHOD,
        "adaptation_scope": (
            "Non-uniform categorical next-hop LDP comparator on the same TrajRACE reporting task; "
            "not an exact reproduction of an external end-to-end trajectory synthesis system."
        ),
        "allocation": {
            "uses_private_risk_score": False,
            "uses_public_current_context": True,
            "public_feature": "out-degree |N(u)|",
            "degree_thresholds": cfg.get("public_degree_thresholds"),
        },
        "server_report_schema": {
            "start": ["event_type", "x_noisy"],
            "count": ["event_type", "x_noisy"],
            "transition": ["event_type", "u", "y", "k_noisy"],
        },
        "privacy_schedule": {
            "B": cfg["B_total"],
            "eps_start": cfg["eps_start"],
            "eps_count": cfg["eps_count"],
            "eps_bucket": cfg["eps_bucket"],
            "eps_event_list": cfg["eps_event_list"],
        },
    }
    save_json(priv_dir / f"{METHOD}_meta.json", meta)
    save_json(priv_dir / f"{METHOD}_summary.json", {
        "dataset_name": dcfg.get("dataset_name"),
        "dataset_variant": dcfg.get("dataset_variant"),
        "exp_tag": ecfg.get("exp_tag"),
        "method": METHOD,
        "splits": summaries,
    })
    print(f"[{METHOD}] DONE")


if __name__ == "__main__":
    main()
