#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""Canonical TrajRACE uniform-allocation comparator.

The uniform mechanism uses the same public domains and the same start/count
budgets as risk-aware TrajRACE. Every transition receives the same event
privacy parameter. A noisy fixed bucket label is still reported so that the
server-visible transition schema and communication cost match the risk-aware
pipeline; importantly, the fixed label is independent of private risk.

Server-visible schemas:
  start      {event_type, x_noisy}
  count      {event_type, x_noisy}
  transition {event_type, u, y, k_noisy}

True risk/target bucket is read only for LOCAL diagnostic stratification and
never affects the uniform mechanism.
"""

import argparse
import heapq
import itertools
import json
import math
import os
import random
import shutil
import sys
import tempfile
import time
from collections import Counter
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from trajrace.versioning_utils import apply_versioning, pretty_version_summary

DEFAULT_DATASET_CONFIG = "configs/dataset.yaml"
DEFAULT_EXP_CONFIG = "configs/exp_main.yaml"


def resolve_path(path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(PROJECT_ROOT, path))


def ensure_parent(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def load_yaml(path: str) -> Dict[str, Any]:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return obj


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, obj: Any) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def iter_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
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


def fmt_sec(sec: float) -> str:
    if sec < 60:
        return f"{sec:.1f}s"
    m = int(sec // 60)
    return f"{m}m{sec - 60*m:.1f}s"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_config", default=DEFAULT_DATASET_CONFIG)
    p.add_argument("--exp_config", default=DEFAULT_EXP_CONFIG)
    p.add_argument("--progress_every", type=int, default=250)
    p.add_argument("--shuffle_chunk_size", type=int, default=200000)
    p.add_argument("--save_debug", action="store_true")
    return p.parse_args()


def rr_sample_fixed_domain(true_value: Any, domain: Sequence[Any], epsilon: float,
                           rng: random.Random, true_is_known_valid: bool = False) -> Any:
    d = len(domain)
    if d <= 0:
        raise ValueError("GRR domain is empty")
    if not true_is_known_valid and true_value not in domain:
        raise ValueError("True value is outside the fixed public GRR domain")
    if d == 1:
        return true_value
    ee = math.exp(float(epsilon))
    p_true = ee / (ee + d - 1)
    if rng.random() < p_true:
        return true_value
    while True:
        candidate = domain[rng.randrange(d)]
        if candidate != true_value:
            return candidate


def nearest_bucket_label(epsilon: float, eps_event_list: Sequence[float]) -> int:
    return min(range(1, len(eps_event_list) + 1),
               key=lambda k: abs(float(epsilon) - float(eps_event_list[k-1])))


def extract_cfg(exp: Dict[str, Any]) -> Dict[str, Any]:
    required = ["B_total", "eps_start", "eps_count", "eps_bucket",
                "eps_event_list", "K", "L_max", "random_seed"]
    for key in required:
        if key not in exp:
            raise KeyError(f"Missing canonical config key: {key}")
    K = int(exp["K"])
    eps_list = [float(x) for x in exp["eps_event_list"]]
    if len(eps_list) != K:
        raise ValueError("eps_event_list length must equal K")
    B = float(exp["B_total"])
    eps_start = float(exp["eps_start"])
    eps_count = float(exp["eps_count"])
    eps_bucket = float(exp["eps_bucket"])
    L_max = int(exp["L_max"])
    tmax = max(0, L_max - 1)
    if "eps_event_uniform" in exp:
        eps_uniform = float(exp["eps_event_uniform"])
    else:
        if tmax <= 0:
            raise ValueError("L_max must be >= 2")
        eps_uniform = (B - eps_start - eps_count) / tmax - eps_bucket
    if eps_uniform <= 0:
        raise ValueError(f"Derived uniform event epsilon is non-positive: {eps_uniform}")
    fixed_label = int(exp.get("uniform_bucket_label",
                              nearest_bucket_label(eps_uniform, eps_list)))
    if not 1 <= fixed_label <= K:
        raise ValueError("uniform_bucket_label outside 1..K")
    max_spend = eps_start + eps_count + tmax * (eps_uniform + eps_bucket)
    if max_spend > B + 1e-10:
        raise ValueError(f"Uniform schedule exceeds B: {max_spend} > {B}")
    return {
        "B_total": B, "eps_start": eps_start, "eps_count": eps_count,
        "eps_bucket": eps_bucket, "eps_event_list": eps_list,
        "eps_event_uniform": eps_uniform, "fixed_bucket_label": fixed_label,
        "K": K, "L_max": L_max, "random_seed": int(exp["random_seed"]),
        "max_full_sbs_schedule_spend": max_spend,
    }


def iter_risk_groups(path: str) -> Iterator[Tuple[str, List[Dict[str, Any]]]]:
    current = None
    group: List[Dict[str, Any]] = []
    for item in iter_jsonl(path):
        sid = str(item["sbs_id"])
        if current is None:
            current = sid
        if sid != current:
            yield current, group
            current, group = sid, []
        group.append(item)
    if current is not None:
        yield current, group


class ExternalShuffleWriter:
    def __init__(self, output_path: str, rng: random.Random, chunk_size: int):
        self.output_path = output_path
        self.rng = rng
        self.chunk_size = max(1000, int(chunk_size))
        ensure_parent(output_path)
        self.temp_dir = tempfile.mkdtemp(prefix=".uniform_shuffle_",
                                         dir=os.path.dirname(output_path))
        self.buffer: List[Tuple[int, int, str]] = []
        self.chunks: List[str] = []
        self.seq = 0
        self.count = 0

    def add(self, report: Dict[str, Any]) -> None:
        key = self.rng.getrandbits(128)
        payload = json.dumps(report, ensure_ascii=False, separators=(",", ":"))
        self.buffer.append((key, self.seq, payload))
        self.seq += 1
        self.count += 1
        if len(self.buffer) >= self.chunk_size:
            self._flush()

    def _flush(self) -> None:
        if not self.buffer:
            return
        self.buffer.sort(key=lambda x: (x[0], x[1]))
        path = os.path.join(self.temp_dir, f"chunk_{len(self.chunks):06d}.txt")
        with open(path, "w", encoding="utf-8") as f:
            for key, seq, payload in self.buffer:
                f.write(f"{key:032x}\t{seq:020d}\t{payload}\n")
        self.chunks.append(path)
        self.buffer = []

    @staticmethod
    def _iter_chunk(path: str):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                a, b, c = line.rstrip("\n").split("\t", 2)
                yield int(a, 16), int(b), c

    def finalize(self) -> int:
        self._flush()
        with open(self.output_path, "w", encoding="utf-8") as out:
            if self.chunks:
                its = [self._iter_chunk(p) for p in self.chunks]
                for _, _, payload in heapq.merge(*its):
                    out.write(payload + "\n")
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        return self.count


class DebugWriter:
    def __init__(self, path: Optional[str]):
        self.f = None
        if path:
            ensure_parent(path)
            self.f = open(path, "w", encoding="utf-8")
    def write(self, obj: Dict[str, Any]) -> None:
        if self.f:
            self.f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    def close(self) -> None:
        if self.f:
            self.f.close()
            self.f = None


def privatize_split(split: str, event_path: str, risk_path: str, report_path: str,
                     debug_path: Optional[str], successor_cache: Dict[str, List[str]],
                     start_domain: Sequence[str], count_domain: Sequence[int],
                     cfg: Dict[str, Any], mechanism_rng: random.Random,
                     shuffle_rng: random.Random, chunk_size: int,
                     progress_every: int) -> Dict[str, Any]:
    writer = ExternalShuffleWriter(report_path, shuffle_rng, chunk_size)
    debug = DebugWriter(debug_path)
    bucket_total, bucket_keep = Counter(), Counter()
    n_sbs = n_trans = 0
    max_spend = 0.0
    min_margin = float("inf")
    try:
        pairs = itertools.zip_longest(iter_jsonl(event_path), iter_risk_groups(risk_path))
        for rec_idx, pair in enumerate(pairs, 1):
            rec, rg = pair
            if rec is None or rg is None:
                raise RuntimeError("Event/risk file length mismatch")
            risk_sid, risks = rg
            sid = str(rec["sbs_id"])
            if risk_sid != sid:
                raise RuntimeError(f"Event/risk SBS mismatch: {sid} vs {risk_sid}")
            transitions = rec["transition_events"]
            if len(transitions) != len(risks):
                raise RuntimeError(f"Transition/risk count mismatch: {sid}")
            n_sbs += 1

            true_start = str(rec["start_event"]["st"])
            true_count = int(rec["count_event"]["cnt"])
            if true_start not in successor_cache:
                raise RuntimeError(f"Start outside fixed public road domain: {true_start}")
            if true_count not in count_domain:
                raise RuntimeError(f"Count outside fixed public count domain: {true_count}")

            y_start = rr_sample_fixed_domain(true_start, start_domain, cfg["eps_start"],
                                             mechanism_rng, true_is_known_valid=True)
            y_count = rr_sample_fixed_domain(true_count, count_domain, cfg["eps_count"], mechanism_rng)
            writer.add({"event_type": "start", "x_noisy": y_start})
            writer.add({"event_type": "count", "x_noisy": int(y_count)})
            debug.write({"event_type": "start", "sbs_id": sid, "true_x": true_start,
                         "noisy_x": y_start, "epsilon_used": cfg["eps_start"]})
            debug.write({"event_type": "count", "sbs_id": sid, "true_x": true_count,
                         "noisy_x": int(y_count), "epsilon_used": cfg["eps_count"]})

            spend = cfg["eps_start"] + cfg["eps_count"]
            for tr, rr in zip(transitions, risks):
                u, v = str(tr["u"]), str(tr["v"])
                if str(rr["u"]) != u or str(rr["v"]) != v:
                    raise RuntimeError(f"Event/risk transition mismatch in {sid}")
                domain = successor_cache.get(u)
                if not domain or v not in domain:
                    raise RuntimeError(f"True successor outside fixed public N(u): {u}->{v}")
                target_bucket = int(rr.get("target_bucket", rr.get("b_t")))
                y = rr_sample_fixed_domain(v, domain, cfg["eps_event_uniform"], mechanism_rng)
                k_noisy = rr_sample_fixed_domain(cfg["fixed_bucket_label"],
                                                  tuple(range(1, cfg["K"] + 1)),
                                                  cfg["eps_bucket"], mechanism_rng)
                writer.add({"event_type": "transition", "u": u, "y": str(y),
                            "k_noisy": int(k_noisy)})
                keep = int(str(y) == v)
                bucket_total[target_bucket] += 1
                bucket_keep[target_bucket] += keep
                n_trans += 1
                spend += cfg["eps_event_uniform"] + cfg["eps_bucket"]
                debug.write({
                    "event_type": "transition", "sbs_id": sid, "t": tr.get("t"),
                    "u": u, "true_v": v, "y": str(y),
                    "target_bucket": target_bucket,
                    "fixed_uniform_bucket_label": cfg["fixed_bucket_label"],
                    "k_noisy": int(k_noisy),
                    "epsilon_event": cfg["eps_event_uniform"],
                    "epsilon_bucket": cfg["eps_bucket"], "keep_flag": keep,
                })

            if spend > cfg["B_total"] + 1e-10:
                raise RuntimeError(f"Uniform SBS schedule exceeds B: {spend}")
            max_spend = max(max_spend, spend)
            min_margin = min(min_margin, cfg["B_total"] - spend)
            if progress_every > 0 and rec_idx % progress_every == 0:
                print(f"[uniform][{split}] SBSs={rec_idx:,}, transitions={n_trans:,}")
    finally:
        debug.close()

    n_reports = writer.finalize()
    expected = 2 * n_sbs + n_trans
    if n_reports != expected:
        raise RuntimeError(f"Report count mismatch: {n_reports} != {expected}")
    keep_by_bucket = {
        str(k): (bucket_keep[k] / bucket_total[k] if bucket_total[k] else 0.0)
        for k in range(1, cfg["K"] + 1)
    }
    overall_keep = sum(bucket_keep.values()) / n_trans if n_trans else 0.0
    return {
        "split": split, "num_sbs": n_sbs, "num_transition_events": n_trans,
        "num_server_reports": n_reports, "expected_server_reports": expected,
        "eps_event_uniform": cfg["eps_event_uniform"],
        "fixed_uniform_bucket_label": cfg["fixed_bucket_label"],
        "target_bucket_counts": {str(k): int(bucket_total[k]) for k in range(1, cfg["K"]+1)},
        "keep_rate_by_target_bucket": keep_by_bucket,
        "keep_rate_overall": overall_keep,
        "max_sbs_schedule_spend": max_spend,
        "minimum_schedule_margin": 0.0 if min_margin == float("inf") else min_margin,
        "server_report_path": report_path,
    }


def main() -> None:
    args = parse_args()
    dcfg_path, ecfg_path = resolve_path(args.dataset_config), resolve_path(args.exp_config)
    assert dcfg_path and ecfg_path
    raw_d, raw_e = load_yaml(dcfg_path), load_yaml(ecfg_path)
    dcfg, ecfg = apply_versioning(raw_d, raw_e)
    cfg = extract_cfg(ecfg)
    print("=" * 90)
    print("[privatize_uniform] Canonical uniform comparator")
    print(json.dumps(pretty_version_summary(raw_d, raw_e), indent=2, ensure_ascii=False))
    print(json.dumps(cfg, indent=2, ensure_ascii=False))
    print("=" * 90)

    successor_path = resolve_path(dcfg["successor_cache_path"])
    privatized_dir = resolve_path(ecfg["privatized_dir"])
    experiment_root = resolve_path(ecfg["experiment_root"])
    assert successor_path and privatized_dir and experiment_root
    os.makedirs(privatized_dir, exist_ok=True)
    successor_cache = load_json(successor_path)
    start_domain = tuple(successor_cache.keys())
    count_domain = tuple(range(1, cfg["L_max"] + 1))
    risk_dir = os.path.join(experiment_root, "risk")
    save_debug = bool(args.save_debug or ecfg.get("save_debug_files", False))
    offsets = {"train": 1101, "valid": 1202, "test": 1303}
    summaries = {}

    for split in ("train", "valid", "test"):
        event_path = resolve_path(dcfg[f"event_{split}"])
        risk_path = os.path.join(risk_dir, f"{split}_risk.jsonl")
        assert event_path
        if not os.path.exists(event_path) or not os.path.exists(risk_path):
            raise FileNotFoundError(f"Missing event/risk input for {split}")
        report_path = os.path.join(privatized_dir, f"uniform_{split}_reports.jsonl")
        debug_path = os.path.join(privatized_dir, f"uniform_{split}_debug.jsonl") if save_debug else None
        seed = cfg["random_seed"] * 10000 + offsets[split]
        t = time.perf_counter()
        summaries[split] = privatize_split(
            split, event_path, risk_path, report_path, debug_path,
            successor_cache, start_domain, count_domain, cfg,
            random.Random(seed), random.Random(seed + 5000003),
            args.shuffle_chunk_size, args.progress_every,
        )
        print(f"[uniform][{split}] DONE | elapsed={fmt_sec(time.perf_counter()-t)} | "
              f"reports={summaries[split]['num_server_reports']:,} | "
              f"max-spend={summaries[split]['max_sbs_schedule_spend']:.6f}")

    meta = {
        "dataset_name": dcfg["dataset_name"], "dataset_variant": dcfg["dataset_variant"],
        "exp_tag": ecfg["exp_tag"], "method": "uniform",
        "uniform_cfg": cfg,
        "public_domains": {
            "start": "all fixed public road segments",
            "count": f"1..{cfg['L_max']}", "successor": "fixed public N(u)",
            "secret_dependent_domain_expansion": False,
        },
        "server_report_schema": {
            "start": ["event_type", "x_noisy"],
            "count": ["event_type", "x_noisy"],
            "transition": ["event_type", "u", "y", "k_noisy"],
        },
        "notes": {
            "risk_does_not_affect_uniform_mechanism": True,
            "target_bucket_used_only_for_local_diagnostics": True,
            "fixed_bucket_label_independent_of_private_data": True,
        },
    }
    summary = {
        "dataset_name": dcfg["dataset_name"], "dataset_variant": dcfg["dataset_variant"],
        "exp_tag": ecfg["exp_tag"], "method": "uniform", "splits": summaries,
    }
    save_json(os.path.join(privatized_dir, "uniform_meta.json"), meta)
    save_json(os.path.join(privatized_dir, "uniform_summary.json"), summary)
    print("[privatize_uniform] DONE")


if __name__ == "__main__":
    main()
