#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
scripts/run_w3d3_experiments.py

Unified Reviewer-1 W3/D3 experiment orchestrator.

This script does NOT alter the TrajRACE mechanism.  It drives only the
reviewer-targeted evidence requested in W3/D3:

    multiseed    Five-seed robustness under the exact Table-II evaluator.
    attack       Mechanism-independent auxiliary-data next-hop attack.
    baseline     Adapted-NU-LDP matched non-uniform comparator.
    ablation     Risk-factor ablations beyond Chengdu@B=1.
    scalability  Fresh 100K-to-full-data reporting/recovery scaling.
    all          Run the requested families sequentially.

Minimal new-script design
-------------------------
The final W3/D3 artifact only needs these four new scripts:
    scripts/run_w3d3_experiments.py
    scripts/evaluate_independent_attack.py
    scripts/privatize_adapted_nuldp.py
    scripts/recover_adapted_nuldp.py

The canonical project scripts remain the source of truth for preprocessing,
risk scoring, client reporting, recovery, synthesis, and Table-II evaluation.

Important hard gates
--------------------
1) Multi-seed:
   The compact scripts/evaluate_compare.py metrics are NOT Table-II Query-Err,
   Dens-Err, Trip-Err, etc.  Formal multiseed mode therefore requires one or
   more exact Table-II metrics CSV files through --table2_metric_csv unless
   --allow_missing_table2 is explicitly used for development.

2) Independent attack:
   The attack model is trained/calibrated on a disjoint auxiliary split.  The
   test release is used only for inference/scoring.

3) Ablation:
   If historical versioning does not include risk weights in exp_tag, this
   runner still remains safe: every variant is forced to recompute and its
   output is copied immediately before the next variant can overwrite the
   shared experiment root.

4) Scalability:
   Scale runs strip fixed_dataset_variant/fixed_dataset_root, force fresh
   selected stages, reject SKIPPED_EXISTING stages, and verify the actual raw
   sample count from the preprocessing summary.  Synthesis is intentionally
   outside the main scaling core because N* is normally fixed near test size;
   report it separately if desired.
"""

import argparse
import csv
import datetime as dt
import importlib.util
import json
import math
import os
import random
import re
import shutil
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from trajrace.versioning_utils import apply_versioning

DEFAULT_DATASET_CONFIG = "configs/dataset.yaml"
DEFAULT_EXP_CONFIG = "configs/exp_main.yaml"


# =============================================================================
# Basic helpers
# =============================================================================

def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def resolve_path(pathlike: str | Path | None) -> Optional[Path]:
    if pathlike is None:
        return None
    p = Path(pathlike)
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


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


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def save_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)

    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def numeric(x: Any) -> Optional[float]:
    try:
        if x is None or str(x).strip() == "":
            return None
        y = float(x)
        return y if math.isfinite(y) else None
    except Exception:
        return None


def newest_file(root: Path, name: str) -> Optional[Path]:
    xs = list(root.rglob(name)) if root.exists() else []
    return max(xs, key=lambda p: p.stat().st_mtime) if xs else None


def run_process(
    cmd: Sequence[str],
    log_path: Path,
    cwd: Path = PROJECT_ROOT,
    monitor_memory: bool = False,
) -> Dict[str, Any]:
    cmd = [str(x) for x in cmd]
    log_path.parent.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 110)
    print("[RUN] " + " ".join(cmd))
    print("[LOG] " + str(log_path))
    print("=" * 110)

    psutil = None
    peak_rss = None
    if monitor_memory:
        try:
            import psutil as _psutil
            psutil = _psutil
        except Exception:
            print("[WARN] psutil unavailable; peak RSS will be null.")

    started = utc_now()
    t0 = time.perf_counter()

    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        pinfo = psutil.Process(proc.pid) if psutil is not None else None
        assert proc.stdout is not None

        while True:
            line = proc.stdout.readline()
            if line:
                print(line, end="")
                log.write(line)

            if pinfo is not None:
                try:
                    procs = [pinfo] + pinfo.children(recursive=True)
                    rss = sum(
                        p.memory_info().rss
                        for p in procs
                        if p.is_running()
                    )
                    peak_rss = max(int(peak_rss or 0), int(rss))
                except Exception:
                    pass

            if proc.poll() is not None:
                for rest in proc.stdout:
                    print(rest, end="")
                    log.write(rest)
                break

            if not line:
                time.sleep(0.1)

        rc = int(proc.wait())

    elapsed = time.perf_counter() - t0
    result = {
        "command": cmd,
        "started_at_utc": started,
        "finished_at_utc": utc_now(),
        "elapsed_sec": elapsed,
        "return_code": rc,
        "peak_rss_mb_process_tree": (
            peak_rss / 1024.0 / 1024.0
            if peak_rss is not None
            else None
        ),
        "log_path": str(log_path),
    }
    if rc != 0:
        raise RuntimeError(f"Command failed ({rc}): {' '.join(cmd)}")
    return result


# =============================================================================
# Config helpers
# =============================================================================

def profiled_exp_config(
    raw_exp: Mapping[str, Any],
    profile: str,
    seed: int,
    *,
    methods: Optional[Sequence[str]] = None,
    strip_fixed_dataset: bool = False,
) -> Dict[str, Any]:
    cfg = dict(raw_exp)

    profiles = cfg.get("privacy_profiles", {})
    if profiles and profile not in profiles:
        raise KeyError(
            f"Privacy profile {profile!r} not found. "
            f"Available: {sorted(profiles)}"
        )

    cfg["active_budget_profile"] = str(profile)
    cfg["default_budget_profile"] = str(profile)
    cfg["random_seed"] = int(seed)

    if methods is not None:
        cfg["methods"] = list(methods)

    if strip_fixed_dataset:
        # Current versioning_utils.py activates fixed reuse only through these
        # two keys.  Remove them for real size scaling.
        cfg.pop("fixed_dataset_variant", None)
        cfg.pop("fixed_dataset_root", None)

    return cfg


def size_specific_dataset(raw_dataset: Mapping[str, Any], size: int) -> Dict[str, Any]:
    if int(size) <= 0:
        raise ValueError("scalability size must be > 0")

    cfg = dict(raw_dataset)
    dataset = str(cfg.get("dataset_name", "dataset")).lower()

    # Set generic + dataset-specific aliases because historical versioning
    # code has used both conventions.
    cfg["raw_sample_size"] = int(size)
    cfg[f"{dataset}_raw_sample_size"] = int(size)
    if dataset == "porto":
        cfg["porto_raw_sample_size"] = int(size)
    elif dataset == "chengdu":
        cfg["chengdu_raw_sample_size"] = int(size)
    elif dataset == "oldenburg":
        cfg["oldenburg_raw_sample_size"] = int(size)

    return cfg


def runner_cmd(
    args: argparse.Namespace,
    dataset_cfg: Path,
    exp_cfg: Path,
    *,
    mode: str,
    run_root: Path,
    stages: Optional[Sequence[str]] = None,
    save_debug: bool = False,
    profiles: Optional[Sequence[str]] = None,
    seeds: Optional[Sequence[int]] = None,
    raw_size: Optional[int] = None,
    joint_mode: str = "none",
    force_rerun: bool = False,
    force_restart_map_matching: bool = False,
) -> List[str]:

    cmd = [
        args.python,
        "scripts/run_rebuttal_experiments.py",
        "--dataset_config",
        str(dataset_cfg),
        "--exp_config",
        str(exp_cfg),
        "--mode",
        mode,
        "--run_root",
        str(run_root),
        "--joint_mode",
        joint_mode,
    ]

    if stages:
        cmd += ["--stages"] + list(stages)
    if save_debug:
        cmd.append("--save_debug")
    if profiles:
        cmd += ["--profiles"] + [str(x) for x in profiles]
    if seeds:
        cmd += ["--seeds"] + [str(x) for x in seeds]
    if raw_size is not None:
        cmd += ["--raw_sample_size_override", str(int(raw_size))]
    if force_rerun or args.rerun:
        cmd.append("--rerun")
    if force_restart_map_matching:
        cmd.append("--force_restart_map_matching")

    return cmd


# =============================================================================
# Metric normalization and aggregation
# =============================================================================

TABLE2_COLUMN_MAP = {
    "Density Error": "dens_err",
    "density error": "dens_err",
    "Dens-Err": "dens_err",
    "dens_err": "dens_err",

    "Query Error": "query_err",
    "query error": "query_err",
    "Query-Err": "query_err",
    "query_err": "query_err",

    "HQ Error": "hq_err",
    "HQ-Err": "hq_err",
    "hq_err": "hq_err",

    "Trip Error": "trip_err",
    "trip error": "trip_err",
    "Trip-Err": "trip_err",
    "trip_err": "trip_err",

    "Length Error": "len_err",
    "length error": "len_err",
    "Len-Err": "len_err",
    "len_err": "len_err",

    "Diameter Error": "diam_err",
    "diameter error": "diam_err",
    "Diam-Err": "diam_err",
    "diam_err": "diam_err",

    "Pattern F1": "pattern_f1",
    "Pattern-F1": "pattern_f1",
    "pattern_f1": "pattern_f1",
}

DEFAULT_REQUIRED_TABLE2 = (
    "dens_err",
    "query_err",
    "trip_err",
    "len_err",
    "diam_err",
    "pattern_f1",
)


def infer_metadata_from_path(path: Path) -> Dict[str, Any]:
    text = str(path).lower()

    seed = None
    m = re.search(r"seed[_-]?(\d+)", text)
    if m:
        seed = int(m.group(1))

    profile = None
    m = re.search(r"\bb(\d+)p(\d+)\b", text)
    if m:
        profile = f"B{m.group(1)}p{m.group(2)}"

    dataset = None
    for name in ("porto", "oldenburg", "chengdu"):
        if name in text:
            dataset = name
            break

    return {
        "seed": seed,
        "profile": profile,
        "dataset_name": dataset,
    }


def normalize_table2_row(
    row: Mapping[str, Any],
    path_meta: Mapping[str, Any],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    # Copy metadata if already present.
    for key in ("dataset_name", "dataset", "dataset_variant", "profile", "B", "seed", "method", "exp_tag"):
        if key in row and row[key] not in (None, ""):
            out["dataset_name" if key == "dataset" else key] = row[key]

    # Fill missing metadata from file path.
    for key in ("dataset_name", "profile", "seed"):
        if out.get(key) in (None, "") and path_meta.get(key) is not None:
            out[key] = path_meta[key]

    # Canonicalize method naming.
    method = str(out.get("method", "")).strip()
    ml = method.lower().replace("-", "_").replace(" ", "_")
    if ml in {"trajrace", "trajrace_riskaware", "riskaware"}:
        out["method"] = "trajrace_riskaware"
    elif ml in {"trajrace_uniform", "uniform"}:
        out["method"] = "trajrace_uniform"
    elif method:
        out["method"] = method

    # Normalize Table-II metric columns.
    for key, value in row.items():
        canonical = TABLE2_COLUMN_MAP.get(str(key).strip())
        if canonical is not None:
            out[canonical] = value

    return out


def expand_csv_patterns(patterns: Sequence[str]) -> List[Path]:
    found: List[Path] = []
    for pat in patterns:
        p = Path(pat)
        if p.is_absolute():
            base = p.parent
            name = p.name
        else:
            rp = resolve_path(pat)
            assert rp is not None
            base = rp.parent
            name = rp.name

        if any(ch in name for ch in "*?["):
            found.extend(sorted(base.glob(name)))
        elif (base / name).exists():
            found.append(base / name)

    # Stable de-duplication.
    unique: List[Path] = []
    seen = set()
    for p in found:
        q = p.resolve()
        if q not in seen:
            seen.add(q)
            unique.append(q)
    return unique


def load_table2_rows(csv_paths: Sequence[Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in csv_paths:
        meta = infer_metadata_from_path(path)
        raw_rows = read_csv(path)
        for raw in raw_rows:
            row = normalize_table2_row(raw, meta)
            if row:
                row["source_csv"] = str(path)
                rows.append(row)
    return rows


def aggregate_mean_std(
    rows: Sequence[Mapping[str, Any]],
    out_csv: Path,
    out_json: Path,
    *,
    required_metrics: Sequence[str] = (),
    min_seeds: int = 1,
    group_keys: Sequence[str] = ("dataset_name", "profile", "method"),
) -> List[Dict[str, Any]]:

    rows = list(rows)
    if not rows:
        raise RuntimeError("No rows supplied to aggregate_mean_std().")

    if required_metrics:
        observed = {
            metric
            for row in rows
            for metric in required_metrics
            if numeric(row.get(metric)) is not None
        }
        missing = [m for m in required_metrics if m not in observed]
        if missing:
            raise RuntimeError(
                "Required Table-II metrics are missing from supplied CSVs: "
                + ", ".join(missing)
            )

    groups: Dict[Tuple[str, ...], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        key = tuple(str(row.get(k, "")) for k in group_keys)
        groups[key].append(row)

    metric_fields: List[str] = []
    candidate_keys = sorted({k for row in rows for k in row})
    ignore = set(group_keys) | {
        "seed",
        "dataset_variant",
        "exp_tag",
        "source_csv",
        "B",
    }
    for key in candidate_keys:
        if key in ignore:
            continue
        if any(numeric(row.get(key)) is not None for row in rows):
            metric_fields.append(key)

    out_rows: List[Dict[str, Any]] = []
    for key, rr in sorted(groups.items()):
        seeds = sorted({
            int(float(row["seed"]))
            for row in rr
            if row.get("seed") not in (None, "")
            and numeric(row.get("seed")) is not None
        })

        if len(seeds) < int(min_seeds):
            raise RuntimeError(
                f"Group {key} has only {len(seeds)} distinct seeds; "
                f"required >= {min_seeds}. Sources: "
                f"{sorted({str(x.get('source_csv','')) for x in rr})}"
            )

        out: Dict[str, Any] = {
            k: key[i] for i, k in enumerate(group_keys)
        }
        out["n_seeds"] = len(seeds)
        out["seeds"] = ",".join(str(x) for x in seeds)

        for metric in metric_fields:
            vals = [
                numeric(row.get(metric))
                for row in rr
                if numeric(row.get(metric)) is not None
            ]
            if not vals:
                continue
            out[f"{metric}_mean"] = statistics.fmean(vals)
            out[f"{metric}_std"] = (
                statistics.stdev(vals) if len(vals) >= 2 else 0.0
            )
            out[f"{metric}_min"] = min(vals)
            out[f"{metric}_max"] = max(vals)

        out_rows.append(out)

    save_csv(out_csv, out_rows)
    save_json(
        out_json,
        {
            "created_at_utc": utc_now(),
            "required_metrics": list(required_metrics),
            "min_seeds": int(min_seeds),
            "rows": out_rows,
        },
    )
    return out_rows


# =============================================================================
# MULTISEED
# =============================================================================

def run_multiseed(args: argparse.Namespace, root: Path) -> Dict[str, Any]:
    out = root / "multiseed"
    runner_root = out / "canonical_runner"

    dataset_path = resolve_path(args.dataset_config)
    exp_path = resolve_path(args.exp_config)
    assert dataset_path and exp_path

    canonical = run_process(
        runner_cmd(
            args,
            dataset_path,
            exp_path,
            mode="grid",
            run_root=runner_root,
            profiles=args.profiles,
            seeds=args.seeds,
            joint_mode="none",
            force_rerun=args.rerun,
        ),
        out / "canonical_grid.log",
    )

    compact_csv = newest_file(runner_root, "rebuttal_aggregate.csv")
    if compact_csv is not None:
        shutil.copy2(compact_csv, out / "compact_rebuttal_aggregate.csv")

    table2_paths = expand_csv_patterns(args.table2_metric_csv)

    if not table2_paths:
        if args.allow_missing_table2:
            save_json(
                out / "TABLE2_MISSING.json",
                {
                    "warning": (
                        "Formal rebuttal multiseed evidence is incomplete. "
                        "Pass exact per-seed Table-II metrics CSVs through "
                        "--table2_metric_csv."
                    )
                },
            )
            return {
                "canonical_runner": canonical,
                "compact_csv": str(compact_csv) if compact_csv else None,
                "table2_summary_csv": None,
                "formal_table2_ready": False,
            }
        raise RuntimeError(
            "Formal multiseed mode requires exact Table-II metrics CSV(s).\n"
            "Example:\n"
            "  --table2_metric_csv 'rebuttal_table2/porto/B0p5/seed*/metrics.csv'\n"
            "The compact evaluate_compare.py output is not Query-Err/Dens-Err."
        )

    table2_rows = load_table2_rows(table2_paths)
    required = tuple(args.require_table2_metrics)

    summary = aggregate_mean_std(
        table2_rows,
        out / "table2_mean_std.csv",
        out / "table2_mean_std.json",
        required_metrics=required,
        min_seeds=args.min_multiseed_runs,
    )

    save_csv(out / "table2_input_rows.normalized.csv", table2_rows)

    return {
        "canonical_runner": canonical,
        "compact_csv": str(compact_csv) if compact_csv else None,
        "table2_sources": [str(x) for x in table2_paths],
        "table2_summary_csv": str(out / "table2_mean_std.csv"),
        "n_table2_summary_rows": len(summary),
        "formal_table2_ready": True,
    }


# =============================================================================
# ATTACK
# =============================================================================

def run_attack(args: argparse.Namespace, root: Path) -> Dict[str, Any]:
    out = root / "attack"
    dataset_path = resolve_path(args.dataset_config)
    exp_path = resolve_path(args.exp_config)
    assert dataset_path and exp_path

    raw_exp = load_yaml(exp_path)
    attack_exp = profiled_exp_config(
        raw_exp,
        profile=args.attack_profile,
        seed=args.attack_seed,
        methods=["riskaware", "uniform"],
    )
    attack_exp_path = out / "configs" / "exp_attack.generated.yaml"
    save_yaml(attack_exp_path, attack_exp)

    # Debug traces for BOTH auxiliary and test splits are produced locally.
    # They must never be published in the final artifact.
    prep = run_process(
        runner_cmd(
            args,
            dataset_path,
            attack_exp_path,
            mode="single",
            run_root=out / "canonical_runner",
            stages=["preprocess", "events", "risk", "riskaware", "uniform"],
            save_debug=True,
            joint_mode="none",
            force_rerun=args.rerun,
        ),
        out / "attack_prepare.log",
    )

    attack_cmd = [
        args.python,
        str(SCRIPTS_DIR / "evaluate_independent_attack.py"),
        "--dataset_config",
        str(dataset_path),
        "--exp_config",
        str(attack_exp_path),
        "--aux_split",
        args.attack_aux_split,
        "--test_split",
        args.attack_test_split,
        "--methods",
        "riskaware",
        "uniform",
        "--shrinkage_mass",
        str(args.attack_shrinkage_mass),
        "--min_support",
        str(args.attack_min_support),
        "--output_dir",
        str(out / "results"),
    ]
    attack = run_process(
        attack_cmd,
        out / "attack_eval.log",
    )

    result_csv = out / "results" / "attack_results.csv"
    if not result_csv.exists():
        raise FileNotFoundError(result_csv)

    return {
        "prepare": prep,
        "attack": attack,
        "profile": args.attack_profile,
        "seed": args.attack_seed,
        "result_csv": str(result_csv),
    }


# =============================================================================
# ADAPTED NON-UNIFORM BASELINE
# =============================================================================

def dynamic_import(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def synthesize_and_evaluate_adapted(
    dataset_cfg: Path,
    exp_cfg: Path,
    out_dir: Path,
) -> Dict[str, Any]:
    """Synthesize/evaluate adapted_nuldp using canonical helper functions.

    This deliberately computes the compact task-matched diagnostics (Trans-JS,
    synthetic Bigram-JS, Count-JS, legality, exact count) without pretending
    they are the full Table-II metric suite.
    """
    synmod = dynamic_import(
        SCRIPTS_DIR / "synthesize_trajectories.py",
        "trajrace_syn_w3d3",
    )
    evmod = dynamic_import(
        SCRIPTS_DIR / "evaluate_compare.py",
        "trajrace_eval_w3d3",
    )

    raw_d = load_yaml(dataset_cfg)
    raw_e = load_yaml(exp_cfg)
    dcfg, ecfg = apply_versioning(raw_d, raw_e)

    recdir = resolve_path(ecfg["recovered_dir"])
    syndir = resolve_path(ecfg["synthetic_dir"])
    succ_path = resolve_path(dcfg["successor_cache_path"])
    test_path = resolve_path(dcfg["event_test"])
    assert recdir and syndir and succ_path and test_path

    syndir.mkdir(parents=True, exist_ok=True)
    succ = load_json(succ_path)

    method = "adapted_nuldp"
    splits = [str(x) for x in ecfg.get("synthesis_splits", ["train", "valid"])]
    for split in splits:
        if split not in {"train", "valid"}:
            raise ValueError(
                "Adapted baseline synthesis must not use test recovery."
            )

    starts = []
    counts = []
    transparts = []

    for split in splits:
        starts.append(
            synmod.load_scalar(
                str(recdir / f"{method}_{split}_start.json")
            )
        )
        counts.append(
            synmod.load_scalar(
                str(recdir / f"{method}_{split}_count.json")
            )
        )
        transparts.append(
            synmod.load_transition(
                str(recdir / f"{method}_{split}_transition_context.json")
            )
        )

    sd = synmod.merge_scalar(starts)
    cd = synmod.merge_scalar(counts)
    tr = synmod.merge_transition(transparts)
    gp = synmod.global_successor_prior(tr)
    sd = synmod.normalize({s: p for s, p in sd.items() if s in succ})
    if not sd:
        raise RuntimeError("Adapted-NU-LDP recovered start distribution is empty.")

    mode = str(ecfg.get("synthetic_num_mode", "match_test_size"))
    if mode == "match_test_size":
        n_syn = synmod.count_jsonl(str(test_path))
    elif mode == "fixed":
        n_syn = int(ecfg.get("synthetic_num", 1000))
    else:
        n_syn = int(mode)

    rng = random.Random(int(ecfg.get("random_seed", 42)) + 97001)
    records = []
    diags = []
    t0 = time.perf_counter()

    for i in range(n_syn):
        rec, diag = synmod.generate_one(
            f"{method}_syn_{i:06d}",
            sd,
            cd,
            tr,
            succ,
            gp,
            int(ecfg["L_max"]),
            rng,
        )
        records.append(rec)
        diags.append(diag)

    syn_path = syndir / f"{method}_test_synthetic.jsonl"
    synmod.write_jsonl(str(syn_path), records)

    exact = (
        sum(int(x["exact"]) for x in diags) / len(diags)
        if diags else 0.0
    )

    test = list(evmod.iter_jsonl(str(test_path)))
    true_dist, true_n = evmod.true_transition(test)
    true_bigram = evmod.bigram_prob(test)
    true_count = evmod.count_prob(test)

    rec_path = recdir / f"{method}_test_transition_context.json"
    recovered = evmod.recovered_transition(str(rec_path))
    synthetic = list(evmod.iter_jsonl(str(syn_path)))

    row: Dict[str, Any] = {
        "method": method,
        "dataset_name": dcfg.get("dataset_name"),
        "dataset_variant": dcfg.get("dataset_variant"),
        "profile": ecfg.get("active_budget_profile", ""),
        "seed": ecfg.get("random_seed"),
        "transition_js": evmod.weighted_transition_js(
            true_dist, true_n, recovered
        ),
        "synthetic_bigram_js": evmod.js(
            true_bigram, evmod.bigram_prob(synthetic)
        ),
        "synthetic_count_js": evmod.js(
            true_count, evmod.count_prob(synthetic)
        ),
        "public_legal_ratio": evmod.public_legal_ratio(
            synthetic, succ
        ),
        "exact_count_rate": exact,
        "num_synthetic_trajectories": len(synthetic),
        "synthesis_elapsed_sec": time.perf_counter() - t0,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    save_csv(out_dir / "adapted_nuldp_compare.csv", [row])
    save_json(out_dir / "adapted_nuldp_compare.json", row)
    return row


def run_baseline(args: argparse.Namespace, root: Path) -> Dict[str, Any]:
    out = root / "baseline"
    dataset_path = resolve_path(args.dataset_config)
    exp_path = resolve_path(args.exp_config)
    assert dataset_path and exp_path

    raw_exp = load_yaml(exp_path)
    baseline_exp = profiled_exp_config(
        raw_exp,
        profile=args.baseline_profile,
        seed=args.baseline_seed,
        methods=["riskaware", "uniform"],
    )
    baseline_exp_path = out / "configs" / "exp_baseline.generated.yaml"
    save_yaml(baseline_exp_path, baseline_exp)

    # Prepare matched canonical outputs.
    canonical = run_process(
        runner_cmd(
            args,
            dataset_path,
            baseline_exp_path,
            mode="single",
            run_root=out / "canonical_runner",
            joint_mode="none",
            force_rerun=args.rerun,
        ),
        out / "canonical_prepare.log",
    )

    # Adapted-NU-LDP: same SBS/public domains/privacy levels/B cap/schema and
    # backend, but protection level is assigned from PUBLIC out-degree only.
    priv = run_process(
        [
            args.python,
            str(SCRIPTS_DIR / "privatize_adapted_nuldp.py"),
            "--dataset_config",
            str(dataset_path),
            "--exp_config",
            str(baseline_exp_path),
        ],
        out / "adapted_privatize.log",
    )

    rec = run_process(
        [
            args.python,
            str(SCRIPTS_DIR / "recover_adapted_nuldp.py"),
            "--dataset_config",
            str(dataset_path),
            "--exp_config",
            str(baseline_exp_path),
        ],
        out / "adapted_recover.log",
    )

    adapted_row = synthesize_and_evaluate_adapted(
        dataset_path,
        baseline_exp_path,
        out / "results",
    )

    # Copy the strict riskaware/uniform compact comparison next to the adapted
    # result so reviewers can inspect all three controlled variants together.
    canonical_csv = newest_file(
        out / "canonical_runner",
        "main_compare.csv",
    )
    canonical_rows: List[Dict[str, Any]] = []
    if canonical_csv is not None:
        canonical_rows = read_csv(canonical_csv)
        shutil.copy2(
            canonical_csv,
            out / "results" / "canonical_compare.csv",
        )

    combined: List[Dict[str, Any]] = []
    for row in canonical_rows:
        r = dict(row)
        r.setdefault("dataset_name", load_yaml(dataset_path).get("dataset_name"))
        r.setdefault("profile", args.baseline_profile)
        r.setdefault("seed", args.baseline_seed)
        combined.append(r)
    combined.append(adapted_row)
    save_csv(out / "results" / "baseline_threeway.csv", combined)

    return {
        "profile": args.baseline_profile,
        "seed": args.baseline_seed,
        "canonical": canonical,
        "adapted_privatize": priv,
        "adapted_recover": rec,
        "threeway_csv": str(out / "results" / "baseline_threeway.csv"),
        "adapted_result": adapted_row,
    }


# =============================================================================
# ABLATION
# =============================================================================

def ablation_variants(
    base_exp: Mapping[str, Any],
    profile: str,
    seed: int,
) -> Dict[str, Dict[str, Any]]:

    def set_weights(
        cfg: Mapping[str, Any],
        endpoint: float,
        stay: float,
        degree: float,
    ) -> Dict[str, Any]:
        out = profiled_exp_config(
            cfg,
            profile=profile,
            seed=seed,
            methods=["riskaware", "uniform"],
        )
        out["lambda_e"] = float(endpoint)
        out["lambda_s"] = float(stay)
        out["lambda_st"] = float(stay)   # historical alias
        out["lambda_d"] = float(degree)
        return out

    return {
        "Full": set_weights(base_exp, 1 / 3, 1 / 3, 1 / 3),
        "w/o-End": set_weights(base_exp, 0.0, 0.5, 0.5),
        "w/o-Stay": set_weights(base_exp, 0.5, 0.0, 0.5),
        "w/o-Deg": set_weights(base_exp, 0.5, 0.5, 0.0),
    }


def run_ablation(args: argparse.Namespace, root: Path) -> Dict[str, Any]:
    out = root / "ablation"
    dataset_path = resolve_path(args.dataset_config)
    exp_path = resolve_path(args.exp_config)
    assert dataset_path and exp_path

    raw_exp = load_yaml(exp_path)
    variants = ablation_variants(
        raw_exp,
        profile=args.ablation_profile,
        seed=args.ablation_seed,
    )

    result_rows: List[Dict[str, Any]] = []
    uniform_row: Optional[Dict[str, Any]] = None
    roots_seen: Dict[str, str] = {}

    for name, cfg in variants.items():
        cfg_path = out / "configs" / f"exp_{name.replace('/', '_')}.yaml"
        save_yaml(cfg_path, cfg)

        # Historical versioning may omit lambda_* from exp_sig.  This is safe
        # here because we force a complete recomputation and copy the result
        # immediately before running the next variant.
        dcfg, ecfg = apply_versioning(
            load_yaml(dataset_path),
            load_yaml(cfg_path),
        )
        exp_root = str(ecfg.get("experiment_root", ecfg.get("exp_tag", "")))
        roots_seen[name] = exp_root

        run = run_process(
            runner_cmd(
                args,
                dataset_path,
                cfg_path,
                mode="single",
                run_root=out / "canonical_runner" / name.replace("/", "_"),
                joint_mode="none",
                force_rerun=True,   # critical when historical exp_tag collides
            ),
            out / "logs" / f"{name.replace('/', '_')}.log",
        )

        aggregate = newest_file(
            out / "canonical_runner" / name.replace("/", "_"),
            "rebuttal_aggregate.csv",
        )
        if aggregate is None:
            raise FileNotFoundError(
                f"rebuttal_aggregate.csv missing for ablation {name}"
            )

        copied = out / "results" / f"{name.replace('/', '_')}_aggregate.csv"
        copied.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(aggregate, copied)

        rows = read_csv(aggregate)
        risk_rows = [
            r for r in rows
            if str(r.get("method", "")).lower() == "riskaware"
        ]
        unif_rows = [
            r for r in rows
            if str(r.get("method", "")).lower() == "uniform"
        ]

        if len(risk_rows) != 1:
            raise RuntimeError(
                f"Expected exactly one riskaware row for {name}; got {len(risk_rows)}"
            )

        rr = dict(risk_rows[0])
        rr["ablation_variant"] = name
        result_rows.append(rr)

        if name == "Full":
            if len(unif_rows) != 1:
                raise RuntimeError(
                    "Expected exactly one Uniform row in Full ablation run."
                )
            uniform_row = dict(unif_rows[0])
            uniform_row["ablation_variant"] = "Uniform"

    if uniform_row is None:
        raise RuntimeError("Uniform ablation control was not captured.")

    result_rows.append(uniform_row)

    # Fixed ordering requested in the rebuttal.
    order = {
        "Full": 0,
        "w/o-End": 1,
        "w/o-Stay": 2,
        "w/o-Deg": 3,
        "Uniform": 4,
    }
    result_rows.sort(key=lambda r: order[str(r["ablation_variant"])])

    save_csv(out / "ablation_results.csv", result_rows)
    save_json(
        out / "ablation_manifest.json",
        {
            "profile": args.ablation_profile,
            "seed": args.ablation_seed,
            "historical_versioning_roots": roots_seen,
            "collision_detected": len(set(roots_seen.values())) != len(roots_seen),
            "collision_safety": (
                "Every variant forced --rerun and was copied immediately."
            ),
            "rows": result_rows,
        },
    )

    return {
        "profile": args.ablation_profile,
        "seed": args.ablation_seed,
        "result_csv": str(out / "ablation_results.csv"),
        "collision_detected": len(set(roots_seen.values())) != len(roots_seen),
    }


# =============================================================================
# SCALABILITY
# =============================================================================

def parse_preprocess_summary(path: Optional[Path]) -> Dict[str, Any]:
    if path is None or not path.exists():
        return {}

    obj = load_json(path)
    out: Dict[str, Any] = {
        "preprocess_summary_path": str(path),
    }

    raw_sampling = obj.get("raw_sampling")
    if isinstance(raw_sampling, dict):
        for key in (
            "total_csv_rows",
            "valid_rows_before_sampling",
            "requested_sample_size",
            "actual_sample_size",
        ):
            if key in raw_sampling:
                out[key] = raw_sampling[key]

    # Historical schemas.
    for key in (
        "num_raw_after_filter",
        "num_valid_mapped_trajectories",
        "num_valid_sequence_records",
        "num_failed_map_matching_or_too_short",
        "num_failed_mapping_or_too_short",
    ):
        if key in obj:
            out[key] = obj[key]

    if "dataset_variant" in obj:
        out["preprocess_dataset_variant"] = obj["dataset_variant"]

    # Derive one canonical "actual raw sample loaded" field.
    actual = None
    if isinstance(raw_sampling, dict) and raw_sampling.get("actual_sample_size") is not None:
        actual = int(raw_sampling["actual_sample_size"])
    elif obj.get("num_raw_after_filter") is not None:
        actual = int(obj["num_raw_after_filter"])
    out["actual_raw_sample_size"] = actual

    # Mapped/admitted count.
    mapped = None
    for key in (
        "num_valid_mapped_trajectories",
        "num_valid_sequence_records",
    ):
        if obj.get(key) is not None:
            mapped = int(obj[key])
            break
    out["mapped_valid_trajectories"] = mapped

    return out


def selected_stage_records(manifest: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    records: List[Mapping[str, Any]] = []
    for rec in manifest.get("shared_stages", []):
        if isinstance(rec, dict):
            records.append(rec)
    for run in manifest.get("runs", []):
        if not isinstance(run, dict):
            continue
        for rec in run.get("stages", []):
            if isinstance(rec, dict):
                records.append(rec)
    return records


def assert_fresh_manifest(
    manifest: Mapping[str, Any],
    expected_stages: Sequence[str],
) -> None:
    records = selected_stage_records(manifest)
    by_stage: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for rec in records:
        by_stage[str(rec.get("stage", ""))].append(rec)

    problems = []
    for stage in expected_stages:
        stage_records = by_stage.get(stage, [])
        if not stage_records:
            problems.append(f"missing stage record: {stage}")
            continue

        statuses = {str(r.get("status", "")) for r in stage_records}
        if "SKIPPED_EXISTING" in statuses:
            problems.append(
                f"{stage} contains SKIPPED_EXISTING; scalability must be fresh."
            )

        if not any(
            str(r.get("status", "")) in {"DONE", "PASS", "SUCCESS"}
            for r in stage_records
        ):
            problems.append(
                f"{stage} has no successful fresh record; statuses={sorted(statuses)}"
            )

    if problems:
        raise RuntimeError(
            "Fresh scalability gate failed:\n- " + "\n- ".join(problems)
        )


def run_scalability(args: argparse.Namespace, root: Path) -> Dict[str, Any]:
    out = root / "scalability"
    dataset_path = resolve_path(args.dataset_config)
    exp_path = resolve_path(args.exp_config)
    assert dataset_path and exp_path

    raw_dataset = load_yaml(dataset_path)
    raw_exp = load_yaml(exp_path)

    scale_exp = profiled_exp_config(
        raw_exp,
        profile=args.scale_profile,
        seed=args.scale_seed,
        methods=["riskaware"],
        strip_fixed_dataset=True,
    )
    scale_exp_path = out / "configs" / "exp_scalability.generated.yaml"
    save_yaml(scale_exp_path, scale_exp)

    core_stages = [
        "events",
        "risk",
        "riskaware",
        "privacy_audit",
        "recovery",
    ]
    selected_stages = [
        "preprocess",
        *core_stages,
    ]

    rows: List[Dict[str, Any]] = []

    for size in [int(x) for x in args.sizes]:
        size_dataset = size_specific_dataset(raw_dataset, size)
        ds_path = out / "configs" / f"dataset_N{size}.yaml"
        save_yaml(ds_path, size_dataset)

        run_root = out / "canonical_runner" / f"N{size}"

        proc = run_process(
            runner_cmd(
                args,
                ds_path,
                scale_exp_path,
                mode="single",
                run_root=run_root,
                stages=selected_stages,
                raw_size=size,
                joint_mode="none",
                force_rerun=True,   # formal scalability is always fresh
                force_restart_map_matching=args.scale_force_restart_map_matching,
            ),
            out / "logs" / f"N{size}.log",
            monitor_memory=True,
        )

        manifest_path = newest_file(run_root, "run_manifest.json")
        if manifest_path is None:
            raise FileNotFoundError(
                f"run_manifest.json missing for scalability N={size}"
            )
        manifest = load_json(manifest_path)
        assert_fresh_manifest(manifest, selected_stages)

        stage_times: Dict[str, float] = defaultdict(float)
        preprocess_marker: Optional[Path] = None
        resolved_dataset_variant = None

        for rec in manifest.get("shared_stages", []):
            if rec.get("elapsed_sec") is not None:
                stage_times[str(rec.get("stage"))] += float(rec["elapsed_sec"])
            if rec.get("stage") == "preprocess" and rec.get("marker"):
                preprocess_marker = Path(str(rec["marker"]))

        for run in manifest.get("runs", []):
            if isinstance(run, dict):
                resolved = run.get("resolved_summary", {})
                if isinstance(resolved, dict):
                    resolved_dataset_variant = resolved.get(
                        "dataset_variant",
                        resolved_dataset_variant,
                    )
                for rec in run.get("stages", []):
                    if rec.get("elapsed_sec") is not None:
                        stage_times[str(rec.get("stage"))] += float(rec["elapsed_sec"])

        prep = parse_preprocess_summary(preprocess_marker)
        actual_raw = prep.get("actual_raw_sample_size")

        if actual_raw is None:
            raise RuntimeError(
                f"Cannot verify actual raw sample size for N={size}. "
                f"Preprocess summary: {preprocess_marker}"
            )

        if int(actual_raw) != int(size):
            if not args.scale_allow_shortfall:
                raise RuntimeError(
                    f"Requested scalability N={size}, but preprocessing reports "
                    f"actual_raw_sample_size={actual_raw}. "
                    "Use the exact available full count in --sizes, or pass "
                    "--scale_allow_shortfall only if you intentionally want to "
                    "report the actual count instead of the requested count."
                )

        row: Dict[str, Any] = {
            "dataset_name": raw_dataset.get("dataset_name"),
            "requested_raw_trajectories": int(size),
            "actual_raw_trajectories": int(actual_raw),
            "mapped_valid_trajectories": prep.get("mapped_valid_trajectories"),
            "dataset_variant": resolved_dataset_variant,
            "wall_sec_runner": proc["elapsed_sec"],
            "peak_rss_mb_process_tree": proc["peak_rss_mb_process_tree"],
            "common_preprocess_sec": stage_times.get("preprocess", 0.0),
            "trajrace_reporting_recovery_sec": sum(
                stage_times.get(stage, 0.0)
                for stage in core_stages
            ),
            "fresh_rerun": True,
            "force_restart_map_matching": bool(
                args.scale_force_restart_map_matching
            ),
            "manifest_path": str(manifest_path),
        }
        for stage in selected_stages:
            row[f"{stage}_sec"] = stage_times.get(stage, 0.0)

        row.update(prep)
        rows.append(row)

        # Incremental persistence: do not lose completed points on long sweeps.
        save_csv(out / "scalability.csv", rows)
        save_json(
            out / "scalability.json",
            {
                "profile": args.scale_profile,
                "seed": args.scale_seed,
                "timing_boundary": {
                    "common_preprocessing": [
                        "raw parsing",
                        "public graph preparation",
                        "map matching",
                    ],
                    "trajrace_reporting_recovery": core_stages,
                    "synthesis": (
                        "Not included in raw-N scaling core because N* is fixed "
                        "near test size; report separately if needed."
                    ),
                },
                "rows": rows,
            },
        )

    return {
        "profile": args.scale_profile,
        "seed": args.scale_seed,
        "result_csv": str(out / "scalability.csv"),
        "n_points": len(rows),
    }


# =============================================================================
# CLI and main
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Unified TrajRACE R1-W3/D3 experiment runner."
    )

    p.add_argument("--dataset_config", default=DEFAULT_DATASET_CONFIG)
    p.add_argument("--exp_config", default=DEFAULT_EXP_CONFIG)
    p.add_argument("--python", default=sys.executable)

    p.add_argument(
        "--mode",
        choices=[
            "multiseed",
            "attack",
            "baseline",
            "ablation",
            "scalability",
            "all",
        ],
        default="all",
    )
    p.add_argument("--output_root", default="w3d3_runs")
    p.add_argument("--rerun", action="store_true")

    # Multiseed.
    p.add_argument(
        "--profiles",
        nargs="*",
        default=["B0p5"],
        help="Formal reviewer priority is B0p5; add B1p0/B1p5 if desired.",
    )
    p.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=[42, 43, 44, 45, 46],
    )
    p.add_argument(
        "--table2_metric_csv",
        nargs="*",
        default=[],
        help=(
            "Exact per-seed Table-II metrics CSV paths/globs. "
            "These should be outputs of eval_ldptrace_3_compute_utility.py "
            "or the exact evaluator used for the submitted Table II."
        ),
    )
    p.add_argument(
        "--require_table2_metrics",
        nargs="*",
        default=list(DEFAULT_REQUIRED_TABLE2),
    )
    p.add_argument("--min_multiseed_runs", type=int, default=5)
    p.add_argument(
        "--allow_missing_table2",
        action="store_true",
        help="Development only. Formal rebuttal runs should not use this.",
    )

    # Attack.
    p.add_argument("--attack_profile", default="B0p5")
    p.add_argument("--attack_seed", type=int, default=42)
    p.add_argument("--attack_aux_split", default="train")
    p.add_argument("--attack_test_split", default="test")
    p.add_argument("--attack_shrinkage_mass", type=float, default=5.0)
    p.add_argument("--attack_min_support", type=int, default=3)

    # Adapted baseline.
    p.add_argument("--baseline_profile", default="B1p0")
    p.add_argument("--baseline_seed", type=int, default=42)

    # Ablation.
    p.add_argument("--ablation_profile", default="B0p5")
    p.add_argument("--ablation_seed", type=int, default=42)

    # Scalability.
    p.add_argument(
        "--sizes",
        nargs="*",
        type=int,
        default=[100000, 300000],
        help=(
            "Pass exact available counts, e.g. Porto "
            "100000 300000 1000000 1641024; Chengdu use the exact "
            "full available count rather than a rounded 6.1M label."
        ),
    )
    p.add_argument("--scale_profile", default="B1p0")
    p.add_argument("--scale_seed", type=int, default=42)
    p.add_argument(
        "--scale_force_restart_map_matching",
        action="store_true",
        help=(
            "Also force map-matching restart. Use for a completely uncached "
            "end-to-end preprocessing measurement; method-specific fresh "
            "reporting/recovery is forced regardless."
        ),
    )
    p.add_argument(
        "--scale_allow_shortfall",
        action="store_true",
        help=(
            "Allow actual raw count < requested size; the CSV will report the "
            "actual count. Prefer passing the exact full available count."
        ),
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()

    required_scripts = [
        "run_rebuttal_experiments.py",
        "evaluate_independent_attack.py",
        "privatize_adapted_nuldp.py",
        "recover_adapted_nuldp.py",
        "synthesize_trajectories.py",
        "evaluate_compare.py",
    ]
    missing = [
        name for name in required_scripts
        if not (SCRIPTS_DIR / name).exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Required scripts are missing from TrajRACE-main/scripts/: "
            + ", ".join(missing)
        )

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    base = resolve_path(args.output_root)
    assert base is not None
    root = base / f"w3d3_{args.mode}_{timestamp}"
    root.mkdir(parents=True, exist_ok=False)

    manifest: Dict[str, Any] = {
        "created_at_utc": utc_now(),
        "mode": args.mode,
        "dataset_config": str(resolve_path(args.dataset_config)),
        "exp_config": str(resolve_path(args.exp_config)),
        "profiles": list(args.profiles),
        "seeds": list(args.seeds),
        "results": {},
    }
    save_json(root / "w3d3_manifest.json", manifest)

    todo = (
        [args.mode]
        if args.mode != "all"
        else ["multiseed", "attack", "baseline", "ablation", "scalability"]
    )

    for mode in todo:
        print("\n" + "#" * 110)
        print(f"# R1-W3/D3 MODE: {mode}")
        print("#" * 110)

        if mode == "multiseed":
            result = run_multiseed(args, root)
        elif mode == "attack":
            result = run_attack(args, root)
        elif mode == "baseline":
            result = run_baseline(args, root)
        elif mode == "ablation":
            result = run_ablation(args, root)
        elif mode == "scalability":
            result = run_scalability(args, root)
        else:
            raise AssertionError(mode)

        manifest["results"][mode] = result
        save_json(root / "w3d3_manifest.json", manifest)

    manifest["finished_at_utc"] = utc_now()
    save_json(root / "w3d3_manifest.json", manifest)
    print(f"\n[DONE] W3/D3 artifact root: {root}")


if __name__ == "__main__":
    main()
