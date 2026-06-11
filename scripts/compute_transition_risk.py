#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import os
import sys
import csv
import json
import time
import argparse
from typing import Any, Dict, List, Optional, Iterable, Tuple

# =============================================================================
# 项目根目录
# =============================================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DEFAULT_DATASET_CONFIG = "configs/dataset_porto.yaml"
DEFAULT_EXP_CONFIG = "configs/exp_main.yaml"

from trajrace.versioning_utils import apply_versioning, pretty_version_summary
from trajrace.risk_utils import (
    compute_transition_risks_for_record,
    summarize_target_buckets,
    build_segment_graph_from_successor_cache,
)


# =============================================================================
# 基础 IO
# =============================================================================

def resolve_path(path_str: Optional[str]) -> Optional[str]:
    if path_str is None:
        return None
    if os.path.isabs(path_str):
        return path_str
    return os.path.abspath(os.path.join(PROJECT_ROOT, path_str))


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def load_yaml_simple(path: str) -> Dict[str, Any]:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, obj: Dict[str, Any]) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def write_jsonl(path: str, items: List[Dict[str, Any]]) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def save_csv(path: str, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# =============================================================================
# 进度与日志
# =============================================================================

def maybe_get_tqdm():
    try:
        from tqdm import tqdm
        return tqdm
    except Exception:
        return None


def fmt_sec(sec: float) -> str:
    if sec < 60:
        return f"{sec:.1f}s"
    m = int(sec // 60)
    s = sec - 60 * m
    return f"{m}m{s:.1f}s"


def log_stage(stage_idx: int, total_stage: int, msg: str) -> float:
    print(f"\n[Stage {stage_idx}/{total_stage}] {msg}")
    return time.perf_counter()


def log_done(stage_start: float, msg: str) -> None:
    dt = time.perf_counter() - stage_start
    print(f"[Done] {msg} | elapsed={fmt_sec(dt)}")


def progress_iter(
    iterable: List[Any],
    desc: str,
    progress_every: int = 50,
    ncols: int = 110
):
    tqdm = maybe_get_tqdm()
    total = len(iterable)

    if tqdm is not None:
        bar = tqdm(iterable, total=total, desc=desc, ncols=ncols)
        for idx, item in enumerate(bar, start=1):
            yield idx, item, bar
    else:
        bar = None
        for idx, item in enumerate(iterable, start=1):
            if idx == 1 or idx % max(1, progress_every) == 0 or idx == total:
                pct = 100.0 * idx / max(1, total)
                print(f"[{desc}] {idx}/{total} ({pct:.1f}%)")
            yield idx, item, bar


# =============================================================================
# 参数
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_config",
        type=str,
        default=DEFAULT_DATASET_CONFIG,
        help=f"Path to dataset yaml. Default: {DEFAULT_DATASET_CONFIG}"
    )
    parser.add_argument(
        "--exp_config",
        type=str,
        default=DEFAULT_EXP_CONFIG,
        help=f"Path to experiment yaml. Default: {DEFAULT_EXP_CONFIG}"
    )
    parser.add_argument(
        "--progress_every",
        type=int,
        default=50,
        help="Fallback text progress interval when tqdm is unavailable."
    )
    return parser.parse_args()


def extract_risk_cfg(exp_cfg: Dict[str, Any]) -> Dict[str, Any]:
    required_keys = [
        "lambda_e", "lambda_s", "lambda_d",
        "sigma_e", "sigma_s",
        "T_stay",
        "K",
        "theta_list",
    ]
    for k in required_keys:
        if k not in exp_cfg:
            raise KeyError(f"Missing required risk parameter in exp_main.yaml: {k}")

    K = int(exp_cfg["K"])
    theta_list = list(exp_cfg["theta_list"])

    if K < 2:
        raise ValueError(f"K must be >= 2, got {K}")
    if len(theta_list) != K - 1:
        raise ValueError(
            f"theta_list length must be K-1={K-1}, got {len(theta_list)}"
        )

    return {
        "lambda_e": float(exp_cfg["lambda_e"]),
        "lambda_s": float(exp_cfg["lambda_s"]),
        "lambda_d": float(exp_cfg["lambda_d"]),
        "sigma_e": float(exp_cfg["sigma_e"]),
        "sigma_s": float(exp_cfg["sigma_s"]),
        "T_stay": float(exp_cfg["T_stay"]),
        "K": K,
        "theta_list": [float(x) for x in theta_list],
    }


# =============================================================================
# 风险计算
# =============================================================================

def compute_risk_for_split(
    event_records: List[Dict[str, Any]],
    split_name: str,
    successor_cache: Dict[str, List[str]],
    distance_cache: Dict[str, float],
    risk_cfg: Dict[str, Any],
    progress_every: int = 50,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    segment_graph = build_segment_graph_from_successor_cache(successor_cache)

    risk_items: List[Dict[str, Any]] = []
    num_event_records = len(event_records)
    num_failed_records = 0

    print(f"[Info] split={split_name}, num_event_records={num_event_records}")

    for idx, rec, bar in progress_iter(
        event_records,
        desc=f"[race_3] {split_name}",
        progress_every=progress_every
    ):
        try:
            items = compute_transition_risks_for_record(
                event_record=rec,
                successor_cache=successor_cache,
                risk_cfg=risk_cfg,
                distance_cache=distance_cache,
                segment_graph=segment_graph
            )
            risk_items.extend(items)

            if bar is not None and (idx == 1 or idx % 10 == 0 or idx == num_event_records):
                bar.set_postfix(risk_items=len(risk_items), failed=num_failed_records)

        except Exception:
            num_failed_records += 1
            if bar is not None and (idx == 1 or idx % 10 == 0 or idx == num_event_records):
                bar.set_postfix(risk_items=len(risk_items), failed=num_failed_records)

    bucket_summary = summarize_target_buckets(risk_items, K=risk_cfg["K"])

    split_summary = {
        "split": split_name,
        "num_event_records": num_event_records,
        "num_failed_records": num_failed_records,
        "num_transition_risk_items": len(risk_items),
        "bucket_summary": bucket_summary,
    }
    return risk_items, split_summary


def bucket_summary_to_rows(bucket_summary: Dict[str, Any], split_name: str) -> List[Dict[str, Any]]:
    rows = []
    for b_t, stats in bucket_summary.items():
        rows.append({
            "split": split_name,
            "b_t": b_t,
            "count": stats.get("count", 0),
            "avg_R_t": stats.get("avg_R_t", 0.0),
            "avg_phi_endpoint": stats.get("avg_phi_endpoint", 0.0),
            "avg_phi_stay": stats.get("avg_phi_stay", 0.0),
            "avg_phi_deg": stats.get("avg_phi_deg", 0.0),
            "avg_delta_t": stats.get("avg_delta_t", 0.0),
        })
    return rows


# =============================================================================
# 主函数
# =============================================================================

def main():
    args = parse_args()

    dataset_config_path = resolve_path(args.dataset_config)
    exp_config_path = resolve_path(args.exp_config)

    if not os.path.exists(dataset_config_path):
        raise FileNotFoundError(f"dataset config not found: {dataset_config_path}")
    if not os.path.exists(exp_config_path):
        raise FileNotFoundError(f"exp config not found: {exp_config_path}")

    raw_dataset_cfg = load_yaml_simple(dataset_config_path)
    raw_exp_cfg = load_yaml_simple(exp_config_path)

    dataset_cfg, exp_cfg = apply_versioning(raw_dataset_cfg, raw_exp_cfg)
    version_info = pretty_version_summary(raw_dataset_cfg, raw_exp_cfg)

    risk_cfg = extract_risk_cfg(exp_cfg)

    dataset_name = dataset_cfg.get("dataset_name", "dataset")
    dataset_variant = dataset_cfg["dataset_variant"]
    exp_tag = exp_cfg["exp_tag"]

    required_dataset_keys = [
        "event_train", "event_valid", "event_test",
        "risk_train", "risk_valid", "risk_test",
        "successor_cache_path", "distance_cache_path",
    ]
    for k in required_dataset_keys:
        if k not in dataset_cfg:
            raise KeyError(f"Missing required key in dataset config: {k}")

    event_train_path = resolve_path(dataset_cfg["event_train"])
    event_valid_path = resolve_path(dataset_cfg["event_valid"])
    event_test_path = resolve_path(dataset_cfg["event_test"])

    risk_train_path = resolve_path(dataset_cfg["risk_train"])
    risk_valid_path = resolve_path(dataset_cfg["risk_valid"])
    risk_test_path = resolve_path(dataset_cfg["risk_test"])

    successor_cache_path = resolve_path(dataset_cfg["successor_cache_path"])
    distance_cache_path = resolve_path(dataset_cfg["distance_cache_path"])

    for p in [event_train_path, event_valid_path, event_test_path, successor_cache_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"required input file not found: {p}")

    total_stages = 6

    # 1) 读取事件文件
    t = log_stage(1, total_stages, "Loading event files ...")
    train_events = read_jsonl(event_train_path)
    valid_events = read_jsonl(event_valid_path)
    test_events = read_jsonl(event_test_path)

    print(f"[Info] dataset_name = {dataset_name}")
    print(f"[Info] dataset_variant = {dataset_variant}")
    print(f"[Info] dataset_root = {dataset_cfg['dataset_root']}")
    print(f"[Info] exp_tag = {exp_tag}")
    print(f"[Info] train/valid/test event records = {len(train_events)}/{len(valid_events)}/{len(test_events)}")
    log_done(t, "Event files loaded")

    # 2) 读取公共先验缓存
    t = log_stage(2, total_stages, "Loading successor cache and distance cache ...")
    successor_cache = load_json(successor_cache_path)

    if os.path.exists(distance_cache_path):
        distance_cache = load_json(distance_cache_path)
    else:
        distance_cache = {}
        save_json(distance_cache_path, distance_cache)

    print(f"[Info] successor cache size = {len(successor_cache)}")
    print(f"[Info] distance cache size (before run) = {len(distance_cache)}")
    log_done(t, "Caches loaded")

    # 3) 计算 train 风险
    t = log_stage(3, total_stages, "Computing transition risk for train split ...")
    train_risk_items, train_summary = compute_risk_for_split(
        event_records=train_events,
        split_name="train",
        successor_cache=successor_cache,
        distance_cache=distance_cache,
        risk_cfg=risk_cfg,
        progress_every=args.progress_every,
    )
    write_jsonl(risk_train_path, train_risk_items)
    log_done(t, f"Train risk written to {risk_train_path}")

    # 4) 计算 valid 风险
    t = log_stage(4, total_stages, "Computing transition risk for valid split ...")
    valid_risk_items, valid_summary = compute_risk_for_split(
        event_records=valid_events,
        split_name="valid",
        successor_cache=successor_cache,
        distance_cache=distance_cache,
        risk_cfg=risk_cfg,
        progress_every=args.progress_every,
    )
    write_jsonl(risk_valid_path, valid_risk_items)
    log_done(t, f"Valid risk written to {risk_valid_path}")

    # 5) 计算 test 风险
    t = log_stage(5, total_stages, "Computing transition risk for test split ...")
    test_risk_items, test_summary = compute_risk_for_split(
        event_records=test_events,
        split_name="test",
        successor_cache=successor_cache,
        distance_cache=distance_cache,
        risk_cfg=risk_cfg,
        progress_every=args.progress_every,
    )
    write_jsonl(risk_test_path, test_risk_items)
    log_done(t, f"Test risk written to {risk_test_path}")

    # 6) 保存 summary / csv / distance cache
    t = log_stage(6, total_stages, "Saving summaries and updated distance cache ...")

    risk_dir = os.path.dirname(risk_train_path)
    summary_prefix = f"{dataset_variant}_{exp_tag}"

    train_summary_json = os.path.join(risk_dir, f"{summary_prefix}_target_bucket_summary_train.json")
    valid_summary_json = os.path.join(risk_dir, f"{summary_prefix}_target_bucket_summary_valid.json")
    test_summary_json = os.path.join(risk_dir, f"{summary_prefix}_target_bucket_summary_test.json")

    save_json(train_summary_json, train_summary)
    save_json(valid_summary_json, valid_summary)
    save_json(test_summary_json, test_summary)

    train_rows = bucket_summary_to_rows(train_summary["bucket_summary"], "train")
    valid_rows = bucket_summary_to_rows(valid_summary["bucket_summary"], "valid")
    test_rows = bucket_summary_to_rows(test_summary["bucket_summary"], "test")

    train_csv = os.path.join(risk_dir, f"{summary_prefix}_target_bucket_summary_train.csv")
    valid_csv = os.path.join(risk_dir, f"{summary_prefix}_target_bucket_summary_valid.csv")
    test_csv = os.path.join(risk_dir, f"{summary_prefix}_target_bucket_summary_test.csv")

    fieldnames = [
        "split", "b_t", "count",
        "avg_R_t", "avg_phi_endpoint", "avg_phi_stay", "avg_phi_deg", "avg_delta_t"
    ]
    save_csv(train_csv, fieldnames=fieldnames, rows=train_rows)
    save_csv(valid_csv, fieldnames=fieldnames, rows=valid_rows)
    save_csv(test_csv, fieldnames=fieldnames, rows=test_rows)

    overall_summary = {
        "project_root": PROJECT_ROOT,
        "dataset_config": dataset_config_path,
        "exp_config": exp_config_path,
        "dataset_name": dataset_name,
        "dataset_variant": dataset_variant,
        "dataset_root": dataset_cfg["dataset_root"],
        "exp_tag": exp_tag,
        "version_info": version_info,
        "risk_cfg": risk_cfg,
        "num_train_transition_risk_items": len(train_risk_items),
        "num_valid_transition_risk_items": len(valid_risk_items),
        "num_test_transition_risk_items": len(test_risk_items),
        "train": train_summary,
        "valid": valid_summary,
        "test": test_summary,
        "distance_cache_size_after_run": len(distance_cache),
    }

    overall_summary_path = os.path.join(risk_dir, f"{summary_prefix}_risk_summary.json")
    save_json(overall_summary_path, overall_summary)

    save_json(distance_cache_path, distance_cache)

    print(f"[Info] distance cache size (after run) = {len(distance_cache)}")
    log_done(t, "Summaries and caches saved")

    print("=" * 100)
    print("[race_3_compute_transition_risk] Done.")
    print(json.dumps(overall_summary, indent=2, ensure_ascii=False))
    print("=" * 100)


if __name__ == "__main__":
    main()