#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import os
import sys
import csv
import json
import math
import time
import argparse
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


# =============================================================================
# 项目根目录
# =============================================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DEFAULT_DATASET_CONFIG = "configs/dataset_porto.yaml"
DEFAULT_EXP_CONFIG = "configs/exp_main.yaml"

from trajrace.versioning_utils import apply_versioning, pretty_version_summary


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


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def save_json(path: str, obj: Dict[str, Any]) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_csv(path: str, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# =============================================================================
# 日志
# =============================================================================

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


# =============================================================================
# 参数与命名
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
    return parser.parse_args()


def build_new_prefix(dataset_variant: str, method_name: str, exp_tag: str) -> str:
    return f"{dataset_variant}_{method_name}_{exp_tag}"


def build_legacy_prefix(dataset_name: str, method_name: str, B_total: float) -> str:
    return f"{dataset_name}_{method_name}_B{B_total}"


def build_result_prefix(dataset_variant: str, exp_tag: str) -> str:
    return f"{dataset_variant}_{exp_tag}"


def find_summary_path(
    privatized_dir: str,
    dataset_variant: str,
    dataset_name: str,
    method_name: str,
    exp_tag: str,
    B_total: float,
) -> Tuple[str, str]:
    new_prefix = build_new_prefix(dataset_variant, method_name, exp_tag)
    new_path = os.path.join(privatized_dir, f"{new_prefix}_summary.json")
    if os.path.exists(new_path):
        return new_path, new_prefix

    legacy_prefix = build_legacy_prefix(dataset_name, method_name, B_total)
    legacy_path = os.path.join(privatized_dir, f"{legacy_prefix}_summary.json")
    if os.path.exists(legacy_path):
        return legacy_path, legacy_prefix

    raise FileNotFoundError(
        f"Missing summary file for method={method_name}.\n"
        f"Tried:\n  {new_path}\n  {legacy_path}"
    )


def find_recovered_transition_path(
    recovered_dir: str,
    dataset_variant: str,
    dataset_name: str,
    method_name: str,
    exp_tag: str,
    B_total: float,
) -> Tuple[str, str]:
    new_prefix = build_new_prefix(dataset_variant, method_name, exp_tag)
    new_path = os.path.join(recovered_dir, f"{new_prefix}_test_transition.json")
    if os.path.exists(new_path):
        return new_path, new_prefix

    legacy_prefix = build_legacy_prefix(dataset_name, method_name, B_total)
    legacy_path = os.path.join(recovered_dir, f"{legacy_prefix}_test_transition.json")
    if os.path.exists(legacy_path):
        return legacy_path, legacy_prefix

    raise FileNotFoundError(
        f"Missing recovered transition file for method={method_name}.\n"
        f"Tried:\n  {new_path}\n  {legacy_path}"
    )


def find_synthetic_path(
    synthetic_dir: str,
    dataset_variant: str,
    dataset_name: str,
    method_name: str,
    exp_tag: str,
    B_total: float,
) -> Tuple[str, str]:
    new_prefix = build_new_prefix(dataset_variant, method_name, exp_tag)
    new_path = os.path.join(synthetic_dir, f"{new_prefix}_test_synthetic.jsonl")
    if os.path.exists(new_path):
        return new_path, new_prefix

    legacy_prefix = build_legacy_prefix(dataset_name, method_name, B_total)
    legacy_path = os.path.join(synthetic_dir, f"{legacy_prefix}_test_synthetic.jsonl")
    if os.path.exists(legacy_path):
        return legacy_path, legacy_prefix

    raise FileNotFoundError(
        f"Missing synthetic file for method={method_name}.\n"
        f"Tried:\n  {new_path}\n  {legacy_path}"
    )


# =============================================================================
# 工具函数
# =============================================================================

def cast_key_if_possible(x: Any) -> Any:
    if isinstance(x, int):
        return x
    if isinstance(x, float) and int(x) == x:
        return int(x)
    try:
        return int(x)
    except Exception:
        return x


def normalize_probs(prob_dict: Dict[Any, float]) -> Dict[Any, float]:
    clean = {}
    for k, v in prob_dict.items():
        try:
            val = float(v)
        except Exception:
            val = 0.0
        clean[k] = max(0.0, val)

    total = sum(clean.values())
    if total <= 0:
        n = len(clean)
        if n == 0:
            return {}
        return {k: 1.0 / n for k in clean}
    return {k: v / total for k, v in clean.items()}


def js_divergence(p_dict: Dict[Any, float], q_dict: Dict[Any, float]) -> float:
    support = set(p_dict.keys()) | set(q_dict.keys())
    if not support:
        return 0.0

    p = normalize_probs({k: p_dict.get(k, 0.0) for k in support})
    q = normalize_probs({k: q_dict.get(k, 0.0) for k in support})
    m = {k: 0.5 * (p.get(k, 0.0) + q.get(k, 0.0)) for k in support}

    def kl(a: Dict[Any, float], b: Dict[Any, float]) -> float:
        s = 0.0
        for k in support:
            av = a.get(k, 0.0)
            bv = b.get(k, 0.0)
            if av > 0 and bv > 0:
                s += av * math.log(av / bv)
        return s

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def build_true_transition_distribution(event_records: List[Dict[str, Any]]) -> Tuple[Dict[Any, Dict[Any, float]], Dict[Any, int]]:
    from collections import defaultdict
    per_u_counter = defaultdict(Counter)
    per_u_count = Counter()

    for rec in event_records:
        for tr in rec["transition_events"]:
            u = tr["u"]
            v = tr["v"]
            per_u_counter[u][v] += 1
            per_u_count[u] += 1

    dist = {}
    for u, counter in per_u_counter.items():
        total = sum(counter.values())
        dist[u] = {v: c / total for v, c in counter.items()}
    return dist, dict(per_u_count)


def parse_recovered_transition_json(path: str) -> Dict[Any, Dict[Any, float]]:
    obj = load_json(path)
    out = {}
    for u, info in obj.items():
        u_key = cast_key_if_possible(u)
        dist = info.get("distribution", {})
        out[u_key] = {cast_key_if_possible(v): float(p) for v, p in dist.items()}
    return out


def build_bigram_counter_from_sequences(records: List[Dict[str, Any]]) -> Counter:
    counter = Counter()
    for rec in records:
        segs = rec.get("segments", [])
        for i in range(len(segs) - 1):
            counter[(segs[i], segs[i + 1])] += 1
    return counter


def counter_to_prob(counter: Counter) -> Dict[Any, float]:
    total = sum(counter.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in counter.items()}


def compute_weighted_transition_js(
    true_dist: Dict[Any, Dict[Any, float]],
    true_counts: Dict[Any, int],
    recovered_dist: Dict[Any, Dict[Any, float]],
) -> Tuple[float, List[Dict[str, Any]]]:
    total_weight = 0.0
    weighted_sum = 0.0
    rows = []

    for u, p_true in true_dist.items():
        count_u = int(true_counts.get(u, 0))
        if count_u <= 0:
            continue

        p_hat = recovered_dist.get(u, {})
        js_u = js_divergence(p_true, p_hat)

        rows.append({
            "u": u,
            "count": count_u,
            "js": js_u,
            "true_domain_size": len(p_true),
            "recovered_domain_size": len(p_hat),
        })

        weighted_sum += js_u * count_u
        total_weight += count_u

    overall = (weighted_sum / total_weight) if total_weight > 0 else 0.0
    return overall, rows


def build_legal_bigram_set(all_event_records: List[Dict[str, Any]]) -> set:
    legal = set()
    for rec in all_event_records:
        segs = rec.get("segments", [])
        for i in range(len(segs) - 1):
            legal.add((segs[i], segs[i + 1]))
    return legal


def compute_legal_ratio(synthetic_records: List[Dict[str, Any]], legal_bigram_set: set) -> float:
    legal_count = 0
    total_count = 0
    for rec in synthetic_records:
        segs = rec.get("segments", [])
        for i in range(len(segs) - 1):
            total_count += 1
            if (segs[i], segs[i + 1]) in legal_bigram_set:
                legal_count += 1
    return (legal_count / total_count) if total_count > 0 else 0.0


def load_method_keep_summary(
    privatized_dir: str,
    dataset_variant: str,
    dataset_name: str,
    method_name: str,
    exp_tag: str,
    B_total: float
) -> Tuple[Dict[str, Any], str]:
    summary_path, used_prefix = find_summary_path(
        privatized_dir=privatized_dir,
        dataset_variant=dataset_variant,
        dataset_name=dataset_name,
        method_name=method_name,
        exp_tag=exp_tag,
        B_total=B_total,
    )
    return load_json(summary_path), used_prefix


# =============================================================================
# 绘图
# =============================================================================

def plot_single_bar(metric_name: str, method_to_value: Dict[str, float], output_path: str) -> None:
    ensure_parent_dir(output_path)
    methods = list(method_to_value.keys())
    values = [method_to_value[m] for m in methods]

    plt.figure(figsize=(6, 4))
    plt.bar(methods, values)
    plt.ylabel(metric_name)
    plt.title(metric_name)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_keep_rate_by_target_bucket(results: List[Dict[str, Any]], output_path: str) -> None:
    ensure_parent_dir(output_path)

    methods = [r["method"] for r in results]
    buckets = ["1", "2", "3"]

    x = [0, 1, 2]
    width = 0.35

    plt.figure(figsize=(7, 4))
    for idx, method in enumerate(methods):
        vals = []
        keep_dict = next(r["keep_rate_by_target_bucket"] for r in results if r["method"] == method)
        for b in buckets:
            vals.append(float(keep_dict.get(b, 0.0)))
        offset = (idx - (len(methods) - 1) / 2.0) * width
        plt.bar([v + offset for v in x], vals, width=width, label=method)

    plt.xticks(x, buckets)
    plt.xlabel("Target Bucket b_t")
    plt.ylabel("Keep Rate")
    plt.title("Keep Rate by Target Bucket")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


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

    dataset_name = dataset_cfg.get("dataset_name", "dataset")
    dataset_variant = dataset_cfg["dataset_variant"]
    exp_tag = exp_cfg["exp_tag"]
    B_total = float(exp_cfg["B_total"])
    methods = exp_cfg.get("methods", ["riskaware", "uniform"])

    event_train_path = resolve_path(dataset_cfg["event_train"])
    event_valid_path = resolve_path(dataset_cfg.get("event_valid"))
    event_test_path = resolve_path(dataset_cfg["event_test"])

    privatized_dir = resolve_path(dataset_cfg["privatized_dir"])
    recovered_dir = resolve_path(dataset_cfg["recovered_dir"])
    synthetic_dir = resolve_path(dataset_cfg["synthetic_dir"])

    outputs_tables_dir = resolve_path("outputs/tables")
    outputs_reports_dir = resolve_path("outputs/reports")
    outputs_figures_dir = resolve_path("outputs/figures")

    os.makedirs(outputs_tables_dir, exist_ok=True)
    os.makedirs(outputs_reports_dir, exist_ok=True)
    os.makedirs(outputs_figures_dir, exist_ok=True)

    result_prefix = build_result_prefix(dataset_variant, exp_tag)

    total_stages = 4

    # 1) real data
    t = log_stage(1, total_stages, "Loading real train/valid/test event data ...")
    for p in [event_train_path, event_test_path, privatized_dir, recovered_dir, synthetic_dir]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required path not found: {p}")

    train_events = read_jsonl(event_train_path)
    valid_events = read_jsonl(event_valid_path) if event_valid_path and os.path.exists(event_valid_path) else []
    test_events = read_jsonl(event_test_path)

    all_real_events = train_events + valid_events + test_events
    legal_bigram_set = build_legal_bigram_set(all_real_events)

    true_transition_dist, true_transition_counts = build_true_transition_distribution(test_events)
    true_bigram_prob = counter_to_prob(build_bigram_counter_from_sequences(test_events))

    print(f"[Info] dataset_name = {dataset_name}")
    print(f"[Info] dataset_variant = {dataset_variant}")
    print(f"[Info] dataset_root = {dataset_cfg['dataset_root']}")
    print(f"[Info] exp_tag = {exp_tag}")
    print(f"[Info] num_train_events = {len(train_events)}")
    print(f"[Info] num_valid_events = {len(valid_events)}")
    print(f"[Info] num_test_events  = {len(test_events)}")
    log_done(t, "Real data loaded")

    # 2) metrics
    t = log_stage(2, total_stages, "Computing evaluation metrics for each method ...")
    results = []
    transition_js_rows_all = []
    keep_rate_bucket_rows = []

    for method_name in methods:
        keep_summary, keep_prefix_used = load_method_keep_summary(
            privatized_dir=privatized_dir,
            dataset_variant=dataset_variant,
            dataset_name=dataset_name,
            method_name=method_name,
            exp_tag=exp_tag,
            B_total=B_total,
        )
        test_keep = keep_summary["test"]

        keep_rate_by_target_bucket = test_keep["keep_rate_by_target_bucket"]
        high_risk_keep_rate = float(keep_rate_by_target_bucket.get("1", 0.0))
        keep_rate_overall = float(test_keep["keep_rate_overall"])

        recovered_transition_path, recovered_prefix_used = find_recovered_transition_path(
            recovered_dir=recovered_dir,
            dataset_variant=dataset_variant,
            dataset_name=dataset_name,
            method_name=method_name,
            exp_tag=exp_tag,
            B_total=B_total,
        )
        recovered_transition = parse_recovered_transition_json(recovered_transition_path)

        transition_js, per_u_rows = compute_weighted_transition_js(
            true_dist=true_transition_dist,
            true_counts=true_transition_counts,
            recovered_dist=recovered_transition
        )
        for row in per_u_rows:
            row["method"] = method_name
            transition_js_rows_all.append(row)

        synthetic_path, synthetic_prefix_used = find_synthetic_path(
            synthetic_dir=synthetic_dir,
            dataset_variant=dataset_variant,
            dataset_name=dataset_name,
            method_name=method_name,
            exp_tag=exp_tag,
            B_total=B_total,
        )
        synthetic_records = read_jsonl(synthetic_path)

        synthetic_bigram_prob = counter_to_prob(build_bigram_counter_from_sequences(synthetic_records))
        synthetic_bigram_js = js_divergence(true_bigram_prob, synthetic_bigram_prob)

        legal_ratio = compute_legal_ratio(synthetic_records, legal_bigram_set)

        result_row = {
            "method": method_name,
            "input_keep_prefix_used": keep_prefix_used,
            "input_recovered_prefix_used": recovered_prefix_used,
            "input_synthetic_prefix_used": synthetic_prefix_used,
            "high_risk_keep_rate": high_risk_keep_rate,
            "keep_rate_overall": keep_rate_overall,
            "keep_rate_by_target_bucket": keep_rate_by_target_bucket,
            "transition_js": transition_js,
            "synthetic_bigram_js": synthetic_bigram_js,
            "legal_ratio": legal_ratio,
            "num_synthetic_trajectories": len(synthetic_records),
        }
        results.append(result_row)

        for b in ["1", "2", "3"]:
            keep_rate_bucket_rows.append({
                "method": method_name,
                "target_bucket_b_t": b,
                "keep_rate": float(keep_rate_by_target_bucket.get(b, 0.0)),
            })

        print(
            f"[Info] method={method_name} | "
            f"high_risk_keep_rate={high_risk_keep_rate:.6f}, "
            f"transition_js={transition_js:.6f}, "
            f"synthetic_bigram_js={synthetic_bigram_js:.6f}, "
            f"legal_ratio={legal_ratio:.6f}"
        )

    log_done(t, "Evaluation metrics computed")

    # 3) save tables/json
    t = log_stage(3, total_stages, "Saving tables and JSON reports ...")

    main_csv_rows = []
    for r in results:
        main_csv_rows.append({
            "method": r["method"],
            "high_risk_keep_rate": r["high_risk_keep_rate"],
            "keep_rate_overall": r["keep_rate_overall"],
            "transition_js": r["transition_js"],
            "synthetic_bigram_js": r["synthetic_bigram_js"],
            "legal_ratio": r["legal_ratio"],
            "num_synthetic_trajectories": r["num_synthetic_trajectories"],
            "input_keep_prefix_used": r["input_keep_prefix_used"],
            "input_recovered_prefix_used": r["input_recovered_prefix_used"],
            "input_synthetic_prefix_used": r["input_synthetic_prefix_used"],
        })

    save_csv(
        os.path.join(outputs_tables_dir, f"{result_prefix}_main_results.csv"),
        fieldnames=[
            "method",
            "high_risk_keep_rate",
            "keep_rate_overall",
            "transition_js",
            "synthetic_bigram_js",
            "legal_ratio",
            "num_synthetic_trajectories",
            "input_keep_prefix_used",
            "input_recovered_prefix_used",
            "input_synthetic_prefix_used",
        ],
        rows=main_csv_rows
    )

    save_csv(
        os.path.join(outputs_tables_dir, f"{result_prefix}_transition_js_per_u.csv"),
        fieldnames=["method", "u", "count", "js", "true_domain_size", "recovered_domain_size"],
        rows=transition_js_rows_all
    )

    save_csv(
        os.path.join(outputs_tables_dir, f"{result_prefix}_keep_rate_by_target_bucket.csv"),
        fieldnames=["method", "target_bucket_b_t", "keep_rate"],
        rows=keep_rate_bucket_rows
    )

    main_json = {
        "dataset_name": dataset_name,
        "dataset_variant": dataset_variant,
        "dataset_root": dataset_cfg["dataset_root"],
        "exp_tag": exp_tag,
        "version_info": version_info,
        "B_total": B_total,
        "results": results,
        "notes": {
            "high_risk_keep_rate": "Lower is better. Here high-risk refers to target bucket b_t=1.",
            "transition_js": "Lower is better.",
            "synthetic_bigram_js": "Lower is better.",
            "legal_ratio": "Higher is better.",
        }
    }
    save_json(os.path.join(outputs_reports_dir, f"{result_prefix}_main_results.json"), main_json)

    log_done(t, "Tables and JSON saved")

    # 4) figures
    t = log_stage(4, total_stages, "Generating figures ...")
    high_risk_fig = os.path.join(outputs_figures_dir, f"{result_prefix}_high_risk_keep_rate.png")
    transition_js_fig = os.path.join(outputs_figures_dir, f"{result_prefix}_transition_js.png")
    synthetic_bigram_js_fig = os.path.join(outputs_figures_dir, f"{result_prefix}_synthetic_bigram_js.png")
    legal_ratio_fig = os.path.join(outputs_figures_dir, f"{result_prefix}_legal_ratio.png")
    keep_rate_by_bucket_fig = os.path.join(outputs_figures_dir, f"{result_prefix}_keep_rate_by_target_bucket.png")

    plot_single_bar(
        "High-risk Keep Rate",
        {r["method"]: r["high_risk_keep_rate"] for r in results},
        high_risk_fig
    )
    plot_single_bar(
        "Transition JS",
        {r["method"]: r["transition_js"] for r in results},
        transition_js_fig
    )
    plot_single_bar(
        "Synthetic Bigram JS",
        {r["method"]: r["synthetic_bigram_js"] for r in results},
        synthetic_bigram_js_fig
    )
    plot_single_bar(
        "Legal Ratio",
        {r["method"]: r["legal_ratio"] for r in results},
        legal_ratio_fig
    )
    plot_keep_rate_by_target_bucket(results, keep_rate_by_bucket_fig)

    figure_manifest = {
        "dataset_name": dataset_name,
        "dataset_variant": dataset_variant,
        "dataset_root": dataset_cfg["dataset_root"],
        "exp_tag": exp_tag,
        "version_info": version_info,
        "input_csv": os.path.join(outputs_tables_dir, f"{result_prefix}_main_results.csv"),
        "input_json": os.path.join(outputs_reports_dir, f"{result_prefix}_main_results.json"),
        "figures": {
            "high_risk_keep_rate": high_risk_fig,
            "transition_js": transition_js_fig,
            "synthetic_bigram_js": synthetic_bigram_js_fig,
            "legal_ratio": legal_ratio_fig,
            "keep_rate_by_target_bucket": keep_rate_by_bucket_fig,
        }
    }
    save_json(os.path.join(outputs_reports_dir, f"{result_prefix}_figure_manifest.json"), figure_manifest)

    log_done(t, "Figures generated")

    print("=" * 100)
    print("[race_8_evaluate_compare] Done.")
    print(json.dumps(main_json, indent=2, ensure_ascii=False))
    print("=" * 100)


if __name__ == "__main__":
    main()