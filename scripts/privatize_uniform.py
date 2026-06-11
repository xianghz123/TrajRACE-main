#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import os
import sys
import json
import math
import time
import random
import argparse
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

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


def build_output_prefix(dataset_variant: str, method_name: str, exp_tag: str) -> str:
    return f"{dataset_variant}_{method_name}_{exp_tag}"


def nearest_bucket_label(eps_evt_uniform: float, eps_evt_list: List[float]) -> int:
    best_idx = 0
    best_gap = abs(float(eps_evt_uniform) - float(eps_evt_list[0]))
    for i in range(1, len(eps_evt_list)):
        gap = abs(float(eps_evt_uniform) - float(eps_evt_list[i]))
        if gap < best_gap:
            best_gap = gap
            best_idx = i
    return best_idx + 1


def extract_uniform_cfg(exp_cfg: Dict[str, Any]) -> Dict[str, Any]:
    required_keys = [
        "B_total", "eps_start", "eps_len", "eps_buck",
        "K", "eps_evt_list", "L_max", "random_seed"
    ]
    for k in required_keys:
        if k not in exp_cfg:
            raise KeyError(f"Missing required key in exp_main.yaml: {k}")

    K = int(exp_cfg["K"])
    eps_evt_list = [float(x) for x in exp_cfg["eps_evt_list"]]

    if len(eps_evt_list) != K:
        raise ValueError(f"eps_evt_list length must equal K={K}, got {len(eps_evt_list)}")

    for i in range(len(eps_evt_list) - 1):
        if not (eps_evt_list[i] < eps_evt_list[i + 1]):
            raise ValueError("eps_evt_list must be strictly increasing.")

    B_total = float(exp_cfg["B_total"])
    eps_start = float(exp_cfg["eps_start"])
    eps_len = float(exp_cfg["eps_len"])
    eps_buck = float(exp_cfg["eps_buck"])
    L_max = int(exp_cfg["L_max"])
    T_max = max(0, L_max - 1)

    if "eps_evt_uniform" in exp_cfg:
        eps_evt_uniform = float(exp_cfg["eps_evt_uniform"])
    else:
        if T_max <= 0:
            eps_evt_uniform = 1e-8
        else:
            eps_evt_uniform = (B_total - eps_start - eps_len) / T_max - eps_buck

    if eps_evt_uniform <= 0:
        raise ValueError(
            f"Derived eps_evt_uniform is non-positive ({eps_evt_uniform}). "
            f"Please increase B_total or decrease eps_start/eps_len/eps_buck/L_max."
        )

    if "uniform_bucket_label" in exp_cfg:
        fixed_exec_bucket = int(exp_cfg["uniform_bucket_label"])
    else:
        fixed_exec_bucket = nearest_bucket_label(eps_evt_uniform, eps_evt_list)

    if fixed_exec_bucket < 1 or fixed_exec_bucket > K:
        raise ValueError(f"uniform_bucket_label must be in [1, {K}]")

    return {
        "B_total": B_total,
        "eps_start": eps_start,
        "eps_len": eps_len,
        "eps_buck": eps_buck,
        "K": K,
        "eps_evt_list": eps_evt_list,
        "eps_evt_uniform": float(eps_evt_uniform),
        "fixed_exec_bucket": fixed_exec_bucket,
        "L_max": L_max,
        "random_seed": int(exp_cfg["random_seed"]),
        "save_debug_files": bool(exp_cfg.get("save_debug_files", True)),
    }


def validate_uniform_budget_feasibility(cfg: Dict[str, Any]) -> None:
    B_total = cfg["B_total"]
    eps_start = cfg["eps_start"]
    eps_len = cfg["eps_len"]
    eps_buck = cfg["eps_buck"]
    eps_evt_uniform = cfg["eps_evt_uniform"]
    T_max = max(0, int(cfg["L_max"]) - 1)

    rhs = eps_start + eps_len + T_max * (eps_evt_uniform + eps_buck)
    if B_total + 1e-12 < rhs:
        raise ValueError(
            f"Uniform budget infeasible: B_total={B_total} < "
            f"eps_start+eps_len+T_max*(eps_evt_uniform+eps_buck)={rhs:.6f}"
        )


def rr_sample(true_value: Any, domain: List[Any], epsilon: float, rng: random.Random) -> Any:
    if true_value not in domain:
        domain = list(domain) + [true_value]

    k = len(domain)
    if k <= 1:
        return true_value

    eps = max(float(epsilon), 1e-8)
    exp_eps = math.exp(eps)
    p_true = exp_eps / (exp_eps + k - 1)
    p_other = 1.0 / (exp_eps + k - 1)

    probs = [p_other] * k
    true_idx = domain.index(true_value)
    probs[true_idx] = p_true

    r = rng.random()
    acc = 0.0
    for val, p in zip(domain, probs):
        acc += p
        if r <= acc:
            return val
    return domain[-1]


def build_start_domain(train_events: List[Dict[str, Any]]) -> List[Any]:
    return sorted({rec["start_event"]["e1"] for rec in train_events})


def build_length_domain(train_events: List[Dict[str, Any]]) -> List[Any]:
    return sorted({rec["length_event"]["L"] for rec in train_events})


def build_risk_index(risk_items: List[Dict[str, Any]]) -> Dict[Tuple[str, int], Dict[str, Any]]:
    idx = {}
    for item in risk_items:
        idx[(item["traj_id"], int(item["pos"]))] = item
    return idx


# =============================================================================
# split 处理
# =============================================================================

def privatize_one_split_uniform(
    split_name: str,
    event_records: List[Dict[str, Any]],
    risk_items: List[Dict[str, Any]],
    successor_cache: Dict[str, List[str]],
    start_domain: List[Any],
    length_domain: List[Any],
    uniform_cfg: Dict[str, Any],
    rng: random.Random,
    progress_every: int = 50,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    eps_start = uniform_cfg["eps_start"]
    eps_len = uniform_cfg["eps_len"]
    eps_buck = uniform_cfg["eps_buck"]
    eps_evt_uniform = uniform_cfg["eps_evt_uniform"]
    fixed_exec_bucket = uniform_cfg["fixed_exec_bucket"]
    K = uniform_cfg["K"]

    risk_index = build_risk_index(risk_items)

    reports: List[Dict[str, Any]] = []
    debug_items: List[Dict[str, Any]] = []

    keep_by_target_bucket = Counter()
    total_by_target_bucket = Counter()

    keep_by_exec_bucket = Counter()
    total_by_exec_bucket = Counter()

    exec_bucket_counter = Counter()
    target_bucket_counter = Counter()

    total_transition_events = 0

    print(f"[Info] split={split_name}, num_event_records={len(event_records)}, num_risk_items={len(risk_items)}")

    for idx, rec, bar in progress_iter(
        event_records,
        desc=f"[race_5] {split_name}",
        progress_every=progress_every
    ):
        traj_id = rec["traj_id"]

        # 1) start
        true_e1 = rec["start_event"]["e1"]
        noisy_e1 = rr_sample(true_e1, start_domain, eps_start, rng)

        reports.append({
            "event_type": "start",
            "x_noisy": noisy_e1
        })

        debug_items.append({
            "event_type": "start",
            "traj_id": traj_id,
            "true_x": true_e1,
            "noisy_x": noisy_e1,
            "epsilon_used": eps_start,
            "keep_flag": int(noisy_e1 == true_e1)
        })

        # 2) length
        true_L = rec["length_event"]["L"]
        noisy_L = rr_sample(true_L, length_domain, eps_len, rng)

        reports.append({
            "event_type": "length",
            "x_noisy": noisy_L
        })

        debug_items.append({
            "event_type": "length",
            "traj_id": traj_id,
            "true_x": true_L,
            "noisy_x": noisy_L,
            "epsilon_used": eps_len,
            "keep_flag": int(noisy_L == true_L)
        })

        # 3) transition
        transitions = rec["transition_events"]

        for tr in transitions:
            pos = int(tr["pos"])
            u = tr["u"]
            v = tr["v"]

            risk_rec = risk_index.get((traj_id, pos))
            if risk_rec is None:
                raise KeyError(f"Missing risk record for (traj_id={traj_id}, pos={pos})")

            b_t = int(risk_rec["b_t"])
            k_t = int(fixed_exec_bucket)

            domain = list(successor_cache.get(u, []))
            if not domain:
                domain = [v]

            y_t = rr_sample(v, domain, eps_evt_uniform, rng)
            k_noisy = rr_sample(k_t, list(range(1, K + 1)), eps_buck, rng)

            reports.append({
                "event_type": "transition",
                "u": u,
                "y": y_t,
                "k_noisy": k_noisy
            })

            keep_flag = int(y_t == v)

            debug_items.append({
                "event_type": "transition",
                "traj_id": traj_id,
                "pos": pos,
                "u": u,
                "true_v": v,
                "y": y_t,
                "target_bucket_b_t": b_t,
                "exec_bucket_k_t": k_t,
                "bucket_k_noisy": k_noisy,
                "epsilon_evt_used": eps_evt_uniform,
                "candidate_size": len(domain),
                "keep_flag": keep_flag,
                "R_t": risk_rec.get("R_t", None),
                "phi_endpoint": risk_rec.get("phi_endpoint", None),
                "phi_stay": risk_rec.get("phi_stay", None),
                "phi_deg": risk_rec.get("phi_deg", None),
            })

            total_transition_events += 1
            target_bucket_counter[b_t] += 1
            exec_bucket_counter[k_t] += 1

            total_by_target_bucket[b_t] += 1
            keep_by_target_bucket[b_t] += keep_flag

            total_by_exec_bucket[k_t] += 1
            keep_by_exec_bucket[k_t] += keep_flag

        if bar is not None and (idx == 1 or idx % 10 == 0 or idx == len(event_records)):
            bar.set_postfix(
                reports=len(reports),
                transitions=total_transition_events
            )

    rng.shuffle(reports)

    keep_rate_by_target_bucket = {}
    keep_rate_by_exec_bucket = {}

    for b in range(1, K + 1):
        denom_t = total_by_target_bucket[b]
        denom_k = total_by_exec_bucket[b]

        keep_rate_by_target_bucket[str(b)] = (
            keep_by_target_bucket[b] / denom_t if denom_t > 0 else 0.0
        )
        keep_rate_by_exec_bucket[str(b)] = (
            keep_by_exec_bucket[b] / denom_k if denom_k > 0 else 0.0
        )

    keep_rate_overall = (
        sum(keep_by_target_bucket.values()) / total_transition_events
        if total_transition_events > 0 else 0.0
    )

    summary = {
        "split": split_name,
        "num_reports": len(reports),
        "num_debug_items": len(debug_items),
        "num_transition_events": total_transition_events,
        "keep_rate_overall": keep_rate_overall,
        "keep_rate_by_target_bucket": keep_rate_by_target_bucket,
        "keep_rate_by_exec_bucket": keep_rate_by_exec_bucket,
        "target_bucket_counts": {str(b): target_bucket_counter[b] for b in range(1, K + 1)},
        "exec_bucket_counts": {str(b): exec_bucket_counter[b] for b in range(1, K + 1)},
        "eps_evt_uniform": eps_evt_uniform,
        "fixed_exec_bucket": fixed_exec_bucket,
    }

    return reports, debug_items, summary


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

    uniform_cfg = extract_uniform_cfg(exp_cfg)
    validate_uniform_budget_feasibility(uniform_cfg)

    dataset_name = dataset_cfg.get("dataset_name", "dataset")
    dataset_variant = dataset_cfg["dataset_variant"]
    exp_tag = exp_cfg["exp_tag"]

    B_total = uniform_cfg["B_total"]
    random_seed = uniform_cfg["random_seed"]
    save_debug_files = uniform_cfg["save_debug_files"]

    required_keys = [
        "event_train", "event_valid", "event_test",
        "risk_train", "risk_valid", "risk_test",
        "successor_cache_path",
        "privatized_dir",
    ]
    for k in required_keys:
        if k not in dataset_cfg:
            raise KeyError(f"Missing required key in dataset config: {k}")

    event_train_path = resolve_path(dataset_cfg["event_train"])
    event_valid_path = resolve_path(dataset_cfg["event_valid"])
    event_test_path = resolve_path(dataset_cfg["event_test"])

    risk_train_path = resolve_path(dataset_cfg["risk_train"])
    risk_valid_path = resolve_path(dataset_cfg["risk_valid"])
    risk_test_path = resolve_path(dataset_cfg["risk_test"])

    successor_cache_path = resolve_path(dataset_cfg["successor_cache_path"])
    privatized_dir = resolve_path(dataset_cfg["privatized_dir"])

    for p in [
        event_train_path, event_valid_path, event_test_path,
        risk_train_path, risk_valid_path, risk_test_path,
        successor_cache_path
    ]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"required file not found: {p}")

    os.makedirs(privatized_dir, exist_ok=True)

    prefix = build_output_prefix(dataset_variant, "uniform", exp_tag)

    total_stages = 6

    # 1) load
    t = log_stage(1, total_stages, "Loading event files and risk files ...")
    train_events = read_jsonl(event_train_path)
    valid_events = read_jsonl(event_valid_path)
    test_events = read_jsonl(event_test_path)

    train_risk = read_jsonl(risk_train_path)
    valid_risk = read_jsonl(risk_valid_path)
    test_risk = read_jsonl(risk_test_path)

    print(f"[Info] dataset_name = {dataset_name}")
    print(f"[Info] dataset_variant = {dataset_variant}")
    print(f"[Info] dataset_root = {dataset_cfg['dataset_root']}")
    print(f"[Info] exp_tag = {exp_tag}")
    print(
        f"[Info] train/valid/test event records = "
        f"{len(train_events)}/{len(valid_events)}/{len(test_events)}"
    )
    print(
        f"[Info] train/valid/test risk items = "
        f"{len(train_risk)}/{len(valid_risk)}/{len(test_risk)}"
    )
    log_done(t, "Event and risk files loaded")

    # 2) cache + domains
    t = log_stage(2, total_stages, "Loading successor cache and building start/length domains ...")
    successor_cache = load_json(successor_cache_path)
    start_domain = build_start_domain(train_events)
    length_domain = build_length_domain(train_events)

    print(f"[Info] successor cache size = {len(successor_cache)}")
    print(f"[Info] start domain size = {len(start_domain)}")
    print(f"[Info] length domain size = {len(length_domain)}")
    print(f"[Info] eps_evt_uniform = {uniform_cfg['eps_evt_uniform']:.6f}")
    print(f"[Info] fixed_exec_bucket = {uniform_cfg['fixed_exec_bucket']}")
    print(f"[Info] output_prefix = {prefix}")
    log_done(t, "Caches and domains ready")

    rng = random.Random(random_seed)

    # 3) train
    t = log_stage(3, total_stages, "Privatizing train split with uniform mechanism ...")
    train_reports, train_debug, train_summary = privatize_one_split_uniform(
        split_name="train",
        event_records=train_events,
        risk_items=train_risk,
        successor_cache=successor_cache,
        start_domain=start_domain,
        length_domain=length_domain,
        uniform_cfg=uniform_cfg,
        rng=rng,
        progress_every=args.progress_every,
    )
    log_done(t, "Train split privatized")

    # 4) valid
    t = log_stage(4, total_stages, "Privatizing valid split with uniform mechanism ...")
    valid_reports, valid_debug, valid_summary = privatize_one_split_uniform(
        split_name="valid",
        event_records=valid_events,
        risk_items=valid_risk,
        successor_cache=successor_cache,
        start_domain=start_domain,
        length_domain=length_domain,
        uniform_cfg=uniform_cfg,
        rng=rng,
        progress_every=args.progress_every,
    )
    log_done(t, "Valid split privatized")

    # 5) test
    t = log_stage(5, total_stages, "Privatizing test split with uniform mechanism ...")
    test_reports, test_debug, test_summary = privatize_one_split_uniform(
        split_name="test",
        event_records=test_events,
        risk_items=test_risk,
        successor_cache=successor_cache,
        start_domain=start_domain,
        length_domain=length_domain,
        uniform_cfg=uniform_cfg,
        rng=rng,
        progress_every=args.progress_every,
    )
    log_done(t, "Test split privatized")

    # 6) save
    t = log_stage(6, total_stages, "Saving reports, debug files, meta and summary ...")

    train_reports_path = os.path.join(privatized_dir, f"{prefix}_train_reports.jsonl")
    valid_reports_path = os.path.join(privatized_dir, f"{prefix}_valid_reports.jsonl")
    test_reports_path = os.path.join(privatized_dir, f"{prefix}_test_reports.jsonl")

    write_jsonl(train_reports_path, train_reports)
    write_jsonl(valid_reports_path, valid_reports)
    write_jsonl(test_reports_path, test_reports)

    if save_debug_files:
        train_debug_path = os.path.join(privatized_dir, f"{prefix}_train_debug.jsonl")
        valid_debug_path = os.path.join(privatized_dir, f"{prefix}_valid_debug.jsonl")
        test_debug_path = os.path.join(privatized_dir, f"{prefix}_test_debug.jsonl")

        write_jsonl(train_debug_path, train_debug)
        write_jsonl(valid_debug_path, valid_debug)
        write_jsonl(test_debug_path, test_debug)

    meta = {
        "project_root": PROJECT_ROOT,
        "dataset_config": dataset_config_path,
        "exp_config": exp_config_path,
        "dataset_name": dataset_name,
        "dataset_variant": dataset_variant,
        "dataset_root": dataset_cfg["dataset_root"],
        "exp_tag": exp_tag,
        "version_info": version_info,
        "output_prefix": prefix,
        "method": "uniform",
        "B_total": B_total,
        "uniform_cfg": uniform_cfg,
        "start_domain_size": len(start_domain),
        "length_domain_size": len(length_domain),
        "successor_cache_size": len(successor_cache),
        "formal_report_format": {
            "start": {"event_type": "start", "x_noisy": "<segment>"},
            "length": {"event_type": "length", "x_noisy": "<length>"},
            "transition": {"event_type": "transition", "u": "<segment>", "y": "<segment>", "k_noisy": "<int>"},
        },
        "notes": {
            "formal_reports_are_deidentified": True,
            "formal_reports_do_not_include_traj_id": True,
            "formal_reports_are_shuffled": True,
            "all_transitions_use_uniform_epsilon": True,
        }
    }
    meta_path = os.path.join(privatized_dir, f"{prefix}_meta.json")
    save_json(meta_path, meta)

    summary = {
        "dataset_name": dataset_name,
        "dataset_variant": dataset_variant,
        "dataset_root": dataset_cfg["dataset_root"],
        "exp_tag": exp_tag,
        "version_info": version_info,
        "method": "uniform",
        "B_total": B_total,
        "train": train_summary,
        "valid": valid_summary,
        "test": test_summary,
    }
    summary_path = os.path.join(privatized_dir, f"{prefix}_summary.json")
    save_json(summary_path, summary)

    log_done(t, "All outputs saved")

    print("=" * 100)
    print("[race_5_privatize_uniform] Done.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("=" * 100)


if __name__ == "__main__":
    main()