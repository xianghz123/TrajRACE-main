#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import os
import sys
import json
import math
import time
import argparse
from collections import Counter, defaultdict
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
    return parser.parse_args()


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
# 基础工具
# =============================================================================

def cast_key_if_possible(x: Any) -> Any:
    if isinstance(x, int):
        return x
    try:
        return int(x)
    except Exception:
        return x


def normalize_dict(d: Dict[Any, float]) -> Dict[Any, float]:
    total = sum(max(0.0, float(v)) for v in d.values())
    if total <= 0:
        n = len(d)
        if n == 0:
            return {}
        return {k: 1.0 / n for k in d}
    return {k: max(0.0, float(v)) / total for k, v in d.items()}


def rr_params(epsilon: float, k: int) -> Tuple[float, float]:
    if k <= 1:
        return 1.0, 0.0

    eps = max(float(epsilon), 1e-8)
    exp_eps = math.exp(eps)
    p = exp_eps / (exp_eps + k - 1)
    q = 1.0 / (exp_eps + k - 1)
    return p, q


def invert_symmetric_rr_distribution(
    observed_counter: Counter,
    domain: List[Any],
    p: float,
    q: float
) -> Dict[Any, float]:
    domain = list(domain)
    n = sum(observed_counter.values())
    if n <= 0:
        return {x: 0.0 for x in domain}

    if abs(p - q) < 1e-12:
        obs = {x: observed_counter.get(x, 0) / n for x in domain}
        return normalize_dict(obs)

    est = {}
    for x in domain:
        obs_freq = observed_counter.get(x, 0) / n
        est_freq = (obs_freq - q) / (p - q)
        est[x] = max(0.0, est_freq)

    return normalize_dict(est)


# =============================================================================
# 域构造
# =============================================================================

def build_start_domain(train_events: List[Dict[str, Any]]) -> List[Any]:
    return sorted({rec["start_event"]["e1"] for rec in train_events})


def build_length_domain(train_events: List[Dict[str, Any]]) -> List[Any]:
    return sorted({rec["length_event"]["L"] for rec in train_events})


def build_transition_domains_from_successor_cache(successor_cache: Dict[str, List[str]]) -> Dict[Any, List[Any]]:
    return {u: list(vs) for u, vs in successor_cache.items()}


# =============================================================================
# start / length 恢复
# =============================================================================

def recover_scalar_distribution_from_reports(
    reports: List[Dict[str, Any]],
    event_type: str,
    domain: List[Any],
    epsilon: float
) -> Dict[str, Any]:
    observed = Counter()
    for rec in reports:
        if rec.get("event_type") == event_type:
            observed[rec["x_noisy"]] += 1

    p, q = rr_params(epsilon, len(domain))
    dist = invert_symmetric_rr_distribution(observed, domain, p, q)

    return {
        "event_type": event_type,
        "domain_size": len(domain),
        "num_reports": sum(observed.values()),
        "distribution": {str(k): v for k, v in dist.items()}
    }


# =============================================================================
# uniform transition 恢复
# =============================================================================

def recover_transition_distribution_uniform(
    reports: List[Dict[str, Any]],
    transition_domains: Dict[Any, List[Any]],
    eps_evt_uniform: float
) -> Dict[str, Any]:
    per_u_obs = defaultdict(Counter)

    for rec in reports:
        if rec.get("event_type") == "transition":
            u = rec["u"]
            y = rec["y"]
            per_u_obs[u][y] += 1

    result = {}
    for u, obs_counter in per_u_obs.items():
        domain = transition_domains.get(u, list(obs_counter.keys()))
        if not domain:
            domain = list(obs_counter.keys())

        p, q = rr_params(eps_evt_uniform, len(domain))
        dist = invert_symmetric_rr_distribution(obs_counter, domain, p, q)

        result[str(u)] = {
            "domain_size": len(domain),
            "num_reports": sum(obs_counter.values()),
            "distribution": {str(v): prob for v, prob in dist.items()}
        }

    return result


# =============================================================================
# riskaware transition 恢复
# =============================================================================

def recover_bucket_mix_from_noisy_counts(
    noisy_counter: Counter,
    K: int,
    eps_buck: float
) -> Dict[int, float]:
    domain = list(range(1, K + 1))
    p, q = rr_params(eps_buck, K)
    dist = invert_symmetric_rr_distribution(noisy_counter, domain, p, q)
    return {int(k): float(v) for k, v in dist.items()}


def shrink_bucket_mix(
    local_mix: Dict[int, float],
    global_mix: Dict[int, float],
    N_u: int,
    tau_shrinkage: float,
    K: int
) -> Dict[int, float]:
    alpha = float(N_u) / float(N_u + tau_shrinkage) if (N_u + tau_shrinkage) > 0 else 0.0

    mixed = {}
    for k in range(1, K + 1):
        v = alpha * float(local_mix.get(k, 0.0)) + (1.0 - alpha) * float(global_mix.get(k, 0.0))
        mixed[k] = v
    return normalize_dict(mixed)


def mixed_rr_params_for_u(
    pi_u: Dict[int, float],
    eps_evt_list: List[float],
    domain_size: int,
    K: int
) -> Tuple[float, float]:
    pbar = 0.0
    qbar = 0.0
    for k in range(1, K + 1):
        eps_k = float(eps_evt_list[k - 1])
        p_k, q_k = rr_params(eps_k, domain_size)
        weight = float(pi_u.get(k, 0.0))
        pbar += weight * p_k
        qbar += weight * q_k
    return pbar, qbar


def recover_transition_distribution_riskaware(
    reports: List[Dict[str, Any]],
    transition_domains: Dict[Any, List[Any]],
    eps_buck: float,
    eps_evt_list: List[float],
    tau_shrinkage: float,
    K: int
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    per_u_y_counter = defaultdict(Counter)
    per_u_noisy_bucket_counter = defaultdict(Counter)
    global_noisy_bucket_counter = Counter()

    for rec in reports:
        if rec.get("event_type") != "transition":
            continue
        u = rec["u"]
        y = rec["y"]
        k_noisy = int(rec["k_noisy"])

        per_u_y_counter[u][y] += 1
        per_u_noisy_bucket_counter[u][k_noisy] += 1
        global_noisy_bucket_counter[k_noisy] += 1

    pi_glob = recover_bucket_mix_from_noisy_counts(global_noisy_bucket_counter, K=K, eps_buck=eps_buck)

    transition_result = {}
    bucket_mix_result = {
        "global_mix": {str(k): v for k, v in pi_glob.items()},
        "per_u_mix": {}
    }

    for u, y_counter in per_u_y_counter.items():
        domain = transition_domains.get(u, list(y_counter.keys()))
        if not domain:
            domain = list(y_counter.keys())

        N_u = sum(y_counter.values())
        local_noisy = per_u_noisy_bucket_counter[u]
        pi_u_loc = recover_bucket_mix_from_noisy_counts(local_noisy, K=K, eps_buck=eps_buck)

        pi_u = shrink_bucket_mix(
            local_mix=pi_u_loc,
            global_mix=pi_glob,
            N_u=N_u,
            tau_shrinkage=tau_shrinkage,
            K=K
        )

        pbar, qbar = mixed_rr_params_for_u(
            pi_u=pi_u,
            eps_evt_list=eps_evt_list,
            domain_size=len(domain),
            K=K
        )

        dist = invert_symmetric_rr_distribution(y_counter, domain, pbar, qbar)

        transition_result[str(u)] = {
            "domain_size": len(domain),
            "num_reports": N_u,
            "distribution": {str(v): prob for v, prob in dist.items()}
        }

        bucket_mix_result["per_u_mix"][str(u)] = {
            "num_reports": N_u,
            "pi_u_loc": {str(k): float(pi_u_loc.get(k, 0.0)) for k in range(1, K + 1)},
            "pi_u_shrunk": {str(k): float(pi_u.get(k, 0.0)) for k in range(1, K + 1)},
            "alpha_u": float(N_u) / float(N_u + tau_shrinkage) if (N_u + tau_shrinkage) > 0 else 0.0
        }

    return transition_result, bucket_mix_result


# =============================================================================
# 主恢复逻辑
# =============================================================================

def recover_one_method_one_split(
    method_name: str,
    split_name: str,
    reports: List[Dict[str, Any]],
    start_domain: List[Any],
    length_domain: List[Any],
    transition_domains: Dict[Any, List[Any]],
    cfg: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Optional[Dict[str, Any]], Dict[str, Any]]:
    start_rec = recover_scalar_distribution_from_reports(
        reports=reports,
        event_type="start",
        domain=start_domain,
        epsilon=cfg["eps_start"]
    )
    length_rec = recover_scalar_distribution_from_reports(
        reports=reports,
        event_type="length",
        domain=length_domain,
        epsilon=cfg["eps_len"]
    )

    if method_name == "uniform":
        transition_rec = recover_transition_distribution_uniform(
            reports=reports,
            transition_domains=transition_domains,
            eps_evt_uniform=cfg["eps_evt_uniform"]
        )
        bucket_mix_rec = None
    else:
        transition_rec, bucket_mix_rec = recover_transition_distribution_riskaware(
            reports=reports,
            transition_domains=transition_domains,
            eps_buck=cfg["eps_buck"],
            eps_evt_list=cfg["eps_evt_list"],
            tau_shrinkage=cfg["tau_shrinkage"],
            K=cfg["K"]
        )

    split_summary = {
        "split": split_name,
        "method": method_name,
        "num_start_reports": start_rec["num_reports"],
        "num_length_reports": length_rec["num_reports"],
        "num_transition_u": len(transition_rec),
        "num_transition_reports": sum(v["num_reports"] for v in transition_rec.values()) if transition_rec else 0,
    }

    return start_rec, length_rec, transition_rec, bucket_mix_rec, split_summary


# =============================================================================
# 参数
# =============================================================================

def extract_recovery_cfg(exp_cfg: Dict[str, Any]) -> Dict[str, Any]:
    required_keys = [
        "B_total", "eps_start", "eps_len", "eps_buck",
        "K", "eps_evt_list", "L_max", "tau_shrinkage"
    ]
    for k in required_keys:
        if k not in exp_cfg:
            raise KeyError(f"Missing required key in exp_main.yaml: {k}")

    K = int(exp_cfg["K"])
    eps_evt_list = [float(x) for x in exp_cfg["eps_evt_list"]]
    if len(eps_evt_list) != K:
        raise ValueError(f"eps_evt_list length must equal K={K}, got {len(eps_evt_list)}")

    B_total = float(exp_cfg["B_total"])
    eps_start = float(exp_cfg["eps_start"])
    eps_len = float(exp_cfg["eps_len"])
    eps_buck = float(exp_cfg["eps_buck"])
    L_max = int(exp_cfg["L_max"])
    T_max = max(0, L_max - 1)

    if "eps_evt_uniform" in exp_cfg:
        eps_evt_uniform = float(exp_cfg["eps_evt_uniform"])
    else:
        eps_evt_uniform = (B_total - eps_start - eps_len) / T_max - eps_buck if T_max > 0 else 1e-8

    if eps_evt_uniform <= 0:
        raise ValueError(
            f"Derived eps_evt_uniform is non-positive ({eps_evt_uniform}). "
            f"Please increase B_total or decrease eps_start/eps_len/eps_buck/L_max."
        )

    return {
        "B_total": B_total,
        "eps_start": eps_start,
        "eps_len": eps_len,
        "eps_buck": eps_buck,
        "K": K,
        "eps_evt_list": eps_evt_list,
        "eps_evt_uniform": eps_evt_uniform,
        "tau_shrinkage": float(exp_cfg["tau_shrinkage"]),
    }


# =============================================================================
# 命名工具
# =============================================================================

def build_new_prefix(dataset_variant: str, method_name: str, exp_tag: str) -> str:
    return f"{dataset_variant}_{method_name}_{exp_tag}"


def build_legacy_prefix(dataset_name: str, method_name: str, B_total: float) -> str:
    return f"{dataset_name}_{method_name}_B{B_total}"


def find_report_path(
    privatized_dir: str,
    dataset_variant: str,
    dataset_name: str,
    method_name: str,
    exp_tag: str,
    B_total: float,
    split_name: str,
) -> Tuple[str, str]:
    new_prefix = build_new_prefix(dataset_variant, method_name, exp_tag)
    new_path = os.path.join(privatized_dir, f"{new_prefix}_{split_name}_reports.jsonl")
    if os.path.exists(new_path):
        return new_path, new_prefix

    legacy_prefix = build_legacy_prefix(dataset_name, method_name, B_total)
    legacy_path = os.path.join(privatized_dir, f"{legacy_prefix}_{split_name}_reports.jsonl")
    if os.path.exists(legacy_path):
        return legacy_path, legacy_prefix

    raise FileNotFoundError(
        f"Missing report file for method={method_name}, split={split_name}. "
        f"Tried:\n  {new_path}\n  {legacy_path}"
    )


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

    cfg = extract_recovery_cfg(exp_cfg)

    dataset_name = str(dataset_cfg.get("dataset_name", "dataset"))
    dataset_variant = dataset_cfg["dataset_variant"]
    exp_tag = exp_cfg["exp_tag"]
    B_total = cfg["B_total"]

    required_keys = [
        "event_train",
        "successor_cache_path",
        "privatized_dir",
        "recovered_dir",
    ]
    for k in required_keys:
        if k not in dataset_cfg:
            raise KeyError(f"Missing required key in dataset config: {k}")

    event_train_path = resolve_path(dataset_cfg["event_train"])
    successor_cache_path = resolve_path(dataset_cfg["successor_cache_path"])
    privatized_dir = resolve_path(dataset_cfg["privatized_dir"])
    recovered_dir = resolve_path(dataset_cfg["recovered_dir"])

    if not os.path.exists(event_train_path):
        raise FileNotFoundError(f"Missing event_train file: {event_train_path}")
    if not os.path.exists(successor_cache_path):
        raise FileNotFoundError(f"Missing successor cache: {successor_cache_path}")
    if not os.path.exists(privatized_dir):
        raise FileNotFoundError(f"Missing privatized_dir: {privatized_dir}")

    os.makedirs(recovered_dir, exist_ok=True)

    methods = exp_cfg.get("methods", ["riskaware", "uniform"])
    methods = [str(x) for x in methods]
    splits = ["train", "valid", "test"]

    total_stages = 4

    # 1) 读取域信息
    t = log_stage(1, total_stages, "Loading train events and successor domains ...")
    train_events = read_jsonl(event_train_path)
    successor_cache = load_json(successor_cache_path)

    start_domain = build_start_domain(train_events)
    length_domain = build_length_domain(train_events)
    transition_domains = build_transition_domains_from_successor_cache(successor_cache)

    print(f"[Info] dataset_name = {dataset_name}")
    print(f"[Info] dataset_variant = {dataset_variant}")
    print(f"[Info] dataset_root = {dataset_cfg['dataset_root']}")
    print(f"[Info] exp_tag = {exp_tag}")
    print(f"[Info] start domain size = {len(start_domain)}")
    print(f"[Info] length domain size = {len(length_domain)}")
    print(f"[Info] transition domain size (#u) = {len(transition_domains)}")
    log_done(t, "Domains ready")

    # 2) 读取 report 文件
    t = log_stage(2, total_stages, "Loading privatized reports ...")
    reports_dict: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(dict)
    used_prefix_dict: Dict[str, str] = {}

    for method in methods:
        used_prefix_for_method = None
        for split in splits:
            path, used_prefix = find_report_path(
                privatized_dir=privatized_dir,
                dataset_variant=dataset_variant,
                dataset_name=dataset_name,
                method_name=method,
                exp_tag=exp_tag,
                B_total=B_total,
                split_name=split,
            )
            reports_dict[method][split] = read_jsonl(path)
            print(f"[Info] loaded {method}/{split} reports = {len(reports_dict[method][split])} | path={path}")

            if used_prefix_for_method is None:
                used_prefix_for_method = used_prefix
            elif used_prefix_for_method != used_prefix:
                raise RuntimeError(
                    f"Inconsistent prefixes used for method={method}: "
                    f"{used_prefix_for_method} vs {used_prefix}"
                )

        used_prefix_dict[method] = used_prefix_for_method or build_new_prefix(dataset_variant, method, exp_tag)

    log_done(t, "Reports loaded")

    # 3) 执行恢复
    t = log_stage(3, total_stages, "Recovering statistics for riskaware and uniform ...")
    all_summary = {
        "dataset_name": dataset_name,
        "dataset_variant": dataset_variant,
        "dataset_root": dataset_cfg["dataset_root"],
        "exp_tag": exp_tag,
        "version_info": version_info,
        "B_total": B_total,
        "methods": {}
    }

    for method in methods:
        output_prefix = build_new_prefix(dataset_variant, method, exp_tag)

        method_summary = {
            "method": method,
            "input_prefix_used": used_prefix_dict[method],
            "output_prefix": output_prefix,
            "splits": {}
        }

        for split in splits:
            start_rec, length_rec, transition_rec, bucket_mix_rec, split_summary = recover_one_method_one_split(
                method_name=method,
                split_name=split,
                reports=reports_dict[method][split],
                start_domain=start_domain,
                length_domain=length_domain,
                transition_domains=transition_domains,
                cfg=cfg
            )

            save_json(os.path.join(recovered_dir, f"{output_prefix}_{split}_start.json"), start_rec)
            save_json(os.path.join(recovered_dir, f"{output_prefix}_{split}_length.json"), length_rec)
            save_json(os.path.join(recovered_dir, f"{output_prefix}_{split}_transition.json"), transition_rec)

            if bucket_mix_rec is not None:
                save_json(os.path.join(recovered_dir, f"{output_prefix}_{split}_bucket_mix.json"), bucket_mix_rec)

            method_summary["splits"][split] = split_summary

        summary_path = os.path.join(recovered_dir, f"{output_prefix}_recovery_summary.json")
        save_json(summary_path, method_summary)
        all_summary["methods"][method] = method_summary

    log_done(t, "Statistics recovered")

    # 4) 保存总览
    t = log_stage(4, total_stages, "Saving recovery overview ...")
    overview_path = os.path.join(recovered_dir, f"{dataset_variant}_{exp_tag}_recovery_overview.json")
    save_json(overview_path, all_summary)
    log_done(t, "Overview saved")

    print("=" * 100)
    print("[race_6_recover_statistics] Done.")
    print(json.dumps(all_summary, indent=2, ensure_ascii=False))
    print("=" * 100)


if __name__ == "__main__":
    main()