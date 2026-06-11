#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import os
import sys
import json
import time
import random
import argparse
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Iterable


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


def write_jsonl(path: str, items: List[Dict[str, Any]]) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


# =============================================================================
# 日志与进度
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
) -> Iterable[Tuple[int, Any, Any]]:
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
    parser.add_argument(
        "--progress_every",
        type=int,
        default=50,
        help="Fallback text progress interval when tqdm is unavailable."
    )
    return parser.parse_args()


def build_new_prefix(dataset_variant: str, method_name: str, exp_tag: str) -> str:
    return f"{dataset_variant}_{method_name}_{exp_tag}"


def build_legacy_prefix(dataset_name: str, method_name: str, B_total: float) -> str:
    return f"{dataset_name}_{method_name}_B{B_total}"


def find_recovered_component_path(
    recovered_dir: str,
    dataset_variant: str,
    dataset_name: str,
    method_name: str,
    exp_tag: str,
    B_total: float,
    split_name: str,
    component_name: str,
) -> Tuple[str, str]:
    """
    查找 recovered 统计文件，查找顺序：
    1) 当前 dataset_variant + method + exp_tag 的新版精确命名
    2) 旧版 legacy 命名
    3) 同一 dataset_variant + method + split + component 下任意已有 exp_tag 的文件（按修改时间最新优先）

    返回:
      - path
      - used_prefix
    """
    import glob

    def strip_suffix(path_str: str, split_name: str, component_name: str) -> str:
        fname = os.path.basename(path_str)
        suffix = f"_{split_name}_{component_name}.json"
        if fname.endswith(suffix):
            return fname[:-len(suffix)]
        return os.path.splitext(fname)[0]

    tried = []

    # 1) 当前 exp_tag 对应的新版命名
    new_prefix = build_new_prefix(dataset_variant, method_name, exp_tag)
    new_path = os.path.join(recovered_dir, f"{new_prefix}_{split_name}_{component_name}.json")
    tried.append(new_path)
    if os.path.exists(new_path):
        return new_path, new_prefix

    # 2) 旧版 legacy 命名
    legacy_prefix = build_legacy_prefix(dataset_name, method_name, B_total)
    legacy_path = os.path.join(recovered_dir, f"{legacy_prefix}_{split_name}_{component_name}.json")
    tried.append(legacy_path)
    if os.path.exists(legacy_path):
        return legacy_path, legacy_prefix

    # 3) 回退：同一 dataset_variant + method + split + component 下，任意已有 exp_tag 文件
    pattern_variant = os.path.join(
        recovered_dir,
        f"{dataset_variant}_{method_name}_*_{split_name}_{component_name}.json"
    )
    candidates_variant = [
        p for p in glob.glob(pattern_variant)
        if os.path.isfile(p)
    ]

    if candidates_variant:
        candidates_variant = sorted(
            set(candidates_variant),
            key=lambda p: os.path.getmtime(p),
            reverse=True
        )
        chosen = candidates_variant[0]
        used_prefix = strip_suffix(chosen, split_name, component_name)
        print(
            f"[Warn] Current exp_tag recovered file not found. "
            f"Fallback to existing recovered file:\n  {chosen}"
        )
        return chosen, used_prefix

    raise FileNotFoundError(
        f"Required recovered file not found for method={method_name}, "
        f"split={split_name}, component={component_name}.\n"
        f"Tried exact/legacy paths:\n  " + "\n  ".join(tried) +
        f"\nAlso searched fallback pattern:\n  {pattern_variant}"
    )


# =============================================================================
# 长度模式工具
# =============================================================================

def parse_length_bucket_edges(exp_cfg: Dict[str, Any]) -> List[int]:
    edges = exp_cfg.get("length_bucket_edges", [5, 10, 15, 20, 25])
    if not isinstance(edges, list) or len(edges) == 0:
        raise ValueError("length_bucket_edges must be a non-empty list.")
    parsed = [int(x) for x in edges]
    parsed = sorted(parsed)
    for i in range(len(parsed) - 1):
        if parsed[i] >= parsed[i + 1]:
            raise ValueError("length_bucket_edges must be strictly increasing.")
    return parsed


def bucket_id_to_range(
    bucket_id: int,
    bucket_edges: List[int],
    min_segments: int,
    max_segments: int,
) -> Tuple[int, int]:
    b = int(bucket_id)
    K = len(bucket_edges) + 1
    if b < 1:
        b = 1
    if b > K:
        b = K

    if b == 1:
        lo = int(min_segments)
        hi = int(bucket_edges[0])
    elif b == K:
        lo = int(bucket_edges[-1]) + 1
        hi = int(max_segments)
    else:
        lo = int(bucket_edges[b - 2]) + 1
        hi = int(bucket_edges[b - 1])

    if lo > hi:
        lo, hi = hi, lo
    return lo, hi


def sample_target_length(
    sampled_len_label: Any,
    length_mode: str,
    bucket_edges: List[int],
    min_segments: int,
    max_segments: int,
    rng: random.Random,
) -> int:
    if sampled_len_label is None:
        return max(1, int(min_segments))

    if length_mode == "num_segments":
        try:
            target_len = int(sampled_len_label)
        except Exception:
            target_len = int(min_segments)
        return max(1, target_len)

    if length_mode == "num_segments_bucket6":
        try:
            bucket_id = int(sampled_len_label)
        except Exception:
            bucket_id = 1
        lo, hi = bucket_id_to_range(
            bucket_id=bucket_id,
            bucket_edges=bucket_edges,
            min_segments=min_segments,
            max_segments=max_segments,
        )
        return max(1, rng.randint(lo, hi))

    raise ValueError(f"Unsupported length_mode: {length_mode}")


# =============================================================================
# 分布工具
# =============================================================================

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


def sample_from_distribution(prob_dict: Dict[Any, float], rng: random.Random) -> Any:
    dist = normalize_probs(prob_dict)
    if not dist:
        return None

    r = rng.random()
    acc = 0.0
    last_key = None
    for k, p in dist.items():
        acc += p
        last_key = k
        if r <= acc:
            return k
    return last_key


def cast_key_if_possible(x: Any) -> Any:
    if isinstance(x, int):
        return x
    if isinstance(x, float) and int(x) == x:
        return int(x)
    try:
        return int(x)
    except Exception:
        return x


def parse_distribution_json(path: str) -> Tuple[Dict[Any, float], int]:
    obj = load_json(path)
    raw = obj.get("distribution", {})
    num_reports = int(obj.get("num_reports", 0))
    dist = {cast_key_if_possible(k): float(v) for k, v in raw.items()}
    return dist, num_reports


def parse_transition_json_with_meta(path: str) -> Dict[Any, Dict[str, Any]]:
    obj = load_json(path)
    out = {}
    for u, info in obj.items():
        u_key = cast_key_if_possible(u)
        dist = info.get("distribution", {})
        out[u_key] = {
            "domain_size": int(info.get("domain_size", 0)),
            "num_reports": int(info.get("num_reports", 0)),
            "distribution": {cast_key_if_possible(v): float(p) for v, p in dist.items()}
        }
    return out


def weighted_merge_scalar_distributions(distributions: List[Tuple[Dict[Any, float], int]]) -> Dict[Any, float]:
    acc = defaultdict(float)
    total_weight = 0.0

    for dist, weight in distributions:
        w = float(max(0, weight))
        if w <= 0:
            continue
        dist = normalize_probs(dist)
        for k, p in dist.items():
            acc[k] += p * w
        total_weight += w

    if total_weight <= 0:
        merged = {}
        for dist, _ in distributions:
            for k, p in dist.items():
                merged[k] = merged.get(k, 0.0) + p
        return normalize_probs(merged)

    return normalize_probs({k: v / total_weight for k, v in acc.items()})


def weighted_merge_transition_distributions(
    trans_dicts: List[Dict[Any, Dict[str, Any]]]
) -> Dict[Any, Dict[str, Any]]:
    """
    输出:
      merged[u] = {
        "num_reports": merged_weight,
        "distribution": {v: p}
      }
    """
    merged = defaultdict(lambda: defaultdict(float))
    merged_weight = defaultdict(float)
    merged_domain_size = defaultdict(int)

    for trans_obj in trans_dicts:
        for u, info in trans_obj.items():
            w = float(max(0, info.get("num_reports", 0)))
            dist = normalize_probs(info.get("distribution", {}))
            domain_size = int(info.get("domain_size", len(dist)))

            if w <= 0:
                w = 1.0

            for v, p in dist.items():
                merged[u][v] += p * w
            merged_weight[u] += w
            merged_domain_size[u] = max(merged_domain_size[u], domain_size)

    final = {}
    for u, vdict in merged.items():
        w = merged_weight[u]
        if w <= 0:
            dist = normalize_probs(vdict)
        else:
            dist = normalize_probs({v: score / w for v, score in vdict.items()})

        final[u] = {
            "num_reports": int(round(w)),
            "domain_size": int(max(merged_domain_size[u], len(dist))),
            "distribution": dist
        }
    return final


def take_topk_and_renorm(prob_dict: Dict[Any, float], topk: int) -> Dict[Any, float]:
    if not prob_dict:
        return {}
    if topk is None or topk <= 0 or len(prob_dict) <= topk:
        return normalize_probs(prob_dict)
    items = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)[:topk]
    return normalize_probs(dict(items))


def build_global_transition_backoff(transition_info: Dict[Any, Dict[str, Any]], global_topk: int) -> Dict[Any, float]:
    acc = defaultdict(float)
    total_weight = 0.0

    for _, info in transition_info.items():
        w = float(max(1, info.get("num_reports", 1)))
        dist = normalize_probs(info.get("distribution", {}))
        for v, p in dist.items():
            acc[v] += p * w
        total_weight += w

    if total_weight <= 0:
        return {}

    raw = {v: score / total_weight for v, score in acc.items()}
    return take_topk_and_renorm(raw, global_topk)


def filter_continuable_candidates(
    cand_dist: Dict[Any, float],
    transition_info: Dict[Any, Dict[str, Any]],
    remaining_steps_after_this: int
) -> Dict[Any, float]:
    if remaining_steps_after_this <= 0:
        return cand_dist

    continuable = {
        v: p for v, p in cand_dist.items()
        if v in transition_info and len(transition_info[v].get("distribution", {})) > 0
    }
    if continuable:
        return continuable
    return cand_dist


def blend_distributions(
    local_dist: Dict[Any, float],
    global_dist: Dict[Any, float],
    alpha_local: float
) -> Dict[Any, float]:
    alpha = min(max(float(alpha_local), 0.0), 1.0)

    local_dist = normalize_probs(local_dist)
    global_dist = normalize_probs(global_dist)

    support = set(local_dist.keys()) | set(global_dist.keys())
    if not support:
        return {}

    out = {}
    for k in support:
        out[k] = alpha * local_dist.get(k, 0.0) + (1.0 - alpha) * global_dist.get(k, 0.0)
    return normalize_probs(out)


def prune_small_probs(prob_dict: Dict[Any, float], min_prob: float) -> Dict[Any, float]:
    if not prob_dict:
        return {}
    threshold = max(0.0, float(min_prob))
    kept = {k: v for k, v in prob_dict.items() if float(v) >= threshold}
    if not kept:
        return normalize_probs(prob_dict)
    return normalize_probs(kept)


def top1_prob(prob_dict: Dict[Any, float]) -> float:
    if not prob_dict:
        return 0.0
    return max(float(v) for v in prob_dict.values())


def prepare_transition_distributions(
    transition_info: Dict[Any, Dict[str, Any]],
    global_backoff_dist: Dict[Any, float],
    use_global_backoff: bool,
    support_blend_cfg: Dict[str, Any],
) -> Dict[Any, Dict[str, Any]]:
    """
    对每个 u 预计算可直接采样的分布，避免 synthesis 时重复混合。

    输出:
      prepared[u] = {
        "distribution": {v: p},
        "num_reports": int,
        "used_low_support_blend": 0/1,
        "alpha_local": float,
        "top1_local": float
      }
    """
    prepared = {}

    min_reports_direct = int(support_blend_cfg["support_blend_min_reports"])
    support_blend_tau = float(support_blend_cfg["support_blend_tau"])
    extreme_low_reports = int(support_blend_cfg["support_blend_extreme_low_reports"])
    top1_threshold = float(support_blend_cfg["support_blend_top1_threshold"])
    min_local_alpha_for_extreme = float(support_blend_cfg["support_blend_extreme_alpha_cap"])
    min_prob = float(support_blend_cfg["support_blend_min_prob"])
    local_topk = int(support_blend_cfg["support_blend_local_topk"])
    global_topk = int(support_blend_cfg["support_blend_global_topk"])

    global_trunc = take_topk_and_renorm(global_backoff_dist, global_topk)

    for u, info in transition_info.items():
        local_dist = normalize_probs(info.get("distribution", {}))
        local_dist = take_topk_and_renorm(local_dist, local_topk)
        num_reports_u = int(info.get("num_reports", 0))
        t1 = top1_prob(local_dist)

        used_blend = 0
        alpha_local = 1.0

        if not local_dist:
            if use_global_backoff and global_trunc:
                final_dist = global_trunc
            else:
                final_dist = {}
            prepared[u] = {
                "distribution": final_dist,
                "num_reports": num_reports_u,
                "used_low_support_blend": 0,
                "alpha_local": 0.0 if final_dist else 1.0,
                "top1_local": 0.0,
            }
            continue

        if use_global_backoff and global_trunc:
            need_blend = num_reports_u < min_reports_direct
            extreme_spiky = (num_reports_u <= extreme_low_reports and t1 >= top1_threshold)

            if need_blend:
                alpha_local = float(num_reports_u) / float(num_reports_u + support_blend_tau) if (num_reports_u + support_blend_tau) > 0 else 0.0
                if extreme_spiky:
                    alpha_local = min(alpha_local, min_local_alpha_for_extreme)

                final_dist = blend_distributions(local_dist, global_trunc, alpha_local)
                used_blend = 1
            else:
                final_dist = local_dist
        else:
            final_dist = local_dist

        final_dist = prune_small_probs(final_dist, min_prob)

        if not final_dist and use_global_backoff and global_trunc:
            final_dist = global_trunc

        prepared[u] = {
            "distribution": final_dist,
            "num_reports": num_reports_u,
            "used_low_support_blend": used_blend,
            "alpha_local": alpha_local,
            "top1_local": t1,
        }

    return prepared


# =============================================================================
# 合成
# =============================================================================

def synthesize_one_trajectory(
    syn_id: str,
    start_dist: Dict[Any, float],
    length_dist: Dict[Any, float],
    prepared_transition_info: Dict[Any, Dict[str, Any]],
    global_backoff_dist: Dict[Any, float],
    use_global_backoff: bool,
    length_mode: str,
    bucket_edges: List[int],
    min_segments: int,
    max_segments: int,
    rng: random.Random,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    start_seg = sample_from_distribution(start_dist, rng)
    sampled_len_label = sample_from_distribution(length_dist, rng)

    if start_seg is None:
        rec = {"syn_id": syn_id, "segments": []}
        diag = {
            "sampled_length_label": None,
            "target_length": 0,
            "actual_length": 0,
            "early_stop": 1,
            "num_hard_backoff_used": 0,
            "num_low_support_blend_used": 0,
            "num_invalid_steps": 0,
        }
        return rec, diag

    target_len = sample_target_length(
        sampled_len_label=sampled_len_label,
        length_mode=length_mode,
        bucket_edges=bucket_edges,
        min_segments=min_segments,
        max_segments=max_segments,
        rng=rng,
    )
    target_len = max(1, target_len)

    segments = [start_seg]
    early_stop = 0
    invalid_steps = 0
    hard_backoff_used = 0
    low_support_blend_used = 0

    while len(segments) < target_len:
        u = segments[-1]
        remaining_after_this = target_len - (len(segments) + 1)

        info = prepared_transition_info.get(u, None)

        if info is None or not info.get("distribution"):
            if use_global_backoff and global_backoff_dist:
                cand_dist = global_backoff_dist
                hard_backoff_used += 1
            else:
                early_stop = 1
                break
        else:
            cand_dist = info["distribution"]
            low_support_blend_used += int(info.get("used_low_support_blend", 0))

        if not cand_dist:
            early_stop = 1
            break

        cand_dist = filter_continuable_candidates(
            cand_dist=cand_dist,
            transition_info=prepared_transition_info,
            remaining_steps_after_this=remaining_after_this
        )

        v = sample_from_distribution(cand_dist, rng)
        if v is None:
            early_stop = 1
            invalid_steps += 1
            break

        segments.append(v)

    if len(segments) < target_len:
        early_stop = 1

    rec = {
        "syn_id": syn_id,
        "segments": segments
    }
    diag = {
        "sampled_length_label": sampled_len_label,
        "target_length": target_len,
        "actual_length": len(segments),
        "early_stop": early_stop,
        "num_hard_backoff_used": hard_backoff_used,
        "num_low_support_blend_used": low_support_blend_used,
        "num_invalid_steps": invalid_steps,
    }
    return rec, diag


def summarize_synthetic(diags: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not diags:
        return {
            "num_synthetic_trajectories": 0,
            "avg_target_length": 0.0,
            "avg_actual_length": 0.0,
            "early_stop_ratio": 0.0,
            "avg_backoff_used": 0.0,
            "avg_hard_backoff_used": 0.0,
            "avg_low_support_blend_used": 0.0,
            "avg_invalid_steps": 0.0,
            "sampled_length_label_counts": {},
        }

    n = len(diags)
    label_counter = defaultdict(int)
    for x in diags:
        label = x.get("sampled_length_label", None)
        label_counter[str(label)] += 1

    avg_hard_backoff = sum(x["num_hard_backoff_used"] for x in diags) / n
    avg_blend = sum(x["num_low_support_blend_used"] for x in diags) / n

    return {
        "num_synthetic_trajectories": n,
        "avg_target_length": sum(x["target_length"] for x in diags) / n,
        "avg_actual_length": sum(x["actual_length"] for x in diags) / n,
        "early_stop_ratio": sum(x["early_stop"] for x in diags) / n,
        "avg_backoff_used": avg_hard_backoff,
        "avg_hard_backoff_used": avg_hard_backoff,
        "avg_low_support_blend_used": avg_blend,
        "avg_invalid_steps": sum(x["num_invalid_steps"] for x in diags) / n,
        "sampled_length_label_counts": dict(sorted(label_counter.items())),
    }


def synthesize_one_method(
    method_name: str,
    dataset_name: str,
    dataset_variant: str,
    exp_tag: str,
    B_total: float,
    synthetic_num: int,
    recovered_dir: str,
    synthetic_dir: str,
    use_global_backoff: bool,
    synthesis_splits: List[str],
    length_mode: str,
    bucket_edges: List[int],
    min_segments: int,
    max_segments: int,
    support_blend_cfg: Dict[str, Any],
    rng: random.Random,
    progress_every: int = 50,
) -> Dict[str, Any]:
    output_prefix = build_new_prefix(dataset_variant, method_name, exp_tag)

    allowed_splits = {"train", "valid", "test"}
    for sp in synthesis_splits:
        if sp not in allowed_splits:
            raise ValueError(f"Unsupported synthesis split: {sp}")

    start_parts = []
    length_parts = []
    trans_parts = []

    used_prefix = None

    for sp in synthesis_splits:
        start_path, used_prefix_start = find_recovered_component_path(
            recovered_dir=recovered_dir,
            dataset_variant=dataset_variant,
            dataset_name=dataset_name,
            method_name=method_name,
            exp_tag=exp_tag,
            B_total=B_total,
            split_name=sp,
            component_name="start",
        )
        length_path, used_prefix_length = find_recovered_component_path(
            recovered_dir=recovered_dir,
            dataset_variant=dataset_variant,
            dataset_name=dataset_name,
            method_name=method_name,
            exp_tag=exp_tag,
            B_total=B_total,
            split_name=sp,
            component_name="length",
        )
        transition_path, used_prefix_transition = find_recovered_component_path(
            recovered_dir=recovered_dir,
            dataset_variant=dataset_variant,
            dataset_name=dataset_name,
            method_name=method_name,
            exp_tag=exp_tag,
            B_total=B_total,
            split_name=sp,
            component_name="transition",
        )

        prefixes_used = {used_prefix_start, used_prefix_length, used_prefix_transition}
        if len(prefixes_used) != 1:
            raise RuntimeError(
                f"Inconsistent recovered prefixes for method={method_name}, split={sp}: {prefixes_used}"
            )

        current_prefix = prefixes_used.pop()
        if used_prefix is None:
            used_prefix = current_prefix
        elif used_prefix != current_prefix:
            raise RuntimeError(
                f"Inconsistent recovered prefix across splits for method={method_name}: "
                f"{used_prefix} vs {current_prefix}"
            )

        start_parts.append(parse_distribution_json(start_path))
        length_parts.append(parse_distribution_json(length_path))
        trans_parts.append(parse_transition_json_with_meta(transition_path))

    merged_start = weighted_merge_scalar_distributions(start_parts)
    merged_length = weighted_merge_scalar_distributions(length_parts)
    merged_transition = weighted_merge_transition_distributions(trans_parts)

    global_backoff_dist = build_global_transition_backoff(
        merged_transition,
        global_topk=int(support_blend_cfg["support_blend_global_topk"])
    )

    prepared_transition_info = prepare_transition_distributions(
        transition_info=merged_transition,
        global_backoff_dist=global_backoff_dist,
        use_global_backoff=use_global_backoff,
        support_blend_cfg=support_blend_cfg,
    )

    syn_records = []
    syn_diags = []

    items = list(range(synthetic_num))
    for idx, i, bar in progress_iter(
        items,
        desc=f"[race_7] {method_name}",
        progress_every=progress_every
    ):
        syn_id = f"{method_name}_syn_{i:06d}"
        rec, diag = synthesize_one_trajectory(
            syn_id=syn_id,
            start_dist=merged_start,
            length_dist=merged_length,
            prepared_transition_info=prepared_transition_info,
            global_backoff_dist=global_backoff_dist,
            use_global_backoff=use_global_backoff,
            length_mode=length_mode,
            bucket_edges=bucket_edges,
            min_segments=min_segments,
            max_segments=max_segments,
            rng=rng,
        )
        syn_records.append(rec)
        syn_diags.append(diag)

        if bar is not None and (idx == 1 or idx % 10 == 0 or idx == synthetic_num):
            bar.set_postfix(
                avg_len=f"{sum(x['actual_length'] for x in syn_diags) / len(syn_diags):.2f}",
                early_stop=f"{sum(x['early_stop'] for x in syn_diags) / len(syn_diags):.3f}"
            )

    syn_path = os.path.join(synthetic_dir, f"{output_prefix}_test_synthetic.jsonl")
    write_jsonl(syn_path, syn_records)

    prepared_low_support_count = sum(
        1 for _, info in prepared_transition_info.items()
        if int(info.get("used_low_support_blend", 0)) == 1
    )

    summary = summarize_synthetic(syn_diags)
    summary.update({
        "dataset_name": dataset_name,
        "dataset_variant": dataset_variant,
        "exp_tag": exp_tag,
        "input_prefix_used": used_prefix,
        "output_prefix": output_prefix,
        "method": method_name,
        "B_total": B_total,
        "synthetic_output_path": syn_path,
        "num_transition_u_merged": len(merged_transition),
        "num_transition_u_prepared": len(prepared_transition_info),
        "num_transition_u_low_support_blended": prepared_low_support_count,
        "global_backoff_domain_size": len(global_backoff_dist),
        "use_global_backoff": use_global_backoff,
        "synthesis_splits": synthesis_splits,
        "length_mode": length_mode,
        "length_bucket_edges": bucket_edges,
        "support_blend_cfg": support_blend_cfg,
    })

    summary_path = os.path.join(synthetic_dir, f"{output_prefix}_test_synthetic_summary.json")
    save_json(summary_path, summary)

    return summary


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

    synthetic_num = int(exp_cfg.get("synthetic_num", 1000))
    random_seed = int(exp_cfg.get("random_seed", 42))
    use_global_backoff = bool(exp_cfg.get("use_global_backoff", True))
    methods = exp_cfg.get("methods", ["riskaware", "uniform"])

    length_mode = exp_cfg.get("length_mode", "num_segments_bucket6")
    bucket_edges = parse_length_bucket_edges(exp_cfg)
    synthesis_splits = exp_cfg.get("synthesis_splits", ["train", "valid"])
    if not isinstance(synthesis_splits, list) or len(synthesis_splits) == 0:
        raise ValueError("synthesis_splits must be a non-empty list.")
    synthesis_splits = [str(x) for x in synthesis_splits]

    min_segments = int(dataset_cfg.get("min_segments_per_traj", 2))
    max_segments = int(exp_cfg.get("L_max", 30))

    recovered_dir = resolve_path(dataset_cfg["recovered_dir"])
    synthetic_dir = resolve_path(dataset_cfg["synthetic_dir"])

    if not recovered_dir or not os.path.exists(recovered_dir):
        raise FileNotFoundError(f"recovered_dir not found: {recovered_dir}")

    os.makedirs(synthetic_dir, exist_ok=True)

   
    support_blend_cfg = {
        
        "support_blend_min_reports": int(exp_cfg.get("support_blend_min_reports", 5)),
        
        "support_blend_tau": float(exp_cfg.get("support_blend_tau", 5.0)),
     
        "support_blend_extreme_low_reports": int(exp_cfg.get("support_blend_extreme_low_reports", 2)),
        
        "support_blend_top1_threshold": float(exp_cfg.get("support_blend_top1_threshold", 0.95)),
       
        "support_blend_extreme_alpha_cap": float(exp_cfg.get("support_blend_extreme_alpha_cap", 0.35)),
        
        "support_blend_min_prob": float(exp_cfg.get("support_blend_min_prob", 1e-3)),
        
        "support_blend_local_topk": int(exp_cfg.get("support_blend_local_topk", 32)),
        
        "support_blend_global_topk": int(exp_cfg.get("support_blend_global_topk", 256)),
    }

    total_stages = 3

    t = log_stage(1, total_stages, "Preparing synthesis configuration ...")
    print(f"[Info] dataset_name = {dataset_name}")
    print(f"[Info] dataset_variant = {dataset_variant}")
    print(f"[Info] dataset_root = {dataset_cfg['dataset_root']}")
    print(f"[Info] exp_tag = {exp_tag}")
    print(f"[Info] B_total = {B_total}")
    print(f"[Info] synthetic_num = {synthetic_num}")
    print(f"[Info] methods = {methods}")
    print(f"[Info] use_global_backoff = {use_global_backoff}")
    print(f"[Info] synthesis_splits = {synthesis_splits}")
    print(f"[Info] length_mode = {length_mode}")
    print(f"[Info] length_bucket_edges = {bucket_edges}")
    print(f"[Info] support_blend_cfg = {support_blend_cfg}")
    log_done(t, "Synthesis configuration ready")

    rng = random.Random(random_seed)

    t = log_stage(2, total_stages, "Synthesizing trajectories for each method ...")
    all_summary = {
        "dataset_name": dataset_name,
        "dataset_variant": dataset_variant,
        "dataset_root": dataset_cfg["dataset_root"],
        "exp_tag": exp_tag,
        "version_info": version_info,
        "B_total": B_total,
        "synthetic_num": synthetic_num,
        "methods": {}
    }

    for method_name in methods:
        method_summary = synthesize_one_method(
            method_name=method_name,
            dataset_name=dataset_name,
            dataset_variant=dataset_variant,
            exp_tag=exp_tag,
            B_total=B_total,
            synthetic_num=synthetic_num,
            recovered_dir=recovered_dir,
            synthetic_dir=synthetic_dir,
            use_global_backoff=use_global_backoff,
            synthesis_splits=synthesis_splits,
            length_mode=length_mode,
            bucket_edges=bucket_edges,
            min_segments=min_segments,
            max_segments=max_segments,
            support_blend_cfg=support_blend_cfg,
            rng=rng,
            progress_every=args.progress_every,
        )
        all_summary["methods"][method_name] = method_summary

    log_done(t, "All synthetic trajectories generated")

    t = log_stage(3, total_stages, "Saving synthesis overview ...")
    overview_path = os.path.join(
        synthetic_dir,
        f"{dataset_variant}_{exp_tag}_synthetic_overview.json"
    )
    save_json(overview_path, all_summary)
    log_done(t, "Overview saved")

    print("=" * 100)
    print("[race_7_synthesize_trajectories] Done.")
    print(json.dumps(all_summary, indent=2, ensure_ascii=False))
    print("=" * 100)


if __name__ == "__main__":
    main()