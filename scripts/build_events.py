#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import os
import sys
import json
import argparse
from collections import Counter
from typing import Any, Dict, List, Optional, Iterable, Tuple

# =============================================================================
# 项目根目录
# =============================================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DEFAULT_DATASET_CONFIG = "configs/dataset_porto_100k.yaml"
DEFAULT_EXP_CONFIG = "configs/exp_main_100k.yaml"

from trajrace.versioning_utils import apply_versioning, pretty_version_summary


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


def save_json(path: str, obj: Dict[str, Any]) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


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
        default=20,
        help="Fallback text progress interval when tqdm is unavailable."
    )
    return parser.parse_args()


def maybe_get_tqdm():
    try:
        from tqdm import tqdm
        return tqdm
    except Exception:
        return None


def iter_with_progress(
    data: List[Dict[str, Any]],
    split_name: str,
    progress_every: int = 20
) -> Iterable[Tuple[int, Dict[str, Any]]]:
    total = len(data)
    tqdm = maybe_get_tqdm()

    if tqdm is not None:
        for idx, item in enumerate(
            tqdm(data, total=total, desc=f"[race_2] {split_name}", ncols=100),
            start=1
        ):
            yield idx, item
    else:
        if total == 0:
            return
        step = max(1, progress_every)
        for idx, item in enumerate(data, start=1):
            if idx == 1 or idx % step == 0 or idx == total:
                pct = 100.0 * idx / total
                print(f"[race_2][{split_name}] {idx}/{total} ({pct:.1f}%)")
            yield idx, item


# =============================================================================
# 长度桶工具
# =============================================================================

def parse_length_bucket_edges(exp_cfg: Dict[str, Any]) -> List[int]:
    """
    默认 6 桶：
      1 -> [2, 5]
      2 -> [6, 10]
      3 -> [11, 15]
      4 -> [16, 20]
      5 -> [21, 25]
      6 -> [26, 30]
    """
    edges = exp_cfg.get("length_bucket_edges", [5, 10, 15, 20, 25])
    if not isinstance(edges, list) or len(edges) == 0:
        raise ValueError("length_bucket_edges must be a non-empty list.")

    parsed = []
    for x in edges:
        v = int(x)
        parsed.append(v)

    parsed = sorted(parsed)
    for i in range(len(parsed) - 1):
        if parsed[i] >= parsed[i + 1]:
            raise ValueError("length_bucket_edges must be strictly increasing.")
    return parsed


def bucketize_length(traj_len: int, bucket_edges: List[int]) -> int:
    L = int(traj_len)
    for idx, upper in enumerate(bucket_edges, start=1):
        if L <= upper:
            return idx
    return len(bucket_edges) + 1


# =============================================================================
# 事件构建
# =============================================================================

def build_events_for_one_record(
    rec: Dict[str, Any],
    length_mode: str = "num_segments",
    bucket_edges: Optional[List[int]] = None,
) -> Dict[str, Any]:
    traj_id = rec["traj_id"]
    taxi_id = rec.get("taxi_id", "")
    start_timestamp = rec.get("start_timestamp", None)
    segments = rec["segments"]
    segment_times = rec["segment_times"]

    if not isinstance(segments, list):
        raise ValueError(f"segments must be a list, got: {type(segments)}")
    if not isinstance(segment_times, list):
        raise ValueError(f"segment_times must be a list, got: {type(segment_times)}")
    if len(segments) != len(segment_times):
        raise ValueError(
            f"len(segments) != len(segment_times) for traj_id={traj_id}: "
            f"{len(segments)} vs {len(segment_times)}"
        )
    if len(segments) < 2:
        raise ValueError(f"trajectory {traj_id} has <2 segments, cannot build transitions.")

    traj_len = len(segments)

    start_event = {
        "e1": segments[0]
    }

    if length_mode == "num_segments":
        length_value = traj_len
    elif length_mode == "num_segments_bucket6":
        if not bucket_edges:
            raise ValueError("bucket_edges is required when length_mode=num_segments_bucket6")
        length_value = bucketize_length(traj_len, bucket_edges)
    else:
        raise ValueError(f"Unsupported length_mode: {length_mode}")

    length_event = {
        "L": int(length_value),
        "L_raw": int(traj_len),
        "length_mode": length_mode,
    }

    transition_events = []
    for i in range(traj_len - 1):
        u = segments[i]
        v = segments[i + 1]
        tau_u = segment_times[i]
        tau_v = segment_times[i + 1]

        try:
            delta_t = int(tau_v) - int(tau_u)
        except Exception:
            delta_t = 0

        if delta_t < 0:
            delta_t = 0

        transition_events.append({
            "u": u,
            "v": v,
            "pos": i + 1,
            "traj_len": traj_len,
            "tau_u": tau_u,
            "tau_v": tau_v,
            "delta_t": delta_t
        })

    out = {
        "traj_id": traj_id,
        "taxi_id": taxi_id,
        "start_timestamp": start_timestamp,
        "segments": segments,
        "segment_times": segment_times,
        "start_event": start_event,
        "length_event": length_event,
        "transition_events": transition_events
    }
    return out


def process_split(
    input_path: str,
    output_path: str,
    length_mode: str,
    bucket_edges: Optional[List[int]],
    split_name: str,
    progress_every: int = 20,
) -> Dict[str, Any]:
    data = read_jsonl(input_path)

    out_items = []
    skipped = 0
    total_transitions = 0
    lengths = []
    length_labels = []
    delta_t_values = []

    print(f"[race_2] Start processing split={split_name}, input={input_path}, num_records={len(data)}")

    for _, rec in iter_with_progress(data, split_name=split_name, progress_every=progress_every):
        try:
            event_rec = build_events_for_one_record(
                rec,
                length_mode=length_mode,
                bucket_edges=bucket_edges
            )
            out_items.append(event_rec)
            total_transitions += len(event_rec["transition_events"])
            lengths.append(int(event_rec["length_event"]["L_raw"]))
            length_labels.append(int(event_rec["length_event"]["L"]))

            for tr in event_rec["transition_events"]:
                delta_t_values.append(tr["delta_t"])

        except Exception:
            skipped += 1

    write_jsonl(output_path, out_items)

    label_counter = Counter(length_labels)

    stats = {
        "input_path": input_path,
        "output_path": output_path,
        "num_input_records": len(data),
        "num_output_records": len(out_items),
        "num_skipped_records": skipped,
        "num_total_transitions": total_transitions,
        "avg_segments_per_traj_raw": (sum(lengths) / len(lengths)) if lengths else 0.0,
        "min_segments_per_traj_raw": min(lengths) if lengths else 0,
        "max_segments_per_traj_raw": max(lengths) if lengths else 0,
        "avg_delta_t": (sum(delta_t_values) / len(delta_t_values)) if delta_t_values else 0.0,
        "max_delta_t": max(delta_t_values) if delta_t_values else 0,
        "length_mode": length_mode,
        "length_label_counts": {str(k): int(v) for k, v in sorted(label_counter.items())},
    }

    print(f"[race_2] Finished split={split_name}, output={output_path}, kept={len(out_items)}, skipped={skipped}")
    return stats


def main():
    args = parse_args()

    dataset_config_path = resolve_path(args.dataset_config)
    exp_config_path = resolve_path(args.exp_config)

    if not os.path.exists(dataset_config_path):
        raise FileNotFoundError(f"dataset config not found: {dataset_config_path}")
    if exp_config_path is not None and not os.path.exists(exp_config_path):
        raise FileNotFoundError(f"exp config not found: {exp_config_path}")

    raw_dataset_cfg = load_yaml_simple(dataset_config_path)
    raw_exp_cfg = load_yaml_simple(exp_config_path) if exp_config_path else {}

    dataset_cfg, exp_cfg = apply_versioning(raw_dataset_cfg, raw_exp_cfg)
    version_info = pretty_version_summary(raw_dataset_cfg, raw_exp_cfg)

    dataset_name = dataset_cfg.get("dataset_name", "unknown")
    dataset_variant = dataset_cfg["dataset_variant"]
    exp_tag = exp_cfg["exp_tag"]

    length_mode = exp_cfg.get("length_mode", "num_segments_bucket6")
    bucket_edges = parse_length_bucket_edges(exp_cfg)

    required_keys = [
        "segment_seq_train", "segment_seq_valid", "segment_seq_test",
        "event_train", "event_valid", "event_test"
    ]
    for k in required_keys:
        if k not in dataset_cfg:
            raise KeyError(f"Missing key in dataset config: {k}")

    seg_train_path = resolve_path(dataset_cfg["segment_seq_train"])
    seg_valid_path = resolve_path(dataset_cfg["segment_seq_valid"])
    seg_test_path = resolve_path(dataset_cfg["segment_seq_test"])

    event_train_path = resolve_path(dataset_cfg["event_train"])
    event_valid_path = resolve_path(dataset_cfg["event_valid"])
    event_test_path = resolve_path(dataset_cfg["event_test"])

    for p in [seg_train_path, seg_valid_path, seg_test_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"segment sequence file not found: {p}")

    print(f"[Info] dataset_name = {dataset_name}")
    print(f"[Info] dataset_variant = {dataset_variant}")
    print(f"[Info] dataset_root = {dataset_cfg['dataset_root']}")
    print(f"[Info] exp_tag = {exp_tag}")
    print(f"[Info] length_mode = {length_mode}")

    train_stats = process_split(
        input_path=seg_train_path,
        output_path=event_train_path,
        length_mode=length_mode,
        bucket_edges=bucket_edges,
        split_name="train",
        progress_every=args.progress_every,
    )
    valid_stats = process_split(
        input_path=seg_valid_path,
        output_path=event_valid_path,
        length_mode=length_mode,
        bucket_edges=bucket_edges,
        split_name="valid",
        progress_every=args.progress_every,
    )
    test_stats = process_split(
        input_path=seg_test_path,
        output_path=event_test_path,
        length_mode=length_mode,
        bucket_edges=bucket_edges,
        split_name="test",
        progress_every=args.progress_every,
    )

    summary = {
        "project_root": PROJECT_ROOT,
        "dataset_config": dataset_config_path,
        "exp_config": exp_config_path,
        "dataset_name": dataset_name,
        "dataset_variant": dataset_variant,
        "dataset_root": dataset_cfg["dataset_root"],
        "exp_tag": exp_tag,
        "version_info": version_info,
        "length_mode": length_mode,
        "length_bucket_edges": bucket_edges,
        "train": train_stats,
        "valid": valid_stats,
        "test": test_stats
    }

    summary_path = os.path.join(
        os.path.dirname(event_train_path),
        f"{dataset_variant}_{exp_tag}_event_summary.json"
    )
    save_json(summary_path, summary)

    print("=" * 80)
    print("[race_2_build_events] Done.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("=" * 80)


if __name__ == "__main__":
    main()