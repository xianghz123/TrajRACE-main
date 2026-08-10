#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

"""
preprocess_dataset.py

Canonical TrajRACE Phase-1 preprocessing.

Pipeline
--------
Raw Porto GPS trajectories
    -> validation / reservoir sampling
    -> point timestamps
    -> fixed public road graph
    -> complete directed road-segment map matching
    -> public-topology legality admission
    -> train / valid / test

Critical invariants
-------------------
1. No L_max cropping is performed here.
2. A mapped trajectory must be COMPLETE and CONTIGUOUS.
3. Every consecutive road-segment pair must satisfy

       v in N(u)

   in the fixed PUBLIC successor graph.
4. If any local map-matching interval fails, the whole parent trajectory
   is rejected locally rather than concatenating disconnected fragments.
"""

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from collections import Counter
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Optional, Tuple


# =============================================================================
# Project root
# =============================================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from trajrace.versioning_utils import (
    apply_versioning,
    pretty_version_summary,
)


DEFAULT_DATASET_CONFIG = "configs/dataset.yaml"
DEFAULT_EXP_CONFIG = "configs/exp_main.yaml"


# =============================================================================
# Basic I/O
# =============================================================================

def resolve_path(path_str: Optional[str]) -> Optional[str]:
    if path_str is None:
        return None

    if os.path.isabs(path_str):
        return path_str

    return os.path.abspath(
        os.path.join(PROJECT_ROOT, path_str)
    )


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        ensure_dir(parent)


def load_yaml(path: str) -> Dict[str, Any]:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)

    if obj is None:
        return {}

    if not isinstance(obj, dict):
        raise ValueError(
            f"YAML root must be a mapping: {path}"
        )

    return obj


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, obj: Any) -> None:
    ensure_parent_dir(path)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            obj,
            f,
            ensure_ascii=False,
            indent=2,
        )


def write_jsonl(
    path: str,
    items: Iterable[Dict[str, Any]],
) -> None:
    ensure_parent_dir(path)

    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(
                json.dumps(
                    item,
                    ensure_ascii=False,
                )
                + "\n"
            )


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []

    if not os.path.exists(path):
        return results

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                results.append(
                    json.loads(line)
                )

    return results


def remove_if_exists(path: str) -> None:
    if os.path.exists(path):
        os.remove(path)


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Canonical TrajRACE raw preprocessing and "
            "topology-consistent map matching."
        )
    )

    parser.add_argument(
        "--dataset_config",
        type=str,
        default=DEFAULT_DATASET_CONFIG,
    )

    parser.add_argument(
        "--exp_config",
        type=str,
        default=DEFAULT_EXP_CONFIG,
    )

    parser.add_argument(
        "--progress_every",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--checkpoint_every",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--force_restart_map_matching",
        action="store_true",
        help=(
            "Delete the old map-matching checkpoint and rebuild "
            "mapped trajectories from scratch."
        ),
    )

    parser.add_argument(
        "--allow_download_osm",
        action="store_true",
    )

    return parser.parse_args()


# =============================================================================
# Logging
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

    minutes = int(sec // 60)
    seconds = sec - 60 * minutes

    return f"{minutes}m{seconds:.1f}s"


def log_stage(
    stage_idx: int,
    total_stages: int,
    message: str,
) -> float:
    print(
        f"\n[Stage {stage_idx}/{total_stages}] "
        f"{message}"
    )

    return time.perf_counter()


def log_done(
    start_time: float,
    message: str,
) -> None:
    elapsed = time.perf_counter() - start_time

    print(
        f"[Done] {message} | "
        f"elapsed={fmt_sec(elapsed)}"
    )


# =============================================================================
# Porto raw CSV
# =============================================================================

def _parse_optional_positive_int(
    value: Any,
) -> Optional[int]:
    if value is None:
        return None

    value = int(value)

    if value <= 0:
        return None

    return value


def parse_porto_row(
    row: Dict[str, Any],
    drop_missing_data: bool,
    min_points_per_traj: int,
    max_points_per_traj_before_map_matching: Optional[int],
) -> Optional[Dict[str, Any]]:

    missing = (
        str(row.get("MISSING_DATA", ""))
        .strip()
        .lower()
        == "true"
    )

    if drop_missing_data and missing:
        return None

    try:
        polyline = json.loads(
            row.get("POLYLINE", "[]")
        )
    except Exception:
        return None

    if not isinstance(polyline, list):
        return None

    if len(polyline) < min_points_per_traj:
        return None

    if (
        max_points_per_traj_before_map_matching is not None
        and len(polyline)
        > max_points_per_traj_before_map_matching
    ):
        return None

    points: List[List[float]] = []

    for point in polyline:
        if (
            not isinstance(point, list)
            or len(point) != 2
        ):
            return None

        try:
            lon = float(point[0])
            lat = float(point[1])
        except (TypeError, ValueError):
            return None

        points.append([lon, lat])

    try:
        start_timestamp = int(
            row.get("TIMESTAMP", "")
        )
    except Exception:
        return None

    return {
        "traj_id": str(
            row.get("TRIP_ID", "")
        ),
        "taxi_id": str(
            row.get("TAXI_ID", "")
        ),
        "start_timestamp": start_timestamp,
        "points": points,
    }


def reservoir_sample_porto_csv(
    csv_path: str,
    sample_size: int,
    seed: int,
    drop_missing_data: bool,
    min_points_per_traj: int,
    max_points_per_traj_before_map_matching: Optional[int],
    progress_every: int = 100000,
) -> Tuple[
    List[Dict[str, Any]],
    Dict[str, Any],
]:

    if sample_size <= 0:
        raise ValueError(
            "raw_sample_size must be > 0"
        )

    rng = random.Random(seed)

    reservoir: List[
        Dict[str, Any]
    ] = []

    total_rows = 0
    valid_rows = 0

    try:
        csv.field_size_limit(
            sys.maxsize
        )
    except OverflowError:
        csv.field_size_limit(
            2**31 - 1
        )

    with open(
        csv_path,
        "r",
        encoding="utf-8",
        newline="",
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:
            total_rows += 1

            item = parse_porto_row(
                row=row,
                drop_missing_data=drop_missing_data,
                min_points_per_traj=min_points_per_traj,
                max_points_per_traj_before_map_matching=(
                    max_points_per_traj_before_map_matching
                ),
            )

            if item is None:
                continue

            valid_rows += 1

            if len(reservoir) < sample_size:
                reservoir.append(item)

            else:
                j = rng.randint(
                    1,
                    valid_rows,
                )

                if j <= sample_size:
                    reservoir[j - 1] = item

            if (
                progress_every > 0
                and total_rows % progress_every == 0
            ):
                print(
                    "[Raw CSV] "
                    f"rows={total_rows:,}, "
                    f"valid={valid_rows:,}, "
                    f"sampled={len(reservoir):,}"
                )

    return reservoir, {
        "total_csv_rows": int(total_rows),
        "valid_rows_before_sampling": int(valid_rows),
        "requested_sample_size": int(sample_size),
        "actual_sample_size": int(len(reservoir)),
    }


# =============================================================================
# Timestamp / optional point downsampling
# =============================================================================

def assign_point_times(
    start_timestamp: int,
    num_points: int,
    sampling_interval_sec: int,
) -> List[int]:

    return [
        int(
            start_timestamp
            + i * sampling_interval_sec
        )
        for i in range(num_points)
    ]


def evenly_downsample_points_and_times(
    points: List[List[float]],
    point_times: List[int],
    max_points_after_downsample: Optional[int],
) -> Tuple[
    List[List[float]],
    List[int],
]:

    if (
        max_points_after_downsample is None
        or max_points_after_downsample <= 0
        or len(points)
        <= max_points_after_downsample
    ):
        return points, point_times

    target = int(
        max_points_after_downsample
    )

    if target < 2:
        raise ValueError(
            "max_points_after_downsample must be >= 2"
        )

    n = len(points)

    indices: List[int] = []

    for i in range(target):
        idx = round(
            i
            * (n - 1)
            / (target - 1)
        )

        indices.append(
            int(idx)
        )

    indices = sorted(
        set(indices)
    )

    return (
        [points[i] for i in indices],
        [point_times[i] for i in indices],
    )


# =============================================================================
# Public road graph
# =============================================================================

def try_import_graph_libs():
    try:
        import osmnx as ox
        import networkx as nx

        return ox, nx

    except Exception as exc:
        raise ImportError(
            "preprocess_dataset.py requires osmnx and networkx."
        ) from exc


def load_or_download_graph(
    graph_path: str,
    configured_download_if_missing: bool,
    allow_download_osm_cli: bool,
    place_name: str,
):
    ox, nx = try_import_graph_libs()

    ensure_parent_dir(
        graph_path
    )

    if os.path.exists(
        graph_path
    ):
        graph = ox.load_graphml(
            graph_path
        )

        return graph, ox, nx

    allow_download = (
        configured_download_if_missing
        or allow_download_osm_cli
    )

    if not allow_download:
        raise FileNotFoundError(
            f"Public road graph not found: {graph_path}"
        )

    graph = ox.graph_from_place(
        place_name,
        network_type="drive",
    )

    ox.save_graphml(
        graph,
        graph_path,
    )

    return graph, ox, nx


# =============================================================================
# Road-segment utilities
# =============================================================================

def make_segment_id(
    u: Any,
    v: Any,
    key: Any,
) -> str:

    return f"{u}__{v}__{key}"


def choose_best_parallel_edge_key(
    graph,
    u: Any,
    v: Any,
):
    edge_data = graph.get_edge_data(
        u,
        v,
        default=None,
    )

    if edge_data is None:
        return None

    best_key = None
    best_length = float("inf")

    for key, attrs in edge_data.items():

        try:
            length = float(
                attrs.get(
                    "length",
                    1.0,
                )
            )

        except Exception:
            length = 1.0

        if length < best_length:
            best_length = length
            best_key = key

    return best_key


def deduplicate_consecutive(
    items: List[Any],
    aligned_times: List[int],
) -> Tuple[
    List[Any],
    List[int],
]:

    if not items:
        return [], []

    new_items = [
        items[0]
    ]

    new_times = [
        aligned_times[0]
    ]

    for item, timestamp in zip(
        items[1:],
        aligned_times[1:],
    ):

        if item != new_items[-1]:

            new_items.append(
                item
            )

            new_times.append(
                timestamp
            )

    return new_items, new_times


# =============================================================================
# Topology legality
# =============================================================================

def is_legal_successor_sequence(
    segments: List[str],
    successor_cache: Dict[
        str,
        List[str],
    ],
) -> Tuple[
    bool,
    Optional[Dict[str, Any]],
]:
    """
    Hard public-topology admission test.

    For every consecutive pair:

        v must belong to N(u).

    Returns:
        (True, None)

    or:
        (False, diagnostic_dict)
    """
    if len(segments) < 2:
        return False, {
            "reason": "too_few_segments"
        }

    for idx, (
        u,
        v,
    ) in enumerate(
        zip(
            segments[:-1],
            segments[1:],
        ),
        start=1,
    ):

        u = str(u)
        v = str(v)

        public_successors = (
            successor_cache.get(
                u,
                [],
            )
        )

        if v not in public_successors:

            return False, {
                "reason": (
                    "illegal_successor_transition"
                ),
                "transition_position": int(
                    idx
                ),
                "u": u,
                "v": v,
            }

    return True, None


# =============================================================================
# Map matching
# =============================================================================

def match_points_to_route_segments(
    graph,
    ox,
    nx,
    points: List[List[float]],
    point_times: List[int],
    deduplicate_consecutive_segments: bool,
) -> Tuple[
    List[str],
    List[int],
    Optional[str],
]:
    """
    Map one GPS trajectory to one COMPLETE directed road-segment trajectory.

    Critical rule
    -------------
    Any failure in any matched-node interval rejects the WHOLE trajectory.

    We never:
        failure -> continue -> concatenate disconnected fragments.
    """

    if len(points) < 2:
        return [], [], "too_few_points"

    longitudes = [
        p[0]
        for p in points
    ]

    latitudes = [
        p[1]
        for p in points
    ]

    try:
        matched_nodes = list(
            ox.distance.nearest_nodes(
                graph,
                X=longitudes,
                Y=latitudes,
            )
        )

    except Exception:
        return [], [], "nearest_node_failure"

    if not matched_nodes:
        return [], [], "empty_nearest_node_result"

    # Consecutive identical matched nodes provide no road movement.
    node_seq = [
        matched_nodes[0]
    ]

    node_times = [
        point_times[0]
    ]

    for node, timestamp in zip(
        matched_nodes[1:],
        point_times[1:],
    ):

        if node != node_seq[-1]:

            node_seq.append(
                node
            )

            node_times.append(
                timestamp
            )

    if len(node_seq) < 2:
        return [], [], "too_few_distinct_matched_nodes"

    segments: List[str] = []
    segment_times: List[int] = []

    for i in range(
        len(node_seq) - 1
    ):

        src = node_seq[i]
        dst = node_seq[i + 1]

        t0 = int(
            node_times[i]
        )

        t1 = int(
            node_times[i + 1]
        )

        if src == dst:
            continue

        # Cheap safe shortcut.
        if graph.has_edge(
            src,
            dst,
        ):

            path_nodes = [
                src,
                dst,
            ]

        else:

            try:
                path_nodes = (
                    nx.shortest_path(
                        graph,
                        source=src,
                        target=dst,
                        weight="length",
                    )
                )

            except Exception:
                # IMPORTANT:
                # reject the entire trajectory.
                return (
                    [],
                    [],
                    "shortest_path_failure",
                )

        if len(path_nodes) < 2:
            return (
                [],
                [],
                "degenerate_shortest_path",
            )

        path_edges = [
            (
                path_nodes[j],
                path_nodes[j + 1],
            )
            for j in range(
                len(path_nodes) - 1
            )
        ]

        m = len(
            path_edges
        )

        dt = max(
            1,
            t1 - t0,
        )

        for j, (
            u,
            v,
        ) in enumerate(
            path_edges
        ):

            key = (
                choose_best_parallel_edge_key(
                    graph,
                    u,
                    v,
                )
            )

            if key is None:

                # IMPORTANT:
                # never skip one road edge and keep the remainder.
                return (
                    [],
                    [],
                    "missing_parallel_edge_key",
                )

            segment_id = (
                make_segment_id(
                    u,
                    v,
                    key,
                )
            )

            segment_time = int(
                round(
                    t0
                    + (
                        j
                        / max(1, m)
                    )
                    * dt
                )
            )

            segments.append(
                segment_id
            )

            segment_times.append(
                segment_time
            )

    if deduplicate_consecutive_segments:

        (
            segments,
            segment_times,
        ) = deduplicate_consecutive(
            segments,
            segment_times,
        )

    if len(segments) < 2:

        return (
            [],
            [],
            "too_few_segments_after_mapping",
        )

    return (
        segments,
        segment_times,
        None,
    )


# =============================================================================
# Public successor cache
# =============================================================================

def build_successor_cache(
    graph,
    progress_every: int = 10000,
) -> Dict[
    str,
    List[str],
]:

    edges = list(
        graph.edges(
            keys=True
        )
    )

    successor_cache: Dict[
        str,
        List[str],
    ] = {}

    total = len(edges)

    for idx, (
        u,
        v,
        key,
    ) in enumerate(
        edges,
        start=1,
    ):

        current_segment = (
            make_segment_id(
                u,
                v,
                key,
            )
        )

        successors = []

        for _, vv, kk in (
            graph.out_edges(
                v,
                keys=True,
            )
        ):

            successors.append(
                make_segment_id(
                    v,
                    vv,
                    kk,
                )
            )

        successor_cache[
            current_segment
        ] = sorted(
            set(successors)
        )

        if (
            progress_every > 0
            and (
                idx == 1
                or idx % progress_every == 0
                or idx == total
            )
        ):

            print(
                "[Successor Cache] "
                f"{idx:,}/{total:,}"
            )

    return successor_cache


# =============================================================================
# Split
# =============================================================================

def split_dataset(
    records: List[Dict[str, Any]],
    train_ratio: float,
    valid_ratio: float,
    test_ratio: float,
    split_seed: int,
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:

    total_ratio = (
        train_ratio
        + valid_ratio
        + test_ratio
    )

    if abs(
        total_ratio - 1.0
    ) > 1e-8:

        raise ValueError(
            "train_ratio + valid_ratio + test_ratio "
            "must equal 1.0"
        )

    records = list(
        records
    )

    rng = random.Random(
        split_seed
    )

    rng.shuffle(
        records
    )

    n = len(
        records
    )

    n_train = int(
        n * train_ratio
    )

    n_valid = int(
        n * valid_ratio
    )

    train = records[
        :n_train
    ]

    valid = records[
        n_train:
        n_train + n_valid
    ]

    test = records[
        n_train + n_valid:
    ]

    return (
        train,
        valid,
        test,
    )


# =============================================================================
# Statistics
# =============================================================================

def nearest_rank_percentile(
    values: List[int],
    percentile: float,
) -> float:

    if not values:
        return 0.0

    xs = sorted(
        values
    )

    rank = int(
        math.ceil(
            percentile
            / 100.0
            * len(xs)
        )
    )

    rank = max(
        1,
        min(
            rank,
            len(xs),
        ),
    )

    return float(
        xs[
            rank - 1
        ]
    )


def summarize_split(
    records: List[Dict[str, Any]],
    name: str,
    audit_L_max: Optional[int],
) -> Dict[str, Any]:

    lengths = [
        len(
            record[
                "segments"
            ]
        )
        for record in records
    ]

    if not lengths:

        return {
            "split": name,
            "num_records": 0,
            "avg_segments_per_traj": 0.0,
            "median_segments_per_traj": 0.0,
            "p95_segments_per_traj": 0.0,
            "max_segments_per_traj": 0,
            "fraction_segments_gt_Lmax": 0.0,
        }

    if audit_L_max is not None:

        num_long = sum(
            1
            for length in lengths
            if length > audit_L_max
        )

        fraction_long = (
            num_long
            / len(lengths)
        )

    else:
        num_long = 0
        fraction_long = 0.0

    return {
        "split": name,

        "num_records": int(
            len(records)
        ),

        "avg_segments_per_traj": float(
            mean(lengths)
        ),

        "median_segments_per_traj": float(
            median(lengths)
        ),

        "p95_segments_per_traj": float(
            nearest_rank_percentile(
                lengths,
                95.0,
            )
        ),

        "min_segments_per_traj": int(
            min(lengths)
        ),

        "max_segments_per_traj": int(
            max(lengths)
        ),

        "num_segments_gt_Lmax": int(
            num_long
        ),

        "fraction_segments_gt_Lmax": float(
            fraction_long
        ),
    }


# =============================================================================
# Checkpoint
# =============================================================================

def checkpoint_paths(
    cleaned_path: str,
) -> Dict[str, str]:

    intermediate_root = os.path.dirname(
        os.path.dirname(
            cleaned_path
        )
    )

    checkpoint_root = os.path.join(
        intermediate_root,
        "checkpoints",
        "preprocess_map_matching",
    )

    return {
        "root": checkpoint_root,

        "progress_json": os.path.join(
            checkpoint_root,
            "progress.json",
        ),

        "partial_jsonl": os.path.join(
            checkpoint_root,
            "mapped_sequences_partial.jsonl",
        ),
    }


def save_checkpoint_progress(
    path: str,
    total_count: int,
    processed_count: int,
    kept_count: int,
    failed_count: int,
    partial_jsonl: str,
    completed: bool,
    failure_reasons: Dict[str, int],
) -> None:

    save_json(
        path,
        {
            "total_count": int(
                total_count
            ),

            "processed_count": int(
                processed_count
            ),

            "kept_count": int(
                kept_count
            ),

            "failed_count": int(
                failed_count
            ),

            "failure_reasons": {
                str(k): int(v)
                for k, v
                in failure_reasons.items()
            },

            "partial_jsonl": (
                partial_jsonl
            ),

            "completed": bool(
                completed
            ),

            "updated_at": int(
                time.time()
            ),
        },
    )


# =============================================================================
# Main
# =============================================================================

def main() -> None:

    args = parse_args()

    dataset_config_path = resolve_path(
        args.dataset_config
    )

    exp_config_path = resolve_path(
        args.exp_config
    )

    if dataset_config_path is None:
        raise ValueError(
            "dataset config path is None"
        )

    if exp_config_path is None:
        raise ValueError(
            "experiment config path is None"
        )

    raw_dataset_cfg = load_yaml(
        dataset_config_path
    )

    raw_exp_cfg = load_yaml(
        exp_config_path
    )

    dataset_cfg, exp_cfg = (
        apply_versioning(
            raw_dataset_cfg,
            raw_exp_cfg,
        )
    )

    print("=" * 80)

    print(
        "[preprocess_dataset] "
        "Canonical configuration"
    )

    print(
        json.dumps(
            pretty_version_summary(
                raw_dataset_cfg,
                raw_exp_cfg,
            ),
            indent=2,
            ensure_ascii=False,
        )
    )

    print("=" * 80)

    raw_csv = resolve_path(
        str(
            dataset_cfg[
                "raw_csv"
            ]
        )
    )

    cleaned_path = resolve_path(
        str(
            dataset_cfg[
                "cleaned_path"
            ]
        )
    )

    segment_seq_train = resolve_path(
        str(
            dataset_cfg[
                "segment_seq_train"
            ]
        )
    )

    segment_seq_valid = resolve_path(
        str(
            dataset_cfg[
                "segment_seq_valid"
            ]
        )
    )

    segment_seq_test = resolve_path(
        str(
            dataset_cfg[
                "segment_seq_test"
            ]
        )
    )

    preprocess_summary_path = resolve_path(
        str(
            dataset_cfg[
                "preprocess_summary_path"
            ]
        )
    )

    road_graph_path = resolve_path(
        str(
            dataset_cfg[
                "road_graph_path"
            ]
        )
    )

    successor_cache_path = resolve_path(
        str(
            dataset_cfg[
                "successor_cache_path"
            ]
        )
    )

    distance_cache_path = resolve_path(
        str(
            dataset_cfg[
                "distance_cache_path"
            ]
        )
    )

    required_paths = [
        raw_csv,
        cleaned_path,
        segment_seq_train,
        segment_seq_valid,
        segment_seq_test,
        preprocess_summary_path,
        road_graph_path,
        successor_cache_path,
        distance_cache_path,
    ]

    if any(
        p is None
        for p in required_paths
    ):
        raise ValueError(
            "A required path resolved to None."
        )

    assert raw_csv is not None
    assert cleaned_path is not None
    assert segment_seq_train is not None
    assert segment_seq_valid is not None
    assert segment_seq_test is not None
    assert preprocess_summary_path is not None
    assert road_graph_path is not None
    assert successor_cache_path is not None
    assert distance_cache_path is not None

    if not os.path.exists(
        raw_csv
    ):
        raise FileNotFoundError(
            f"Raw Porto CSV not found: {raw_csv}"
        )

    raw_sample_size = int(
        dataset_cfg[
            "raw_sample_size"
        ]
    )

    split_seed = int(
        dataset_cfg.get(
            "split_seed",
            42,
        )
    )

    sampling_interval_sec = int(
        dataset_cfg.get(
            "sampling_interval_sec",
            15,
        )
    )

    drop_missing_data = bool(
        dataset_cfg.get(
            "drop_missing_data",
            True,
        )
    )

    min_points_per_traj = int(
        dataset_cfg.get(
            "min_points_per_traj",
            10,
        )
    )

    max_points_before_match = (
        _parse_optional_positive_int(
            dataset_cfg.get(
                "max_points_per_traj_before_map_matching",
                None,
            )
        )
    )

    enable_downsample = bool(
        dataset_cfg.get(
            "enable_point_downsample_before_map_matching",
            False,
        )
    )

    max_points_after_downsample = (
        _parse_optional_positive_int(
            dataset_cfg.get(
                "max_points_after_downsample_before_map_matching",
                None,
            )
        )
    )

    min_segments_per_traj = int(
        dataset_cfg.get(
            "min_segments_per_traj",
            2,
        )
    )

    preserve_full = bool(
        dataset_cfg.get(
            "preserve_full_mapped_trajectory",
            True,
        )
    )

    truncate_mapped = bool(
        dataset_cfg.get(
            "truncate_mapped_trajectory",
            False,
        )
    )

    if not preserve_full:
        raise ValueError(
            "preserve_full_mapped_trajectory "
            "must be true"
        )

    if truncate_mapped:
        raise ValueError(
            "truncate_mapped_trajectory "
            "must be false"
        )

    map_matching_method = str(
        dataset_cfg.get(
            "map_matching_method",
            "nearest_node_shortest_path",
        )
    )

    if (
        map_matching_method
        != "nearest_node_shortest_path"
    ):
        raise ValueError(
            "Canonical implementation requires "
            "map_matching_method="
            "'nearest_node_shortest_path'"
        )

    train_ratio = float(
        dataset_cfg.get(
            "train_ratio",
            0.8,
        )
    )

    valid_ratio = float(
        dataset_cfg.get(
            "valid_ratio",
            0.1,
        )
    )

    test_ratio = float(
        dataset_cfg.get(
            "test_ratio",
            0.1,
        )
    )

    deduplicate = bool(
        dataset_cfg.get(
            "deduplicate_consecutive_segments",
            True,
        )
    )

    download_if_missing = bool(
        dataset_cfg.get(
            "download_osm_if_missing",
            False,
        )
    )

    osm_place_name = str(
        dataset_cfg.get(
            "osm_place_name",
            "Porto, Portugal",
        )
    )

    audit_L_max = int(
        exp_cfg.get(
            "L_max",
            30,
        )
    )

    total_stages = 7

    # =========================================================================
    # Stage 1
    # =========================================================================

    t = log_stage(
        1,
        total_stages,
        "Scanning Porto CSV and reservoir-sampling valid trajectories...",
    )

    raw_items, raw_stats = (
        reservoir_sample_porto_csv(
            csv_path=raw_csv,
            sample_size=raw_sample_size,
            seed=split_seed,
            drop_missing_data=drop_missing_data,
            min_points_per_traj=min_points_per_traj,
            max_points_per_traj_before_map_matching=(
                max_points_before_match
            ),
        )
    )

    if not raw_items:
        raise RuntimeError(
            "No valid Porto trajectories sampled."
        )

    print(
        json.dumps(
            raw_stats,
            indent=2,
            ensure_ascii=False,
        )
    )

    log_done(
        t,
        "Raw CSV scan and sampling finished",
    )

    # =========================================================================
    # Stage 2
    # =========================================================================

    t = log_stage(
        2,
        total_stages,
        "Constructing point timestamps...",
    )

    cleaned_items: List[
        Dict[str, Any]
    ] = []

    for idx, record in enumerate(
        raw_items,
        start=1,
    ):

        point_times = assign_point_times(
            start_timestamp=record[
                "start_timestamp"
            ],
            num_points=len(
                record[
                    "points"
                ]
            ),
            sampling_interval_sec=(
                sampling_interval_sec
            ),
        )

        points = record[
            "points"
        ]

        if enable_downsample:

            points, point_times = (
                evenly_downsample_points_and_times(
                    points=points,
                    point_times=point_times,
                    max_points_after_downsample=(
                        max_points_after_downsample
                    ),
                )
            )

        cleaned_items.append(
            {
                "traj_id": record[
                    "traj_id"
                ],

                "taxi_id": record[
                    "taxi_id"
                ],

                "start_timestamp": record[
                    "start_timestamp"
                ],

                "points": points,

                "point_times": (
                    point_times
                ),
            }
        )

        if (
            args.progress_every > 0
            and idx
            % args.progress_every
            == 0
        ):

            print(
                "[Clean] "
                f"{idx:,}/{len(raw_items):,}"
            )

    write_jsonl(
        cleaned_path,
        cleaned_items,
    )

    log_done(
        t,
        "Point-level cleaned records written",
    )

    # =========================================================================
    # Stage 3
    # =========================================================================

    t = log_stage(
        3,
        total_stages,
        "Loading fixed public road graph...",
    )

    graph, ox, nx = (
        load_or_download_graph(
            graph_path=road_graph_path,
            configured_download_if_missing=(
                download_if_missing
            ),
            allow_download_osm_cli=(
                args.allow_download_osm
            ),
            place_name=osm_place_name,
        )
    )

    print(
        "[Info] public road graph: "
        f"nodes={len(graph.nodes()):,}, "
        f"edges={len(graph.edges()):,}"
    )

    log_done(
        t,
        "Public road graph ready",
    )

    # =========================================================================
    # Stage 4
    # =========================================================================

    t = log_stage(
        4,
        total_stages,
        "Preparing public topology caches...",
    )

    if os.path.exists(
        successor_cache_path
    ):

        successor_cache = load_json(
            successor_cache_path
        )

        print(
            "[Info] Loaded successor cache: "
            f"{len(successor_cache):,} segments"
        )

    else:

        successor_cache = (
            build_successor_cache(
                graph
            )
        )

        save_json(
            successor_cache_path,
            successor_cache,
        )

    if not os.path.exists(
        distance_cache_path
    ):
        save_json(
            distance_cache_path,
            {},
        )

    log_done(
        t,
        "Public topology caches ready",
    )

    # =========================================================================
    # Stage 5
    # =========================================================================

    t = log_stage(
        5,
        total_stages,
        "Map matching to COMPLETE topology-consistent trajectories...",
    )

    ckpt = checkpoint_paths(
        cleaned_path
    )

    ensure_parent_dir(
        ckpt[
            "progress_json"
        ]
    )

    if args.force_restart_map_matching:

        remove_if_exists(
            ckpt[
                "progress_json"
            ]
        )

        remove_if_exists(
            ckpt[
                "partial_jsonl"
            ]
        )

        print(
            "[Info] Previous map-matching "
            "checkpoint removed."
        )

    total_count = len(
        cleaned_items
    )

    processed_count = 0
    kept_count = 0
    failed_count = 0

    failure_reasons: Counter = Counter()

    if (
        os.path.exists(
            ckpt[
                "progress_json"
            ]
        )
        and os.path.exists(
            ckpt[
                "partial_jsonl"
            ]
        )
    ):

        progress = load_json(
            ckpt[
                "progress_json"
            ]
        )

        processed_count = int(
            progress.get(
                "processed_count",
                0,
            )
        )

        kept_count = int(
            progress.get(
                "kept_count",
                0,
            )
        )

        failed_count = int(
            progress.get(
                "failed_count",
                0,
            )
        )

        failure_reasons.update(
            progress.get(
                "failure_reasons",
                {},
            )
        )

        print(
            "[Resume] "
            f"processed={processed_count:,}, "
            f"kept={kept_count:,}, "
            f"failed={failed_count:,}"
        )

    else:

        remove_if_exists(
            ckpt[
                "partial_jsonl"
            ]
        )

        save_checkpoint_progress(
            path=ckpt[
                "progress_json"
            ],
            total_count=total_count,
            processed_count=0,
            kept_count=0,
            failed_count=0,
            partial_jsonl=ckpt[
                "partial_jsonl"
            ],
            completed=False,
            failure_reasons={},
        )

    tqdm = maybe_get_tqdm()

    output_mode = (
        "a"
        if processed_count > 0
        else "w"
    )

    with open(
        ckpt[
            "partial_jsonl"
        ],
        output_mode,
        encoding="utf-8",
    ) as fout:

        iterator = range(
            processed_count,
            total_count,
        )

        if tqdm is not None:

            iterator = tqdm(
                iterator,
                total=total_count,
                initial=processed_count,
                desc="Map Matching",
                ncols=120,
            )

        for absolute_index in iterator:

            record = cleaned_items[
                absolute_index
            ]

            (
                segments,
                segment_times,
                failure_reason,
            ) = match_points_to_route_segments(
                graph=graph,
                ox=ox,
                nx=nx,
                points=record[
                    "points"
                ],
                point_times=record[
                    "point_times"
                ],
                deduplicate_consecutive_segments=(
                    deduplicate
                ),
            )

            if (
                failure_reason is None
                and len(segments)
                >= min_segments_per_traj
            ):

                legal, diagnostic = (
                    is_legal_successor_sequence(
                        segments=segments,
                        successor_cache=(
                            successor_cache
                        ),
                    )
                )

                if not legal:

                    failure_reason = (
                        "topology_legality_failure"
                    )

                    if diagnostic:
                        failure_reasons[
                            str(
                                diagnostic.get(
                                    "reason",
                                    "unknown_legality_failure",
                                )
                            )
                        ] += 1

            elif failure_reason is None:

                failure_reason = (
                    "too_few_segments"
                )

            if failure_reason is None:

                output_record = {
                    "traj_id": record[
                        "traj_id"
                    ],

                    "taxi_id": record[
                        "taxi_id"
                    ],

                    "start_timestamp": record[
                        "start_timestamp"
                    ],

                    "segments": segments,

                    "segment_times": (
                        segment_times
                    ),
                }

                fout.write(
                    json.dumps(
                        output_record,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                kept_count += 1

            else:

                failed_count += 1

                failure_reasons[
                    failure_reason
                ] += 1

            processed_count += 1

            if (
                processed_count
                % max(
                    1,
                    args.checkpoint_every,
                )
                == 0
                or processed_count
                == total_count
            ):

                fout.flush()

                os.fsync(
                    fout.fileno()
                )

                save_checkpoint_progress(
                    path=ckpt[
                        "progress_json"
                    ],
                    total_count=total_count,
                    processed_count=(
                        processed_count
                    ),
                    kept_count=(
                        kept_count
                    ),
                    failed_count=(
                        failed_count
                    ),
                    partial_jsonl=ckpt[
                        "partial_jsonl"
                    ],
                    completed=False,
                    failure_reasons=dict(
                        failure_reasons
                    ),
                )

    save_checkpoint_progress(
        path=ckpt[
            "progress_json"
        ],
        total_count=total_count,
        processed_count=processed_count,
        kept_count=kept_count,
        failed_count=failed_count,
        partial_jsonl=ckpt[
            "partial_jsonl"
        ],
        completed=True,
        failure_reasons=dict(
            failure_reasons
        ),
    )

    sequence_records = read_jsonl(
        ckpt[
            "partial_jsonl"
        ]
    )

    if not sequence_records:
        raise RuntimeError(
            "No valid mapped trajectories produced."
        )

    # Final global topology assertion.
    final_illegal_count = 0

    for record in sequence_records:

        legal, diagnostic = (
            is_legal_successor_sequence(
                record[
                    "segments"
                ],
                successor_cache,
            )
        )

        if not legal:

            final_illegal_count += 1

            raise RuntimeError(
                "Internal Phase-1 topology invariant failed: "
                f"traj_id={record.get('traj_id')}, "
                f"diagnostic={diagnostic}"
            )

    print(
        "[Topology Gate] "
        f"accepted={len(sequence_records):,}, "
        f"illegal_accepted={final_illegal_count}"
    )

    print(
        "[Map Matching Failures] "
        + json.dumps(
            dict(
                failure_reasons
            ),
            ensure_ascii=False,
        )
    )

    log_done(
        t,
        "Complete topology-consistent map matching finished",
    )

    # =========================================================================
    # Stage 6
    # =========================================================================

    t = log_stage(
        6,
        total_stages,
        "Splitting mapped trajectories...",
    )

    (
        train_records,
        valid_records,
        test_records,
    ) = split_dataset(
        records=sequence_records,
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
        test_ratio=test_ratio,
        split_seed=split_seed,
    )

    write_jsonl(
        segment_seq_train,
        train_records,
    )

    write_jsonl(
        segment_seq_valid,
        valid_records,
    )

    write_jsonl(
        segment_seq_test,
        test_records,
    )

    log_done(
        t,
        "Dataset split written",
    )

    # =========================================================================
    # Stage 7
    # =========================================================================

    t = log_stage(
        7,
        total_stages,
        "Writing preprocessing audit summary...",
    )

    all_lengths = [
        len(
            record[
                "segments"
            ]
        )
        for record in sequence_records
    ]

    num_gt_Lmax = sum(
        1
        for length in all_lengths
        if length > audit_L_max
    )

    summary = {
        "project_root": PROJECT_ROOT,

        "dataset_config": (
            dataset_config_path
        ),

        "exp_config": (
            exp_config_path
        ),

        "dataset_name": dataset_cfg[
            "dataset_name"
        ],

        "dataset_variant": dataset_cfg[
            "dataset_variant"
        ],

        "dataset_root": dataset_cfg[
            "dataset_root"
        ],

        "raw_csv": raw_csv,

        "road_graph_path": (
            road_graph_path
        ),

        "successor_cache_path": (
            successor_cache_path
        ),

        "distance_cache_path": (
            distance_cache_path
        ),

        "raw_sampling": raw_stats,

        "map_matching_method": (
            map_matching_method
        ),

        "preserve_full_mapped_trajectory": True,

        "trajectory_cropping_applied": False,

        "topology_admission_rule": (
            "every consecutive segment v must belong to public N(u)"
        ),

        "illegal_accepted_trajectories": int(
            final_illegal_count
        ),

        "num_failed_map_matching_or_topology": int(
            failed_count
        ),

        "map_matching_failure_reasons": {
            str(k): int(v)
            for k, v
            in failure_reasons.items()
        },

        "num_valid_mapped_trajectories": int(
            len(sequence_records)
        ),

        "mapped_length_overall": {
            "avg": float(
                mean(all_lengths)
            ),

            "median": float(
                median(all_lengths)
            ),

            "p95": float(
                nearest_rank_percentile(
                    all_lengths,
                    95.0,
                )
            ),

            "min": int(
                min(all_lengths)
            ),

            "max": int(
                max(all_lengths)
            ),
        },

        "audit_reference_L_max": int(
            audit_L_max
        ),

        "num_mapped_trajectories_gt_Lmax": int(
            num_gt_Lmax
        ),

        "fraction_mapped_trajectories_gt_Lmax": float(
            num_gt_Lmax
            / len(sequence_records)
        ),

        "train": summarize_split(
            train_records,
            "train",
            audit_L_max,
        ),

        "valid": summarize_split(
            valid_records,
            "valid",
            audit_L_max,
        ),

        "test": summarize_split(
            test_records,
            "test",
            audit_L_max,
        ),

        "checkpoint_dir": ckpt[
            "root"
        ],
    }

    save_json(
        preprocess_summary_path,
        summary,
    )

    print("=" * 80)

    print(
        "[preprocess_dataset] PHASE 1 DONE"
    )

    print(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        )
    )

    print("=" * 80)

    log_done(
        t,
        "Preprocessing summary written",
    )


if __name__ == "__main__":
    main()