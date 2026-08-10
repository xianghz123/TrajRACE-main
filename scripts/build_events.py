#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

"""
build_events.py

Canonical TrajRACE Phase-2 SBS/event construction.

Input
-----
Complete topology-consistent map-matched trajectories from Phase 1.

Output
------
For every split:

    T
      -> transition-preserving SBS decomposition
      -> I(S_j) = {st_j, cnt_j, X_j}

Hard invariants
---------------
1. Every SBS contains at most L_max road segments.
2. Every parent transition appears exactly once after SBS decomposition.
3. Every transition satisfies v in PUBLIC N(u).
4. No invalid parent trajectory is silently accepted.
"""

import argparse
import json
import math
import os
import sys
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Optional, Tuple


# =============================================================================
# Project root
# =============================================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(
        0,
        PROJECT_ROOT,
    )


from trajrace.event_utils import (
    build_event_records_from_sequence,
)

from trajrace.versioning_utils import (
    apply_versioning,
    pretty_version_summary,
)


DEFAULT_DATASET_CONFIG = (
    "configs/dataset.yaml"
)

DEFAULT_EXP_CONFIG = (
    "configs/exp_main.yaml"
)


# =============================================================================
# I/O
# =============================================================================

def resolve_path(
    path_str: Optional[str],
) -> Optional[str]:

    if path_str is None:
        return None

    if os.path.isabs(
        path_str
    ):
        return path_str

    return os.path.abspath(
        os.path.join(
            PROJECT_ROOT,
            path_str,
        )
    )


def ensure_parent_dir(
    path: str,
) -> None:

    parent = os.path.dirname(
        path
    )

    if parent:
        os.makedirs(
            parent,
            exist_ok=True,
        )


def load_yaml(
    path: str,
) -> Dict[str, Any]:

    import yaml

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        obj = yaml.safe_load(f)

    if obj is None:
        return {}

    if not isinstance(
        obj,
        dict,
    ):
        raise ValueError(
            f"YAML root must be a mapping: {path}"
        )

    return obj


def load_json(
    path: str,
) -> Any:

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


def iter_jsonl(
    path: str,
) -> Iterable[
    Dict[str, Any]
]:

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        for line_no, line in enumerate(
            f,
            start=1,
        ):

            line = line.strip()

            if not line:
                continue

            try:
                obj = json.loads(
                    line
                )

            except json.JSONDecodeError as exc:

                raise ValueError(
                    f"Invalid JSON at "
                    f"{path}:{line_no}"
                ) from exc

            if not isinstance(
                obj,
                dict,
            ):
                raise ValueError(
                    f"Expected JSON object at "
                    f"{path}:{line_no}"
                )

            yield obj


def save_json(
    path: str,
    obj: Dict[str, Any],
) -> None:

    ensure_parent_dir(
        path
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            obj,
            f,
            ensure_ascii=False,
            indent=2,
        )


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


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Canonical TrajRACE Phase-2 SBS/event construction."
        )
    )

    parser.add_argument(
        "--dataset_config",
        type=str,
        default=(
            DEFAULT_DATASET_CONFIG
        ),
    )

    parser.add_argument(
        "--exp_config",
        type=str,
        default=(
            DEFAULT_EXP_CONFIG
        ),
    )

    parser.add_argument(
        "--progress_every",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--allow_skip_invalid",
        action="store_true",
        help=(
            "Debug only. Canonical runs should not silently skip "
            "invalid parent trajectories."
        ),
    )

    return parser.parse_args()


# =============================================================================
# Configuration validation
# =============================================================================

def validate_canonical_config(
    dataset_cfg: Dict[str, Any],
    exp_cfg: Dict[str, Any],
) -> None:

    if not bool(
        dataset_cfg.get(
            "preserve_full_mapped_trajectory",
            True,
        )
    ):

        raise ValueError(
            "preserve_full_mapped_trajectory "
            "must be true"
        )

    if bool(
        dataset_cfg.get(
            "truncate_mapped_trajectory",
            False,
        )
    ):

        raise ValueError(
            "truncate_mapped_trajectory "
            "must be false"
        )

    L_max = int(
        exp_cfg.get(
            "L_max",
            30,
        )
    )

    if L_max < 2:

        raise ValueError(
            "L_max must be >= 2"
        )

    partition_mode = str(
        exp_cfg.get(
            "sbs_partition_mode",
            "",
        )
    )

    if (
        partition_mode
        != "transition_preserving_overlap"
    ):

        raise ValueError(
            "Canonical TrajRACE requires "
            "sbs_partition_mode="
            "'transition_preserving_overlap'"
        )

    count_mode = str(
        exp_cfg.get(
            "count_mode",
            "",
        )
    )

    if (
        count_mode
        != "exact_num_segments"
    ):

        raise ValueError(
            "Canonical TrajRACE requires "
            "count_mode='exact_num_segments'"
        )

    count_domain_min = int(
        exp_cfg.get(
            "count_domain_min",
            1,
        )
    )

    count_domain_max = int(
        exp_cfg.get(
            "count_domain_max",
            L_max,
        )
    )

    if count_domain_min != 1:

        raise ValueError(
            "count_domain_min must equal 1"
        )

    if count_domain_max != L_max:

        raise ValueError(
            "count_domain_max must equal L_max"
        )


# =============================================================================
# Public topology validation
# =============================================================================

def validate_transition_legality(
    event_records: List[
        Dict[str, Any]
    ],
    successor_cache: Dict[
        str,
        List[str],
    ],
) -> Tuple[
    int,
    Optional[Dict[str, Any]],
]:
    """
    Verify that every transition in all SBSs satisfies

        v in N(u).

    Returns:
        (num_checked, None)

    Raises no exception here so caller can attach parent-trajectory metadata.
    """
    checked = 0

    for event_record in event_records:

        for transition in (
            event_record.get(
                "transition_events",
                [],
            )
        ):

            checked += 1

            u = str(
                transition[
                    "u"
                ]
            )

            v = str(
                transition[
                    "v"
                ]
            )

            public_successors = (
                successor_cache.get(
                    u,
                    [],
                )
            )

            if v not in public_successors:

                return checked, {
                    "traj_id": (
                        event_record.get(
                            "traj_id"
                        )
                    ),

                    "sbs_id": (
                        event_record.get(
                            "sbs_id"
                        )
                    ),

                    "t": (
                        transition.get(
                            "t"
                        )
                    ),

                    "u": u,
                    "v": v,

                    "public_domain_size": int(
                        len(
                            public_successors
                        )
                    ),
                }

    return checked, None


# =============================================================================
# Split processing
# =============================================================================

def process_split(
    input_path: str,
    output_path: str,
    split_name: str,
    L_max: int,
    partition_mode: str,
    successor_cache: Dict[
        str,
        List[str],
    ],
    progress_every: int,
    allow_skip_invalid: bool,
) -> Dict[str, Any]:

    if not os.path.exists(
        input_path
    ):

        raise FileNotFoundError(
            f"Input mapped-trajectory file not found: "
            f"{input_path}"
        )

    ensure_parent_dir(
        output_path
    )

    num_parent_trajectories = 0
    num_output_sbs = 0
    num_skipped = 0

    total_transition_events = 0
    expected_parent_transitions = 0

    num_legal_transition_checks = 0
    num_illegal_transitions = 0

    parent_segment_counts: List[
        int
    ] = []

    sbs_counts_per_parent: List[
        int
    ] = []

    sbs_segment_counts: List[
        int
    ] = []

    num_single_sbs = 0
    num_multi_sbs = 0

    error_examples: List[
        Dict[str, Any]
    ] = []

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as fout:

        for record_index, record in enumerate(
            iter_jsonl(
                input_path
            ),
            start=1,
        ):

            try:

                event_records = (
                    build_event_records_from_sequence(
                        seq_record=record,
                        L_max=L_max,
                        partition_mode=(
                            partition_mode
                        ),
                    )
                )

                if not event_records:

                    raise RuntimeError(
                        "No SBS event records generated."
                    )

                # -------------------------------------------------------------
                # NEW HARD GATE:
                # public-road legality before writing ANY SBS.
                # -------------------------------------------------------------

                (
                    checked,
                    illegal_example,
                ) = validate_transition_legality(
                    event_records=(
                        event_records
                    ),
                    successor_cache=(
                        successor_cache
                    ),
                )

                num_legal_transition_checks += int(
                    checked
                )

                if illegal_example is not None:

                    num_illegal_transitions += 1

                    raise RuntimeError(
                        "Illegal public-road transition detected: "
                        f"{illegal_example}"
                    )

            except Exception as exc:

                num_skipped += 1

                if len(
                    error_examples
                ) < 20:

                    error_examples.append(
                        {
                            "record_index": int(
                                record_index
                            ),

                            "traj_id": (
                                record.get(
                                    "traj_id"
                                )
                            ),

                            "error": repr(
                                exc
                            ),
                        }
                    )

                if allow_skip_invalid:
                    continue

                raise RuntimeError(
                    "Phase-2 canonical SBS/event construction failed: "
                    f"split={split_name}, "
                    f"record_index={record_index}, "
                    f"traj_id={record.get('traj_id')}"
                ) from exc

            num_parent_trajectories += 1

            parent_num_segments = int(
                event_records[
                    0
                ][
                    "parent_num_segments"
                ]
            )

            q = len(
                event_records
            )

            parent_segment_counts.append(
                parent_num_segments
            )

            sbs_counts_per_parent.append(
                q
            )

            expected_parent_transitions += max(
                0,
                parent_num_segments - 1,
            )

            if q == 1:
                num_single_sbs += 1
            else:
                num_multi_sbs += 1

            for event_record in (
                event_records
            ):

                cnt = int(
                    event_record[
                        "count_event"
                    ][
                        "cnt"
                    ]
                )

                if cnt > L_max:

                    raise RuntimeError(
                        "SBS length exceeds L_max: "
                        f"cnt={cnt}, L_max={L_max}"
                    )

                transitions = (
                    event_record[
                        "transition_events"
                    ]
                )

                if len(
                    transitions
                ) != max(
                    0,
                    cnt - 1,
                ):

                    raise RuntimeError(
                        "SBS transition-count invariant failed: "
                        f"cnt={cnt}, "
                        f"transitions={len(transitions)}"
                    )

                fout.write(
                    json.dumps(
                        event_record,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                num_output_sbs += 1

                sbs_segment_counts.append(
                    cnt
                )

                total_transition_events += len(
                    transitions
                )

            if (
                progress_every > 0
                and record_index
                % progress_every
                == 0
            ):

                print(
                    f"[build_events][{split_name}] "
                    f"parents={record_index:,}, "
                    f"SBSs={num_output_sbs:,}, "
                    f"illegal={num_illegal_transitions:,}, "
                    f"skipped={num_skipped:,}"
                )

    # -------------------------------------------------------------------------
    # Hard Gate 1: transition coverage
    # -------------------------------------------------------------------------

    if (
        total_transition_events
        != expected_parent_transitions
    ):

        raise RuntimeError(
            "Transition-coverage failure: "
            f"split={split_name}, "
            f"built={total_transition_events}, "
            f"expected={expected_parent_transitions}"
        )

    # -------------------------------------------------------------------------
    # Hard Gate 2: topology legality
    # -------------------------------------------------------------------------

    if num_illegal_transitions != 0:

        raise RuntimeError(
            "Topology-legality failure: "
            f"split={split_name}, "
            f"illegal={num_illegal_transitions}"
        )

    if num_parent_trajectories > 0:

        fraction_multi_sbs = (
            num_multi_sbs
            / num_parent_trajectories
        )

    else:
        fraction_multi_sbs = 0.0

    if expected_parent_transitions > 0:

        transition_coverage_ratio = (
            total_transition_events
            / expected_parent_transitions
        )

    else:
        transition_coverage_ratio = 0.0

    legal_transition_ratio = (
        1.0
        if total_transition_events > 0
        else 0.0
    )

    stats: Dict[str, Any] = {
        "split": split_name,

        "input_path": input_path,

        "output_path": output_path,

        "num_parent_trajectories": int(
            num_parent_trajectories
        ),

        "num_output_sbs": int(
            num_output_sbs
        ),

        "num_skipped_parent_trajectories": int(
            num_skipped
        ),

        "num_single_sbs_trajectories": int(
            num_single_sbs
        ),

        "num_multi_sbs_trajectories": int(
            num_multi_sbs
        ),

        "fraction_multi_sbs": float(
            fraction_multi_sbs
        ),

        "avg_sbs_per_trajectory": (
            float(
                mean(
                    sbs_counts_per_parent
                )
            )
            if sbs_counts_per_parent
            else 0.0
        ),

        "median_sbs_per_trajectory": (
            float(
                median(
                    sbs_counts_per_parent
                )
            )
            if sbs_counts_per_parent
            else 0.0
        ),

        "p95_sbs_per_trajectory": (
            nearest_rank_percentile(
                sbs_counts_per_parent,
                95.0,
            )
        ),

        "max_sbs_per_trajectory": (
            int(
                max(
                    sbs_counts_per_parent
                )
            )
            if sbs_counts_per_parent
            else 0
        ),

        "avg_parent_segments": (
            float(
                mean(
                    parent_segment_counts
                )
            )
            if parent_segment_counts
            else 0.0
        ),

        "min_parent_segments": (
            int(
                min(
                    parent_segment_counts
                )
            )
            if parent_segment_counts
            else 0
        ),

        "max_parent_segments": (
            int(
                max(
                    parent_segment_counts
                )
            )
            if parent_segment_counts
            else 0
        ),

        "avg_sbs_segments": (
            float(
                mean(
                    sbs_segment_counts
                )
            )
            if sbs_segment_counts
            else 0.0
        ),

        "min_sbs_segments": (
            int(
                min(
                    sbs_segment_counts
                )
            )
            if sbs_segment_counts
            else 0
        ),

        "max_sbs_segments": (
            int(
                max(
                    sbs_segment_counts
                )
            )
            if sbs_segment_counts
            else 0
        ),

        "num_start_events": int(
            num_output_sbs
        ),

        "num_count_events": int(
            num_output_sbs
        ),

        "num_transition_events": int(
            total_transition_events
        ),

        "expected_parent_transitions": int(
            expected_parent_transitions
        ),

        "transition_coverage_ratio": float(
            transition_coverage_ratio
        ),

        # NEW topology audit fields.
        "num_public_topology_checks": int(
            num_legal_transition_checks
        ),

        "num_illegal_transitions": int(
            num_illegal_transitions
        ),

        "legal_transition_ratio": float(
            legal_transition_ratio
        ),

        "all_transitions_in_public_successor_domain": True,

        "num_shared_boundary_segments": int(
            num_output_sbs
            - num_parent_trajectories
        ),

        "error_examples": (
            error_examples
        ),
    }

    print(
        f"[build_events][{split_name}] DONE | "
        f"parents={num_parent_trajectories:,}, "
        f"SBSs={num_output_sbs:,}, "
        f"multi-SBS={fraction_multi_sbs:.2%}, "
        f"max-SBS-len={stats['max_sbs_segments']}, "
        f"transitions={total_transition_events:,}, "
        f"coverage={transition_coverage_ratio:.6f}, "
        f"illegal=0"
    )

    return stats


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
            "dataset_config path is None"
        )

    if exp_config_path is None:
        raise ValueError(
            "exp_config path is None"
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

    validate_canonical_config(
        dataset_cfg,
        exp_cfg,
    )

    L_max = int(
        exp_cfg[
            "L_max"
        ]
    )

    partition_mode = str(
        exp_cfg[
            "sbs_partition_mode"
        ]
    )

    successor_cache_path = resolve_path(
        str(
            dataset_cfg[
                "successor_cache_path"
            ]
        )
    )

    if (
        successor_cache_path is None
        or not os.path.exists(
            successor_cache_path
        )
    ):

        raise FileNotFoundError(
            "Public successor cache not found: "
            f"{successor_cache_path}"
        )

    print("=" * 80)

    print(
        "[build_events] Canonical Phase-2 configuration"
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

    print(
        f"L_max               = {L_max}"
    )

    print(
        f"sbs_partition_mode  = {partition_mode}"
    )

    print(
        f"count_mode           = "
        f"{exp_cfg.get('count_mode')}"
    )

    print("=" * 80)

    print(
        "[build_events] Loading PUBLIC successor cache..."
    )

    successor_cache = load_json(
        successor_cache_path
    )

    if not isinstance(
        successor_cache,
        dict,
    ):

        raise ValueError(
            "successor cache must be a JSON object"
        )

    print(
        "[build_events] PUBLIC successor cache size = "
        f"{len(successor_cache):,}"
    )

    split_paths = {
        "train": (
            dataset_cfg[
                "segment_seq_train"
            ],
            dataset_cfg[
                "event_train"
            ],
        ),

        "valid": (
            dataset_cfg[
                "segment_seq_valid"
            ],
            dataset_cfg[
                "event_valid"
            ],
        ),

        "test": (
            dataset_cfg[
                "segment_seq_test"
            ],
            dataset_cfg[
                "event_test"
            ],
        ),
    }

    split_stats: Dict[
        str,
        Any,
    ] = {}

    for split_name, (
        input_path,
        output_path,
    ) in split_paths.items():

        input_path = resolve_path(
            str(input_path)
        )

        output_path = resolve_path(
            str(output_path)
        )

        assert input_path is not None
        assert output_path is not None

        split_stats[
            split_name
        ] = process_split(
            input_path=input_path,
            output_path=output_path,
            split_name=split_name,
            L_max=L_max,
            partition_mode=(
                partition_mode
            ),
            successor_cache=(
                successor_cache
            ),
            progress_every=(
                args.progress_every
            ),
            allow_skip_invalid=(
                args.allow_skip_invalid
            ),
        )

    summary = {
        "dataset_name": dataset_cfg[
            "dataset_name"
        ],

        "dataset_variant": dataset_cfg[
            "dataset_variant"
        ],

        "dataset_root": dataset_cfg[
            "dataset_root"
        ],

        "L_max": L_max,

        "sbs_partition_mode": (
            partition_mode
        ),

        "count_mode": exp_cfg.get(
            "count_mode"
        ),

        "public_successor_cache": (
            successor_cache_path
        ),

        "topology_hard_gate_enabled": True,

        "splits": (
            split_stats
        ),
    }

    summary_path = resolve_path(
        str(
            dataset_cfg[
                "event_summary_path"
            ]
        )
    )

    assert summary_path is not None

    save_json(
        summary_path,
        summary,
    )

    print("=" * 80)

    print(
        "[build_events] PHASE 2 DONE"
    )

    print(
        "[Hard Gate] "
        "transition coverage = PASS"
    )

    print(
        "[Hard Gate] "
        "public successor legality = PASS"
    )

    print(
        f"Event summary:\n  {summary_path}"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()