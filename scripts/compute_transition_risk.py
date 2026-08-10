#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

"""
compute_transition_risk.py

Canonical TrajRACE Phase-3 risk/bucket construction.

Input
-----
Local SBS event records produced by:

    scripts/build_events.py

For every transition unit:
    1. compute endpoint-proximity risk;
    2. compute long-stay-association risk;
    3. compute low-out-degree risk;
    4. combine them into the composite risk score;
    5. map the score to a target bucket using fixed PUBLIC thresholds.

Important
---------
This stage is LOCAL/client-side preprocessing.

Raw risk scores, true successors, trajectory/SBS identifiers, and true target
buckets are NOT server-visible outputs.
"""

import argparse
import csv
import json
import os
import sys
import time
from typing import Any, Dict, Iterable, List, Optional


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


from trajrace.risk_utils import (
    SegmentDistanceOracle,
    compute_transition_risks_for_record,
    validate_risk_config,
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
# Generic I/O
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


def ensure_dir(
    path: str,
) -> None:

    os.makedirs(
        path,
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

        obj = yaml.safe_load(
            f
        )

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

        return json.load(
            f
        )


def save_json(
    path: str,
    obj: Any,
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


def save_csv(
    path: str,
    rows: List[
        Dict[str, Any]
    ],
    fieldnames: List[str],
) -> None:

    ensure_parent_dir(
        path
    )

    with open(
        path,
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(
                row
            )


# =============================================================================
# Logging
# =============================================================================

def fmt_sec(
    seconds: float,
) -> str:

    if seconds < 60:
        return f"{seconds:.1f}s"

    minutes = int(
        seconds // 60
    )

    remainder = (
        seconds
        - 60 * minutes
    )

    return (
        f"{minutes}m"
        f"{remainder:.1f}s"
    )


def log_stage(
    index: int,
    total: int,
    message: str,
) -> float:

    print(
        f"\n[Stage {index}/{total}] "
        f"{message}"
    )

    return time.perf_counter()


def log_done(
    start: float,
    message: str,
) -> None:

    elapsed = (
        time.perf_counter()
        - start
    )

    print(
        f"[Done] {message} | "
        f"elapsed={fmt_sec(elapsed)}"
    )


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Canonical TrajRACE Phase-3 "
            "transition-risk computation."
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
        default=250,
        help=(
            "Print progress every N SBS records."
        ),
    )

    parser.add_argument(
        "--distance_cache_max_entries",
        type=int,
        default=250000,
        help=(
            "Maximum number of LOCAL in-memory "
            "road-distance cache entries."
        ),
    )

    parser.add_argument(
        "--allow_skip_invalid",
        action="store_true",
        help=(
            "Skip invalid SBS records instead of failing. "
            "Canonical runs should normally leave this disabled."
        ),
    )

    return parser.parse_args()


# =============================================================================
# Canonical risk configuration
# =============================================================================

def extract_risk_cfg(
    exp_cfg: Dict[str, Any],
) -> Dict[str, Any]:

    raw_cfg = {
        "lambda_e": exp_cfg.get(
            "lambda_e"
        ),

        "lambda_s": exp_cfg.get(
            "lambda_s"
        ),

        "lambda_d": exp_cfg.get(
            "lambda_d"
        ),

        "sigma_e": exp_cfg.get(
            "sigma_e"
        ),

        "sigma_st": exp_cfg.get(
            "sigma_st"
        ),

        "delta_st_sec": exp_cfg.get(
            "delta_st_sec"
        ),

        "distance_mode": exp_cfg.get(
            "distance_mode",
            "road_network_shortest_path",
        ),

        "road_distance_definition": (
            exp_cfg.get(
                "road_distance_definition",
                "symmetric_shortest_segment_hops",
            )
        ),

        "K": exp_cfg.get(
            "K"
        ),

        "theta_list": exp_cfg.get(
            "theta_list"
        ),
    }

    return validate_risk_config(
        raw_cfg
    )


# =============================================================================
# Streaming summary accumulator
# =============================================================================

class RiskSummaryAccumulator:

    def __init__(
        self,
        K: int,
    ) -> None:

        self.K = int(
            K
        )

        self.num_event_records = 0
        self.num_failed_records = 0
        self.num_transition_items = 0

        self.num_sbs_with_stay_anchors = 0

        self.bucket_stats: Dict[
            int,
            Dict[str, float],
        ] = {
            bucket: {
                "count": 0.0,
                "sum_risk_score": 0.0,
                "sum_endpoint": 0.0,
                "sum_stay": 0.0,
                "sum_degree": 0.0,
                "sum_delta_t": 0.0,
            }
            for bucket in range(
                1,
                self.K + 1,
            )
        }

    def add_record(
        self,
        items: List[
            Dict[str, Any]
        ],
    ) -> None:

        self.num_event_records += 1

        if (
            items
            and int(
                items[0].get(
                    "num_stay_anchor_segments",
                    0,
                )
            ) > 0
        ):
            self.num_sbs_with_stay_anchors += 1

        for item in items:

            self.num_transition_items += 1

            bucket = int(
                item[
                    "target_bucket"
                ]
            )

            if bucket not in self.bucket_stats:
                raise RuntimeError(
                    f"Invalid target bucket: {bucket}"
                )

            stats = self.bucket_stats[
                bucket
            ]

            stats["count"] += 1

            stats[
                "sum_risk_score"
            ] += float(
                item[
                    "risk_score"
                ]
            )

            stats[
                "sum_endpoint"
            ] += float(
                item[
                    "risk_endpoint"
                ]
            )

            stats[
                "sum_stay"
            ] += float(
                item[
                    "risk_stay"
                ]
            )

            stats[
                "sum_degree"
            ] += float(
                item[
                    "risk_degree"
                ]
            )

            delta_t = item.get(
                "delta_t",
                None,
            )

            if delta_t is not None:

                stats[
                    "sum_delta_t"
                ] += float(
                    delta_t
                )

    def add_failure(
        self,
    ) -> None:

        self.num_failed_records += 1

    def finalize(
        self,
    ) -> Dict[str, Any]:

        bucket_summary: Dict[
            str,
            Any,
        ] = {}

        total = int(
            self.num_transition_items
        )

        for bucket in range(
            1,
            self.K + 1,
        ):

            raw = self.bucket_stats[
                bucket
            ]

            count = int(
                raw["count"]
            )

            denominator = max(
                1,
                count,
            )

            bucket_summary[
                str(bucket)
            ] = {
                "count": count,

                "fraction": (
                    float(
                        count / total
                    )
                    if total
                    else 0.0
                ),

                "avg_risk_score": float(
                    raw[
                        "sum_risk_score"
                    ]
                    / denominator
                ),

                "avg_risk_endpoint": float(
                    raw[
                        "sum_endpoint"
                    ]
                    / denominator
                ),

                "avg_risk_stay": float(
                    raw[
                        "sum_stay"
                    ]
                    / denominator
                ),

                "avg_risk_degree": float(
                    raw[
                        "sum_degree"
                    ]
                    / denominator
                ),

                "avg_delta_t": float(
                    raw[
                        "sum_delta_t"
                    ]
                    / denominator
                ),
            }

        return {
            "num_event_records": int(
                self.num_event_records
            ),

            "num_failed_records": int(
                self.num_failed_records
            ),

            "num_transition_risk_items": int(
                self.num_transition_items
            ),

            "num_sbs_with_stay_anchors": int(
                self.num_sbs_with_stay_anchors
            ),

            "fraction_sbs_with_stay_anchors": (
                float(
                    self.num_sbs_with_stay_anchors
                    / self.num_event_records
                )
                if self.num_event_records
                else 0.0
            ),

            "bucket_summary": (
                bucket_summary
            ),
        }


# =============================================================================
# One split
# =============================================================================

def process_split(
    input_path: str,
    output_path: str,
    split_name: str,

    successor_cache: Dict[
        str,
        List[str],
    ],

    risk_cfg: Dict[str, Any],

    distance_oracle: (
        SegmentDistanceOracle
    ),

    progress_every: int,

    allow_skip_invalid: bool,
) -> Dict[str, Any]:

    if not os.path.exists(
        input_path
    ):
        raise FileNotFoundError(
            f"Event file not found: {input_path}"
        )

    ensure_parent_dir(
        output_path
    )

    accumulator = (
        RiskSummaryAccumulator(
            K=risk_cfg["K"]
        )
    )

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

                items = (
                    compute_transition_risks_for_record(
                        event_record=record,
                        successor_cache=(
                            successor_cache
                        ),
                        risk_cfg=(
                            risk_cfg
                        ),
                        distance_oracle=(
                            distance_oracle
                        ),
                    )
                )

                expected = len(
                    record.get(
                        "transition_events",
                        [],
                    )
                )

                if len(items) != expected:
                    raise RuntimeError(
                        "Risk item count mismatch: "
                        f"expected={expected}, "
                        f"got={len(items)}"
                    )

                for item in items:

                    fout.write(
                        json.dumps(
                            item,
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

                accumulator.add_record(
                    items
                )

            except Exception as exc:

                accumulator.add_failure()

                if (
                    len(
                        error_examples
                    )
                    < 20
                ):
                    error_examples.append(
                        {
                            "record_index": (
                                record_index
                            ),
                            "traj_id": (
                                record.get(
                                    "traj_id"
                                )
                            ),
                            "sbs_id": (
                                record.get(
                                    "sbs_id"
                                )
                            ),
                            "error": repr(
                                exc
                            ),
                        }
                    )

                if not allow_skip_invalid:

                    raise RuntimeError(
                        "Canonical risk computation failed: "
                        f"split={split_name}, "
                        f"record_index={record_index}, "
                        f"sbs_id={record.get('sbs_id')}"
                    ) from exc

            if (
                progress_every > 0
                and record_index
                % progress_every
                == 0
            ):

                current = (
                    accumulator.finalize()
                )

                print(
                    f"[risk][{split_name}] "
                    f"SBSs={record_index:,}, "
                    f"risk-items="
                    f"{current['num_transition_risk_items']:,}, "
                    f"failed="
                    f"{current['num_failed_records']:,}"
                )

    summary = (
        accumulator.finalize()
    )

    summary[
        "split"
    ] = split_name

    summary[
        "input_path"
    ] = input_path

    summary[
        "output_path"
    ] = output_path

    summary[
        "error_examples"
    ] = error_examples

    print(
        f"[risk][{split_name}] DONE | "
        f"SBSs="
        f"{summary['num_event_records']:,}, "
        f"risk-items="
        f"{summary['num_transition_risk_items']:,}, "
        f"stay-SBS="
        f"{summary['fraction_sbs_with_stay_anchors']:.2%}, "
        f"failed="
        f"{summary['num_failed_records']:,}"
    )

    return summary


# =============================================================================
# Bucket CSV
# =============================================================================

def bucket_summary_to_rows(
    split_name: str,
    summary: Dict[str, Any],
) -> List[Dict[str, Any]]:

    rows: List[
        Dict[str, Any]
    ] = []

    for bucket, stats in (
        summary[
            "bucket_summary"
        ].items()
    ):

        rows.append(
            {
                "split": split_name,
                "target_bucket": bucket,

                "count": stats[
                    "count"
                ],

                "fraction": stats[
                    "fraction"
                ],

                "avg_risk_score": stats[
                    "avg_risk_score"
                ],

                "avg_risk_endpoint": stats[
                    "avg_risk_endpoint"
                ],

                "avg_risk_stay": stats[
                    "avg_risk_stay"
                ],

                "avg_risk_degree": stats[
                    "avg_risk_degree"
                ],

                "avg_delta_t": stats[
                    "avg_delta_t"
                ],
            }
        )

    return rows


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

    risk_cfg = extract_risk_cfg(
        exp_cfg
    )

    print("=" * 80)
    print(
        "[compute_transition_risk] "
        "Canonical Phase-3 configuration"
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
        json.dumps(
            {
                "risk_cfg": risk_cfg,
            },
            indent=2,
            ensure_ascii=False,
        )
    )

    print("=" * 80)

    # -------------------------------------------------------------------------
    # Input event files.
    # -------------------------------------------------------------------------

    event_paths = {
        "train": resolve_path(
            dataset_cfg[
                "event_train"
            ]
        ),

        "valid": resolve_path(
            dataset_cfg[
                "event_valid"
            ]
        ),

        "test": resolve_path(
            dataset_cfg[
                "event_test"
            ]
        ),
    }

    for split_name, path in (
        event_paths.items()
    ):

        if (
            path is None
            or not os.path.exists(
                path
            )
        ):
            raise FileNotFoundError(
                f"Missing Phase-2 event file "
                f"for {split_name}: {path}"
            )

    # -------------------------------------------------------------------------
    # Public topology.
    # -------------------------------------------------------------------------

    successor_cache_path = (
        resolve_path(
            dataset_cfg[
                "successor_cache_path"
            ]
        )
    )

    if (
        successor_cache_path
        is None
        or not os.path.exists(
            successor_cache_path
        )
    ):
        raise FileNotFoundError(
            "Public successor cache not found: "
            f"{successor_cache_path}"
        )

    total_stages = 5

    # =========================================================================
    # Stage 1: load public successor topology
    # =========================================================================

    start = log_stage(
        1,
        total_stages,
        "Loading PUBLIC legal-successor cache...",
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
        "[Info] public successor cache size = "
        f"{len(successor_cache):,}"
    )

    distance_oracle = (
        SegmentDistanceOracle(
            successor_cache=(
                successor_cache
            ),
            max_cache_entries=(
                args.distance_cache_max_entries
            ),
        )
    )

    log_done(
        start,
        "Public topology loaded",
    )

    # -------------------------------------------------------------------------
    # Risk output is experiment-specific because risk parameters/thresholds
    # belong to exp_main.yaml.
    # -------------------------------------------------------------------------

    experiment_root = resolve_path(
        exp_cfg[
            "experiment_root"
        ]
    )

    assert experiment_root is not None

    risk_dir = os.path.join(
        experiment_root,
        "risk",
    )

    ensure_dir(
        risk_dir
    )

    risk_paths = {
        "train": os.path.join(
            risk_dir,
            "train_risk.jsonl",
        ),

        "valid": os.path.join(
            risk_dir,
            "valid_risk.jsonl",
        ),

        "test": os.path.join(
            risk_dir,
            "test_risk.jsonl",
        ),
    }

    split_summaries: Dict[
        str,
        Any,
    ] = {}

    # =========================================================================
    # Stage 2: train
    # =========================================================================

    start = log_stage(
        2,
        total_stages,
        "Computing train transition risks...",
    )

    split_summaries[
        "train"
    ] = process_split(
        input_path=event_paths[
            "train"
        ],
        output_path=risk_paths[
            "train"
        ],
        split_name="train",

        successor_cache=(
            successor_cache
        ),

        risk_cfg=risk_cfg,

        distance_oracle=(
            distance_oracle
        ),

        progress_every=(
            args.progress_every
        ),

        allow_skip_invalid=(
            args.allow_skip_invalid
        ),
    )

    log_done(
        start,
        "Train risks completed",
    )

    # =========================================================================
    # Stage 3: valid
    # =========================================================================

    start = log_stage(
        3,
        total_stages,
        "Computing validation transition risks...",
    )

    split_summaries[
        "valid"
    ] = process_split(
        input_path=event_paths[
            "valid"
        ],
        output_path=risk_paths[
            "valid"
        ],
        split_name="valid",

        successor_cache=(
            successor_cache
        ),

        risk_cfg=risk_cfg,

        distance_oracle=(
            distance_oracle
        ),

        progress_every=(
            args.progress_every
        ),

        allow_skip_invalid=(
            args.allow_skip_invalid
        ),
    )

    log_done(
        start,
        "Validation risks completed",
    )

    # =========================================================================
    # Stage 4: test
    # =========================================================================

    start = log_stage(
        4,
        total_stages,
        "Computing test transition risks...",
    )

    split_summaries[
        "test"
    ] = process_split(
        input_path=event_paths[
            "test"
        ],
        output_path=risk_paths[
            "test"
        ],
        split_name="test",

        successor_cache=(
            successor_cache
        ),

        risk_cfg=risk_cfg,

        distance_oracle=(
            distance_oracle
        ),

        progress_every=(
            args.progress_every
        ),

        allow_skip_invalid=(
            args.allow_skip_invalid
        ),
    )

    log_done(
        start,
        "Test risks completed",
    )

    # =========================================================================
    # Stage 5: summary
    # =========================================================================

    start = log_stage(
        5,
        total_stages,
        "Saving canonical risk summaries...",
    )

    overall_summary = {
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

        "exp_tag": exp_cfg[
            "exp_tag"
        ],

        "experiment_root": (
            experiment_root
        ),

        "risk_output_dir": (
            risk_dir
        ),

        "risk_cfg": risk_cfg,

        "bucket_mapping_strategy": (
            "fixed_public_thresholds"
        ),

        "private_global_ranking_used": (
            False
        ),

        "private_quantile_threshold_used": (
            False
        ),

        "distance_oracle": (
            distance_oracle.summary()
        ),

        "splits": (
            split_summaries
        ),

        # Risk records remain local/private.
        "risk_records_are_server_visible": (
            False
        ),
    }

    summary_path = os.path.join(
        risk_dir,
        "risk_summary.json",
    )

    save_json(
        summary_path,
        overall_summary,
    )

    rows: List[
        Dict[str, Any]
    ] = []

    for split_name in (
        "train",
        "valid",
        "test",
    ):

        rows.extend(
            bucket_summary_to_rows(
                split_name=(
                    split_name
                ),
                summary=(
                    split_summaries[
                        split_name
                    ]
                ),
            )
        )

    csv_path = os.path.join(
        risk_dir,
        "bucket_summary.csv",
    )

    save_csv(
        path=csv_path,
        rows=rows,
        fieldnames=[
            "split",
            "target_bucket",
            "count",
            "fraction",
            "avg_risk_score",
            "avg_risk_endpoint",
            "avg_risk_stay",
            "avg_risk_degree",
            "avg_delta_t",
        ],
    )

    log_done(
        start,
        "Risk summaries written",
    )

    print("=" * 80)
    print(
        "[compute_transition_risk] "
        "PHASE 3 DONE"
    )

    print(
        f"Risk directory:\n"
        f"  {risk_dir}"
    )

    print(
        f"Risk summary:\n"
        f"  {summary_path}"
    )

    print(
        f"Bucket summary:\n"
        f"  {csv_path}"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()