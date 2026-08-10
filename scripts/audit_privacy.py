#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

"""
audit_privacy.py

Canonical TrajRACE Phase-4 privacy implementation audit.

This script is NOT a substitute for the mathematical proof.

It checks that the implementation satisfies the assumptions used by the
formal conditional event-level privacy argument.

Hard checks
-----------
1. Public topology admission already passed.
2. SBS transition coverage and legality already passed.
3. Risk buckets are based on fixed PUBLIC thresholds.
4. Start/count/successor domains are fixed PUBLIC domains.
5. No server-visible report contains private identifiers, true successors,
   risks, true buckets, execution buckets, or event order.
6. The SBS adaptive scheduler never exceeds B.
7. The future minimum cost is always reserved.
8. GRR uses the public N(u) domain without secret-dependent expansion.
9. Cross-bucket successor GRR is bounded by max_k eps_event[k].
10. Joint transition output (Y, K_noisy) is bounded by

        eps_bar = max_k eps_event[k] + eps_bucket.

Important scope statement
-------------------------
B is audited as an SBS REPORTING-SCHEDULE CAP.

This script does NOT assert arbitrary complete-SBS or complete-trajectory
B-LDP because:
    - U=current road segment is treated as public/unprotected context;
    - the execution epsilon may depend on private context;
    - the formal transition claim is conditional event-level LDP.
"""

import argparse
import itertools
import json
import math
import os
import sys
from collections import Counter
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Tuple


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


from trajrace.versioning_utils import (
    apply_versioning,
    pretty_version_summary,
)


DEFAULT_DATASET_CONFIG = "configs/dataset.yaml"
DEFAULT_EXP_CONFIG = "configs/exp_main.yaml"


# =============================================================================
# Generic I/O
# =============================================================================

def resolve_path(
    path_str: Optional[str],
) -> Optional[str]:

    if path_str is None:
        return None

    if os.path.isabs(path_str):
        return path_str

    return os.path.abspath(
        os.path.join(
            PROJECT_ROOT,
            path_str,
        )
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


def save_json(
    path: str,
    obj: Any,
) -> None:

    parent = os.path.dirname(
        path
    )

    if parent:
        ensure_dir(
            parent
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
) -> Iterator[
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

            yield obj


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Canonical TrajRACE Phase-4 privacy audit."
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

    return parser.parse_args()


# =============================================================================
# Privacy config
# =============================================================================

def extract_privacy_cfg(
    exp_cfg: Dict[str, Any],
) -> Dict[str, Any]:

    required = [
        "B_total",
        "eps_start",
        "eps_count",
        "eps_bucket",
        "eps_event_list",
        "K",
        "L_max",
    ]

    for key in required:

        if key not in exp_cfg:

            raise KeyError(
                f"Missing privacy config: {key}"
            )

    return {
        "B_total": float(
            exp_cfg[
                "B_total"
            ]
        ),

        "eps_start": float(
            exp_cfg[
                "eps_start"
            ]
        ),

        "eps_count": float(
            exp_cfg[
                "eps_count"
            ]
        ),

        "eps_bucket": float(
            exp_cfg[
                "eps_bucket"
            ]
        ),

        "eps_event_list": [
            float(x)
            for x in exp_cfg[
                "eps_event_list"
            ]
        ],

        "K": int(
            exp_cfg[
                "K"
            ]
        ),

        "L_max": int(
            exp_cfg[
                "L_max"
            ]
        ),

        "privacy_scope": str(
            exp_cfg.get(
                "privacy_scope",
                "",
            )
        ),

        "B_semantics": str(
            exp_cfg.get(
                "B_semantics",
                "",
            )
        ),
    }


def transition_cost(
    bucket: int,
    cfg: Dict[str, Any],
) -> float:

    return (
        cfg[
            "eps_event_list"
        ][
            bucket - 1
        ]
        + cfg[
            "eps_bucket"
        ]
    )


def feasible_exec_buckets(
    remaining: float,
    future_after: int,
    cfg: Dict[str, Any],
) -> List[int]:

    min_cost = (
        min(
            cfg[
                "eps_event_list"
            ]
        )
        + cfg[
            "eps_bucket"
        ]
    )

    result = []

    for k in range(
        1,
        cfg["K"] + 1,
    ):

        needed = (
            transition_cost(
                k,
                cfg,
            )
            + future_after
            * min_cost
        )

        if needed <= (
            remaining
            + 1e-12
        ):

            result.append(k)

    return result


def choose_exec_bucket(
    target: int,
    feasible: Sequence[int],
) -> int:

    feasible = sorted(
        set(
            int(x)
            for x in feasible
        )
    )

    if target in feasible:
        return int(target)

    no_weaker = [
        x
        for x in feasible
        if x <= target
    ]

    if not no_weaker:

        raise RuntimeError(
            "No feasible no-weaker bucket."
        )

    return int(
        max(
            no_weaker
        )
    )


# =============================================================================
# Risk-group iterator
# =============================================================================

def iter_risk_groups(
    risk_path: str,
) -> Iterator[
    Tuple[
        str,
        List[Dict[str, Any]],
    ]
]:

    current_id = None
    group = []

    for item in iter_jsonl(
        risk_path
    ):

        sbs_id = str(
            item[
                "sbs_id"
            ]
        )

        if current_id is None:
            current_id = sbs_id

        if sbs_id != current_id:

            yield (
                current_id,
                group,
            )

            current_id = sbs_id
            group = []

        group.append(
            item
        )

    if current_id is not None:

        yield (
            current_id,
            group,
        )


# =============================================================================
# Gate manager
# =============================================================================

class GateManager:

    def __init__(self) -> None:

        self.items = []
        self.num_failed = 0

    def add(
        self,
        name: str,
        passed: bool,
        detail: Any,
        hard: bool = True,
    ) -> None:

        item = {
            "name": name,
            "passed": bool(
                passed
            ),
            "hard": bool(
                hard
            ),
            "detail": detail,
        }

        self.items.append(
            item
        )

        label = (
            "PASS"
            if passed
            else (
                "FAIL"
                if hard
                else "WARN"
            )
        )

        print(
            f"[{label}] {name}: {detail}"
        )

        if (
            hard
            and not passed
        ):
            self.num_failed += 1


# =============================================================================
# GRR cross-epsilon numerical sanity
# =============================================================================

def max_grr_cross_epsilon_log_ratio(
    domain_size: int,
    eps_values: Sequence[float],
) -> float:
    """
    Numerically maximize the GRR probability ratio across:
        different truths,
        different epsilon choices,
        same fixed public domain.

    The analytic theorem gives:
        log ratio <= max(eps_values).
    """

    d = int(
        domain_size
    )

    if d <= 1:
        return 0.0

    maximum = 1.0

    for eps_a in eps_values:

        for eps_b in eps_values:

            da = (
                math.exp(
                    eps_a
                )
                + d
                - 1
            )

            db = (
                math.exp(
                    eps_b
                )
                + d
                - 1
            )

            # Numerator output equals numerator truth,
            # denominator output differs from denominator truth.
            r1 = (
                math.exp(
                    eps_a
                )
                * db
                / da
            )

            # Numerator mismatch, denominator match.
            r2 = (
                db
                / (
                    da
                    * math.exp(
                        eps_b
                    )
                )
            )

            # Same truth/output match under different eps.
            r3 = (
                math.exp(
                    eps_a
                    - eps_b
                )
                * db
                / da
            )

            # Both mismatch.
            r4 = (
                db
                / da
            )

            maximum = max(
                maximum,
                r1,
                r2,
                r3,
                r4,
            )

    return float(
        math.log(
            maximum
        )
    )


# =============================================================================
# Upstream audits
# =============================================================================

def audit_upstream(
    gates: GateManager,
    dataset_cfg: Dict[str, Any],
    exp_cfg: Dict[str, Any],
) -> Dict[str, Any]:

    preprocess_summary_path = (
        resolve_path(
            dataset_cfg[
                "preprocess_summary_path"
            ]
        )
    )

    event_summary_path = (
        resolve_path(
            dataset_cfg[
                "event_summary_path"
            ]
        )
    )

    assert preprocess_summary_path
    assert event_summary_path

    preprocess_summary = (
        load_json(
            preprocess_summary_path
        )
    )

    event_summary = (
        load_json(
            event_summary_path
        )
    )

    gates.add(
        "phase1_no_illegal_accepted_trajectory",
        (
            int(
                preprocess_summary.get(
                    "illegal_accepted_trajectories",
                    -1,
                )
            )
            == 0
        ),
        preprocess_summary.get(
            "illegal_accepted_trajectories"
        ),
    )

    all_event_splits_ok = True

    for split in [
        "train",
        "valid",
        "test",
    ]:

        stats = (
            event_summary[
                "splits"
            ][
                split
            ]
        )

        split_ok = (
            abs(
                float(
                    stats[
                        "transition_coverage_ratio"
                    ]
                )
                - 1.0
            )
            <= 1e-12
            and int(
                stats.get(
                    "num_illegal_transitions",
                    0,
                )
            )
            == 0
            and bool(
                stats.get(
                    "all_transitions_in_public_successor_domain",
                    True,
                )
            )
        )

        gates.add(
            f"phase2_{split}_coverage_and_legality",
            split_ok,
            {
                "coverage": stats[
                    "transition_coverage_ratio"
                ],
                "illegal": stats.get(
                    "num_illegal_transitions",
                    0,
                ),
            },
        )

        all_event_splits_ok = (
            all_event_splits_ok
            and split_ok
        )

    experiment_root = (
        resolve_path(
            exp_cfg[
                "experiment_root"
            ]
        )
    )

    assert experiment_root

    risk_summary_path = os.path.join(
        experiment_root,
        "risk",
        "risk_summary.json",
    )

    risk_summary = load_json(
        risk_summary_path
    )

    gates.add(
        "phase3_fixed_public_thresholds",
        (
            risk_summary.get(
                "bucket_mapping_strategy"
            )
            == "fixed_public_thresholds"
        ),
        risk_summary.get(
            "bucket_mapping_strategy"
        ),
    )

    gates.add(
        "phase3_no_private_global_ranking",
        (
            risk_summary.get(
                "private_global_ranking_used"
            )
            is False
        ),
        risk_summary.get(
            "private_global_ranking_used"
        ),
    )

    for split in [
        "train",
        "valid",
        "test",
    ]:

        failed = int(
            risk_summary[
                "splits"
            ][
                split
            ][
                "num_failed_records"
            ]
        )

        gates.add(
            f"phase3_{split}_no_failed_risk_records",
            failed == 0,
            failed,
        )

    return {
        "preprocess_summary_path": (
            preprocess_summary_path
        ),
        "event_summary_path": (
            event_summary_path
        ),
        "risk_summary_path": (
            risk_summary_path
        ),
    }


# =============================================================================
# Scheduler recomputation audit
# =============================================================================

def audit_scheduler_split(
    split: str,
    event_path: str,
    risk_path: str,
    successor_cache: Dict[
        str,
        List[str],
    ],
    cfg: Dict[str, Any],
) -> Dict[str, Any]:

    num_sbs = 0
    num_transitions = 0
    fallback_count = 0

    max_spend = 0.0
    min_margin = float(
        "inf"
    )

    used_domain_sizes: Set[
        int
    ] = set()

    for event_record, risk_group in (
        itertools.zip_longest(
            iter_jsonl(
                event_path
            ),
            iter_risk_groups(
                risk_path
            ),
        )
    ):

        if (
            event_record is None
            or risk_group is None
        ):

            raise RuntimeError(
                "Event/risk length mismatch."
            )

        risk_sbs_id, risk_items = (
            risk_group
        )

        sbs_id = str(
            event_record[
                "sbs_id"
            ]
        )

        if (
            risk_sbs_id
            != sbs_id
        ):

            raise RuntimeError(
                "Event/risk SBS ordering mismatch."
            )

        transitions = (
            event_record[
                "transition_events"
            ]
        )

        if (
            len(transitions)
            != len(risk_items)
        ):

            raise RuntimeError(
                "Transition/risk count mismatch."
            )

        num_sbs += 1

        remaining = (
            cfg[
                "B_total"
            ]
            - cfg[
                "eps_start"
            ]
            - cfg[
                "eps_count"
            ]
        )

        transition_spend = 0.0

        for local_idx, (
            transition,
            risk_item,
        ) in enumerate(
            zip(
                transitions,
                risk_items,
            ),
            start=1,
        ):

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

            if (
                str(
                    risk_item[
                        "u"
                    ]
                )
                != u
                or str(
                    risk_item[
                        "v"
                    ]
                )
                != v
            ):

                raise RuntimeError(
                    "Event/risk transition mismatch."
                )

            domain = (
                successor_cache.get(
                    u,
                    None,
                )
            )

            if (
                domain is None
                or v not in domain
            ):

                raise RuntimeError(
                    "True successor outside fixed PUBLIC N(u)."
                )

            used_domain_sizes.add(
                len(
                    domain
                )
            )

            target = int(
                risk_item.get(
                    "target_bucket",
                    risk_item.get(
                        "b_t"
                    ),
                )
            )

            future_after = (
                len(transitions)
                - local_idx
            )

            feasible = (
                feasible_exec_buckets(
                    remaining,
                    future_after,
                    cfg,
                )
            )

            if not feasible:

                raise RuntimeError(
                    "Empty feasible set."
                )

            execution = (
                choose_exec_bucket(
                    target,
                    feasible,
                )
            )

            if execution > target:

                raise RuntimeError(
                    "Execution bucket gives weaker protection "
                    "than target bucket."
                )

            if execution != target:
                fallback_count += 1

            cost = transition_cost(
                execution,
                cfg,
            )

            remaining -= cost
            transition_spend += cost

            if remaining < -1e-10:

                raise RuntimeError(
                    "Schedule cap exceeded."
                )

            num_transitions += 1

        total_spend = (
            cfg[
                "eps_start"
            ]
            + cfg[
                "eps_count"
            ]
            + transition_spend
        )

        max_spend = max(
            max_spend,
            total_spend,
        )

        min_margin = min(
            min_margin,
            cfg[
                "B_total"
            ]
            - total_spend,
        )

        if (
            total_spend
            > cfg[
                "B_total"
            ]
            + 1e-10
        ):

            raise RuntimeError(
                "Per-SBS schedule cap exceeded."
            )

    if min_margin == float(
        "inf"
    ):
        min_margin = 0.0

    return {
        "num_sbs": int(
            num_sbs
        ),

        "num_transitions": int(
            num_transitions
        ),

        "scheduler_fallback_count": int(
            fallback_count
        ),

        "scheduler_fallback_rate": (
            float(
                fallback_count
                / num_transitions
            )
            if num_transitions
            else 0.0
        ),

        "max_schedule_spend": float(
            max_spend
        ),

        "minimum_schedule_margin": float(
            min_margin
        ),

        "used_successor_domain_sizes": sorted(
            int(x)
            for x in used_domain_sizes
        ),
    }


# =============================================================================
# Server schema audit
# =============================================================================

def audit_server_report_file(
    path: str,
    successor_cache: Dict[
        str,
        List[str],
    ],
    L_max: int,
    K: int,
) -> Dict[str, Any]:

    counts = Counter()

    banned_fields = {
        "traj_id",
        "taxi_id",
        "sbs_id",
        "sbs_index",
        "num_sbs",
        "true_v",
        "v",
        "risk_score",
        "risk_endpoint",
        "risk_stay",
        "risk_degree",
        "target_bucket",
        "exec_bucket",
        "epsilon_event",
        "epsilon_bucket",
        "t",
        "global_pos",
        "delta_t",
        "feasible_buckets",
    }

    expected_schema = {
        "start": {
            "event_type",
            "x_noisy",
        },

        "count": {
            "event_type",
            "x_noisy",
        },

        "transition": {
            "event_type",
            "u",
            "y",
            "k_noisy",
        },
    }

    schema_violations = 0
    domain_violations = 0
    banned_field_violations = 0

    for report in iter_jsonl(
        path
    ):

        event_type = report.get(
            "event_type"
        )

        if event_type not in (
            expected_schema
        ):

            schema_violations += 1
            continue

        counts[
            event_type
        ] += 1

        if set(
            report.keys()
        ) != expected_schema[
            event_type
        ]:

            schema_violations += 1

        if (
            set(
                report.keys()
            )
            & banned_fields
        ):

            banned_field_violations += 1

        if event_type == "start":

            x = str(
                report[
                    "x_noisy"
                ]
            )

            if x not in successor_cache:
                domain_violations += 1

        elif event_type == "count":

            try:
                x = int(
                    report[
                        "x_noisy"
                    ]
                )

            except Exception:

                domain_violations += 1
                continue

            if not (
                1
                <= x
                <= L_max
            ):

                domain_violations += 1

        elif event_type == "transition":

            u = str(
                report[
                    "u"
                ]
            )

            y = str(
                report[
                    "y"
                ]
            )

            try:
                k_noisy = int(
                    report[
                        "k_noisy"
                    ]
                )

            except Exception:

                domain_violations += 1
                continue

            domain = (
                successor_cache.get(
                    u,
                    None,
                )
            )

            if (
                domain is None
                or y not in domain
            ):

                domain_violations += 1

            if not (
                1
                <= k_noisy
                <= K
            ):

                domain_violations += 1

    return {
        "counts": {
            str(k): int(v)
            for k, v
            in counts.items()
        },

        "schema_violations": int(
            schema_violations
        ),

        "domain_violations": int(
            domain_violations
        ),

        "banned_field_violations": int(
            banned_field_violations
        ),
    }


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

    assert dataset_config_path
    assert exp_config_path

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

    cfg = extract_privacy_cfg(
        exp_cfg
    )

    gates = GateManager()

    print("=" * 90)

    print(
        "[audit_privacy] "
        "Canonical Phase-4 privacy audit"
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

    print("=" * 90)

    # -------------------------------------------------------------------------
    # 1. Scope/config gates
    # -------------------------------------------------------------------------

    gates.add(
        "formal_privacy_scope",
        (
            cfg[
                "privacy_scope"
            ]
            == "conditional_event_level_successor"
        ),
        cfg[
            "privacy_scope"
        ],
    )

    gates.add(
        "B_semantics_is_schedule_cap",
        (
            cfg[
                "B_semantics"
            ]
            == "sbs_reporting_schedule_cap"
        ),
        cfg[
            "B_semantics"
        ],
    )

    gates.add(
        "public_start_domain_mode",
        (
            exp_cfg.get(
                "start_domain_mode"
            )
            == "public_road_segments"
        ),
        exp_cfg.get(
            "start_domain_mode"
        ),
    )

    gates.add(
        "public_count_domain_mode",
        (
            exp_cfg.get(
                "count_domain_mode"
            )
            == "public_exact_count"
        ),
        exp_cfg.get(
            "count_domain_mode"
        ),
    )

    gates.add(
        "public_successor_domain_mode",
        (
            exp_cfg.get(
                "successor_domain_mode"
            )
            == "public_legal_successors"
        ),
        exp_cfg.get(
            "successor_domain_mode"
        ),
    )

    gates.add(
        "fixed_public_risk_threshold_mode",
        (
            exp_cfg.get(
                "bucket_mapping_mode"
            )
            == "fixed_public_thresholds"
        ),
        exp_cfg.get(
            "bucket_mapping_mode"
        ),
    )

    # -------------------------------------------------------------------------
    # 2. Global budget feasibility
    # -------------------------------------------------------------------------

    min_transition_cost = (
        min(
            cfg[
                "eps_event_list"
            ]
        )
        + cfg[
            "eps_bucket"
        ]
    )

    minimum_schedule_required = (
        cfg[
            "eps_start"
        ]
        + cfg[
            "eps_count"
        ]
        + (
            cfg[
                "L_max"
            ]
            - 1
        )
        * min_transition_cost
    )

    gates.add(
        "global_schedule_feasibility",
        (
            minimum_schedule_required
            <= cfg[
                "B_total"
            ]
            + 1e-12
        ),
        {
            "minimum_required": (
                minimum_schedule_required
            ),
            "B": cfg[
                "B_total"
            ],
        },
    )

    # -------------------------------------------------------------------------
    # 3. Upstream gates
    # -------------------------------------------------------------------------

    upstream_paths = (
        audit_upstream(
            gates,
            dataset_cfg,
            exp_cfg,
        )
    )

    # -------------------------------------------------------------------------
    # 4. Load public topology
    # -------------------------------------------------------------------------

    successor_cache_path = (
        resolve_path(
            dataset_cfg[
                "successor_cache_path"
            ]
        )
    )

    assert successor_cache_path

    successor_cache = load_json(
        successor_cache_path
    )

    gates.add(
        "public_successor_cache_nonempty",
        (
            isinstance(
                successor_cache,
                dict,
            )
            and len(
                successor_cache
            )
            > 0
        ),
        len(
            successor_cache
        ),
    )

    # -------------------------------------------------------------------------
    # 5. Phase-4 metadata
    # -------------------------------------------------------------------------

    privatized_dir = resolve_path(
        exp_cfg[
            "privatized_dir"
        ]
    )

    assert privatized_dir

    meta_path = os.path.join(
        privatized_dir,
        "riskaware_meta.json",
    )

    summary_path = os.path.join(
        privatized_dir,
        "riskaware_summary.json",
    )

    if not os.path.exists(
        meta_path
    ):

        raise FileNotFoundError(
            f"Missing Phase-4 meta: {meta_path}"
        )

    meta = load_json(
        meta_path
    )

    phase4_summary = load_json(
        summary_path
    )

    gates.add(
        "no_secret_dependent_domain_expansion",
        (
            meta[
                "public_domains"
            ][
                "secret_dependent_domain_expansion"
            ]
            is False
        ),
        meta[
            "public_domains"
        ][
            "secret_dependent_domain_expansion"
        ],
    )

    gates.add(
        "no_full_sbs_B_ldp_overclaim",
        (
            meta.get(
                "formal_full_sbs_B_ldp_claim"
            )
            is False
        ),
        meta.get(
            "formal_full_sbs_B_ldp_claim"
        ),
    )

    # -------------------------------------------------------------------------
    # 6. Scheduler recomputation
    # -------------------------------------------------------------------------

    experiment_root = resolve_path(
        exp_cfg[
            "experiment_root"
        ]
    )

    assert experiment_root

    risk_dir = os.path.join(
        experiment_root,
        "risk",
    )

    schedule_audits = {}

    all_used_domain_sizes: Set[
        int
    ] = set()

    for split in [
        "train",
        "valid",
        "test",
    ]:

        event_path = resolve_path(
            dataset_cfg[
                f"event_{split}"
            ]
        )

        risk_path = os.path.join(
            risk_dir,
            f"{split}_risk.jsonl",
        )

        assert event_path

        result = (
            audit_scheduler_split(
                split=split,
                event_path=event_path,
                risk_path=risk_path,
                successor_cache=(
                    successor_cache
                ),
                cfg=cfg,
            )
        )

        schedule_audits[
            split
        ] = result

        all_used_domain_sizes.update(
            result[
                "used_successor_domain_sizes"
            ]
        )

        gates.add(
            f"{split}_SBS_schedule_cap",
            (
                result[
                    "max_schedule_spend"
                ]
                <= cfg[
                    "B_total"
                ]
                + 1e-10
            ),
            {
                "max_spend": (
                    result[
                        "max_schedule_spend"
                    ]
                ),
                "B": (
                    cfg[
                        "B_total"
                    ]
                ),
                "min_margin": (
                    result[
                        "minimum_schedule_margin"
                    ]
                ),
            },
        )

    # -------------------------------------------------------------------------
    # 7. Server-visible report schema
    # -------------------------------------------------------------------------

    report_audits = {}

    for split in [
        "train",
        "valid",
        "test",
    ]:

        report_path = os.path.join(
            privatized_dir,
            f"riskaware_{split}_reports.jsonl",
        )

        if not os.path.exists(
            report_path
        ):

            raise FileNotFoundError(
                f"Missing server report file: "
                f"{report_path}"
            )

        result = (
            audit_server_report_file(
                path=report_path,
                successor_cache=(
                    successor_cache
                ),
                L_max=(
                    cfg[
                        "L_max"
                    ]
                ),
                K=cfg[
                    "K"
                ],
            )
        )

        report_audits[
            split
        ] = result

        gates.add(
            f"{split}_server_schema",
            (
                result[
                    "schema_violations"
                ]
                == 0
            ),
            result[
                "schema_violations"
            ],
        )

        gates.add(
            f"{split}_server_public_domains",
            (
                result[
                    "domain_violations"
                ]
                == 0
            ),
            result[
                "domain_violations"
            ],
        )

        gates.add(
            f"{split}_server_no_private_fields",
            (
                result[
                    "banned_field_violations"
                ]
                == 0
            ),
            result[
                "banned_field_violations"
            ],
        )

        expected = (
            phase4_summary[
                "splits"
            ][
                split
            ]
        )

        actual_total = sum(
            result[
                "counts"
            ].values()
        )

        gates.add(
            f"{split}_server_report_count",
            (
                actual_total
                == int(
                    expected[
                        "expected_server_reports"
                    ]
                )
            ),
            {
                "actual": (
                    actual_total
                ),
                "expected": (
                    expected[
                        "expected_server_reports"
                    ]
                ),
            },
        )

    # -------------------------------------------------------------------------
    # 8. Formal cross-bucket privacy envelope
    # -------------------------------------------------------------------------

    max_event_epsilon = max(
        cfg[
            "eps_event_list"
        ]
    )

    numeric_max_successor = 0.0

    for domain_size in (
        all_used_domain_sizes
    ):

        numeric = (
            max_grr_cross_epsilon_log_ratio(
                domain_size=(
                    domain_size
                ),
                eps_values=(
                    cfg[
                        "eps_event_list"
                    ]
                ),
            )
        )

        numeric_max_successor = max(
            numeric_max_successor,
            numeric,
        )

    gates.add(
        "cross_bucket_successor_GRR_bound",
        (
            numeric_max_successor
            <= max_event_epsilon
            + 1e-10
        ),
        {
            "numeric_max_log_ratio": (
                numeric_max_successor
            ),
            "analytic_bound": (
                max_event_epsilon
            ),
        },
    )

    transition_privacy_envelope = (
        max_event_epsilon
        + cfg[
            "eps_bucket"
        ]
    )

    meta_envelope = float(
        meta[
            "formal_transition_guarantee"
        ][
            "epsilon_bar_transition"
        ]
    )

    gates.add(
        "joint_transition_privacy_envelope",
        (
            abs(
                meta_envelope
                - transition_privacy_envelope
            )
            <= 1e-12
        ),
        {
            "expected": (
                transition_privacy_envelope
            ),
            "meta": (
                meta_envelope
            ),
        },
    )

    # -------------------------------------------------------------------------
    # 9. Scope warning / no overclaim
    # -------------------------------------------------------------------------

    conditional_context_sequence_bound = (
        cfg[
            "eps_start"
        ]
        + cfg[
            "eps_count"
        ]
        + (
            cfg[
                "L_max"
            ]
            - 1
        )
        * transition_privacy_envelope
    )

    gates.add(
        "full_SBS_B_LDP_not_asserted",
        True,
        {
            "B": cfg[
                "B_total"
            ],

            "conditional_fixed_context_sequence_bound": (
                conditional_context_sequence_bound
            ),

            "reason": (
                "B is a reporting-schedule cap; "
                "U is public/unprotected and transition "
                "privacy is conditional event-level."
            ),
        },
        hard=False,
    )

    # -------------------------------------------------------------------------
    # 10. Optional local debug warning
    # -------------------------------------------------------------------------

    debug_files = []

    for split in [
        "train",
        "valid",
        "test",
    ]:

        debug_path = os.path.join(
            privatized_dir,
            f"riskaware_{split}_debug.jsonl",
        )

        if os.path.exists(
            debug_path
        ):

            debug_files.append(
                debug_path
            )

    if debug_files:

        gates.add(
            "private_debug_files_present",
            False,
            {
                "files": debug_files,
                "instruction": (
                    "These are LOCAL audit files only. "
                    "Do not treat them as server-visible reports "
                    "or publish them with main artifact outputs."
                ),
            },
            hard=False,
        )

    # -------------------------------------------------------------------------
    # 11. Save audit report
    # -------------------------------------------------------------------------

    audit_dir = resolve_path(
        exp_cfg[
            "audit_dir"
        ]
    )

    assert audit_dir

    ensure_dir(
        audit_dir
    )

    audit_report = {
        "dataset_name": (
            dataset_cfg[
                "dataset_name"
            ]
        ),

        "dataset_variant": (
            dataset_cfg[
                "dataset_variant"
            ]
        ),

        "exp_tag": (
            exp_cfg[
                "exp_tag"
            ]
        ),

        "formal_scope": {
            "privacy_scope": (
                cfg[
                    "privacy_scope"
                ]
            ),

            "public_context": (
                "U=current road segment"
            ),

            "protected_transition_value": (
                "V=true successor in fixed public N(U)"
            ),

            "epsilon_transition": (
                transition_privacy_envelope
            ),

            "B_semantics": (
                cfg[
                    "B_semantics"
                ]
            ),

            "full_sbs_B_ldp_claim": (
                False
            ),

            "full_trajectory_ldp_claim": (
                False
            ),
        },

        "schedule_audits": (
            schedule_audits
        ),

        "server_report_audits": (
            report_audits
        ),

        "upstream_paths": (
            upstream_paths
        ),

        "theorem_sanity": {
            "max_eps_event": (
                max_event_epsilon
            ),

            "eps_bucket": (
                cfg[
                    "eps_bucket"
                ]
            ),

            "numeric_cross_bucket_successor_log_ratio": (
                numeric_max_successor
            ),

            "transition_privacy_envelope": (
                transition_privacy_envelope
            ),

            "conditional_fixed_context_sequence_composition_bound": (
                conditional_context_sequence_bound
            ),
        },

        "gates": (
            gates.items
        ),

        "num_hard_failures": int(
            gates.num_failed
        ),

        "overall_status": (
            "PASS"
            if gates.num_failed
            == 0
            else "FAIL"
        ),
    }

    audit_path = os.path.join(
        audit_dir,
        "privacy_audit.json",
    )

    save_json(
        audit_path,
        audit_report,
    )

    print("=" * 90)

    print(
        f"[audit_privacy] "
        f"OVERALL STATUS = "
        f"{audit_report['overall_status']}"
    )

    print(
        f"Hard failures = "
        f"{gates.num_failed}"
    )

    print(
        f"Transition conditional "
        f"event-level epsilon_bar = "
        f"{transition_privacy_envelope:.6f}"
    )

    print(
        f"Audit report:\n"
        f"  {audit_path}"
    )

    print(
        "\nIMPORTANT:"
        "\n  PASS means that the implementation satisfies "
        "the audited assumptions of the stated conditional "
        "event-level privacy guarantee."
        "\n  It does NOT mean arbitrary SBS or complete "
        "trajectories satisfy B-LDP."
    )

    print("=" * 90)

    if gates.num_failed > 0:

        raise SystemExit(
            1
        )


if __name__ == "__main__":
    main()