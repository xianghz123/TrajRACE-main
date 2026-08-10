#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

"""
privatize_riskaware.py

Canonical TrajRACE Phase-4 risk-aware local privatization.

Inputs
------
1. Phase-2 SBS event records.
2. Phase-3 local transition-risk records.
3. Fixed PUBLIC road-successor topology.
4. Canonical experiment/privacy configuration.

Server-visible outputs
----------------------
Start report:
    {
        "event_type": "start",
        "x_noisy": ...
    }

Count report:
    {
        "event_type": "count",
        "x_noisy": ...
    }

Transition report:
    {
        "event_type": "transition",
        "u": ...,
        "y": ...,
        "k_noisy": ...
    }

The server reports contain NO:
    traj_id
    taxi_id
    sbs_id
    true successor
    true risk
    true bucket
    execution bucket
    event order

Formal scope
------------
For fixed PUBLIC current segment U=u, the protected transition value
is the true successor V in the fixed public domain N(u).

Even when the execution bucket is selected adaptively from private
context, the complete transition output (Y, K_noisy) admits the
cross-bucket conditional event-level privacy envelope

    eps_bar_x = max_k eps_event[k] + eps_bucket.

B is used as an SBS adaptive-reporting schedule cap. This script does
NOT claim arbitrary SBS-level or trajectory-level B-LDP.
"""

import argparse
import heapq
import itertools
import json
import math
import os
import random
import shutil
import sys
import tempfile
import time

from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


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


def ensure_parent_dir(
    path: str,
) -> None:

    parent = os.path.dirname(path)

    if parent:
        ensure_dir(parent)


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

    if not isinstance(obj, dict):
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

    ensure_parent_dir(path)

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
                obj = json.loads(line)

            except json.JSONDecodeError as exc:

                raise ValueError(
                    f"Invalid JSON at {path}:{line_no}"
                ) from exc

            if not isinstance(obj, dict):
                raise ValueError(
                    f"Expected JSON object at {path}:{line_no}"
                )

            yield obj


# =============================================================================
# Logging
# =============================================================================

def fmt_sec(
    sec: float,
) -> str:

    if sec < 60:
        return f"{sec:.1f}s"

    minutes = int(sec // 60)
    seconds = sec - 60 * minutes

    return (
        f"{minutes}m"
        f"{seconds:.1f}s"
    )


def log_stage(
    idx: int,
    total: int,
    message: str,
) -> float:

    print(
        f"\n[Stage {idx}/{total}] "
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
            "Canonical TrajRACE Phase-4 "
            "risk-aware local privatization."
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
        default=250,
    )

    parser.add_argument(
        "--shuffle_chunk_size",
        type=int,
        default=200000,
        help=(
            "Number of server reports held in memory "
            "before flushing one external-shuffle chunk."
        ),
    )

    parser.add_argument(
        "--save_debug",
        action="store_true",
        help=(
            "Save LOCAL private debug traces for smoke/audit runs. "
            "Do not publish these files as server-visible outputs."
        ),
    )

    return parser.parse_args()


# =============================================================================
# Privacy configuration
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
        "random_seed",
    ]

    for key in required:

        if key not in exp_cfg:

            raise KeyError(
                f"Missing canonical privacy key: {key}"
            )

    eps_event_list = [
        float(x)
        for x in exp_cfg[
            "eps_event_list"
        ]
    ]

    K = int(
        exp_cfg["K"]
    )

    if len(
        eps_event_list
    ) != K:

        raise ValueError(
            "eps_event_list length must equal K"
        )

    for idx in range(
        len(eps_event_list) - 1
    ):

        if not (
            eps_event_list[idx]
            < eps_event_list[idx + 1]
        ):

            raise ValueError(
                "eps_event_list must be strictly increasing: "
                "bucket 1 must provide the strongest protection."
            )

    cfg = {
        "B_total": float(
            exp_cfg["B_total"]
        ),

        "eps_start": float(
            exp_cfg["eps_start"]
        ),

        "eps_count": float(
            exp_cfg["eps_count"]
        ),

        "eps_bucket": float(
            exp_cfg["eps_bucket"]
        ),

        "eps_event_list": (
            eps_event_list
        ),

        "K": K,

        "L_max": int(
            exp_cfg["L_max"]
        ),

        "random_seed": int(
            exp_cfg["random_seed"]
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

        "reserve_future_minimum_cost": bool(
            exp_cfg.get(
                "reserve_future_minimum_cost",
                True,
            )
        ),

        "require_budget_feasibility_check": bool(
            exp_cfg.get(
                "require_budget_feasibility_check",
                True,
            )
        ),

        "infeasible_target_policy": str(
            exp_cfg.get(
                "infeasible_target_policy",
                "",
            )
        ),

        "shuffle_transition_reports": bool(
            exp_cfg.get(
                "shuffle_transition_reports",
                True,
            )
        ),

        "save_debug_files": bool(
            exp_cfg.get(
                "save_debug_files",
                False,
            )
        ),
    }

    return cfg


def validate_canonical_privacy_cfg(
    exp_cfg: Dict[str, Any],
    cfg: Dict[str, Any],
) -> None:

    if (
        cfg["privacy_scope"]
        != "conditional_event_level_successor"
    ):
        raise ValueError(
            "Canonical privacy_scope must be "
            "'conditional_event_level_successor'."
        )

    if (
        cfg["B_semantics"]
        != "sbs_reporting_schedule_cap"
    ):
        raise ValueError(
            "B_semantics must be "
            "'sbs_reporting_schedule_cap'."
        )

    if not cfg[
        "reserve_future_minimum_cost"
    ]:
        raise ValueError(
            "reserve_future_minimum_cost must be true."
        )

    if (
        cfg[
            "infeasible_target_policy"
        ]
        != "closest_feasible_no_weaker"
    ):
        raise ValueError(
            "Canonical infeasible_target_policy must be "
            "'closest_feasible_no_weaker'."
        )

    if not cfg[
        "shuffle_transition_reports"
    ]:
        raise ValueError(
            "Canonical server reports must be shuffled."
        )

    expected_modes = {
        "start_domain_mode": (
            "public_road_segments"
        ),

        "count_domain_mode": (
            "public_exact_count"
        ),

        "successor_domain_mode": (
            "public_legal_successors"
        ),

        "bucket_mapping_mode": (
            "fixed_public_thresholds"
        ),

        "start_mechanism": "grr",

        "count_mechanism": "grr",

        "successor_mechanism": (
            "conditional_grr"
        ),

        "bucket_report_mechanism": (
            "grr"
        ),
    }

    for key, expected in (
        expected_modes.items()
    ):

        actual = exp_cfg.get(
            key,
            None,
        )

        if actual != expected:

            raise ValueError(
                f"{key} must be '{expected}', "
                f"got '{actual}'"
            )

    configured_fields = (
        exp_cfg.get(
            "server_transition_fields",
            None,
        )
    )

    if configured_fields is not None:

        if set(
            configured_fields
        ) != {
            "u",
            "y",
            "k_noisy",
        }:

            raise ValueError(
                "server_transition_fields must be "
                "[u, y, k_noisy]."
            )


def transition_cost(
    bucket: int,
    cfg: Dict[str, Any],
) -> float:

    return (
        float(
            cfg[
                "eps_event_list"
            ][bucket - 1]
        )
        + float(
            cfg[
                "eps_bucket"
            ]
        )
    )


def validate_budget_feasibility(
    cfg: Dict[str, Any],
) -> Dict[str, float]:

    min_transition_cost = (
        min(
            cfg["eps_event_list"]
        )
        + cfg["eps_bucket"]
    )

    max_transition_cost = (
        max(
            cfg["eps_event_list"]
        )
        + cfg["eps_bucket"]
    )

    max_num_transitions = (
        cfg["L_max"]
        - 1
    )

    minimum_required = (
        cfg["eps_start"]
        + cfg["eps_count"]
        + max_num_transitions
        * min_transition_cost
    )

    if (
        cfg[
            "require_budget_feasibility_check"
        ]
        and cfg["B_total"]
        + 1e-12
        < minimum_required
    ):

        raise ValueError(
            "Budget configuration is infeasible:\n"
            f"B={cfg['B_total']} < "
            f"eps_start + eps_count + "
            f"(L_max-1)*min_transition_cost "
            f"= {minimum_required}"
        )

    return {
        "min_transition_cost": float(
            min_transition_cost
        ),

        "max_transition_cost": float(
            max_transition_cost
        ),

        "minimum_required_schedule_cap": float(
            minimum_required
        ),

        "transition_privacy_envelope": float(
            max_transition_cost
        ),
    }


# =============================================================================
# Fixed-domain GRR
# =============================================================================

def grr_sample_fixed_domain(
    true_value: Any,
    domain: Sequence[Any],
    epsilon: float,
    rng: random.Random,
    true_is_known_valid: bool = False,
) -> Any:
    """
    Generalized randomized response over a FIXED domain.

    IMPORTANT:
    The domain is NEVER expanded using the private true value.
    """

    d = len(domain)

    if d <= 0:
        raise ValueError(
            "GRR domain must be non-empty."
        )

    if (
        not true_is_known_valid
        and true_value not in domain
    ):

        raise ValueError(
            "Private true value is not in the fixed public domain."
        )

    if d == 1:
        return true_value

    epsilon = float(
        epsilon
    )

    exp_eps = math.exp(
        epsilon
    )

    p_true = (
        exp_eps
        / (
            exp_eps
            + d
            - 1
        )
    )

    if rng.random() < p_true:
        return true_value

    # Uniformly sample from all OTHER public-domain values.
    # Rejection sampling avoids building a d-sized probability vector.
    while True:

        candidate = domain[
            rng.randrange(d)
        ]

        if candidate != true_value:
            return candidate


# =============================================================================
# Scheduler
# =============================================================================

def feasible_exec_buckets(
    remaining_budget: float,
    future_after_current: int,
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

    feasible: List[int] = []

    for bucket in range(
        1,
        cfg["K"] + 1,
    ):

        current_cost = (
            transition_cost(
                bucket,
                cfg,
            )
        )

        required = (
            current_cost
            + future_after_current
            * min_cost
        )

        if (
            required
            <= remaining_budget
            + 1e-12
        ):

            feasible.append(
                bucket
            )

    return feasible


def choose_exec_bucket(
    target_bucket: int,
    feasible: Sequence[int],
) -> int:
    """
    Bucket 1 has the smallest epsilon and strongest protection.

    If target bucket is infeasible, choose the closest feasible bucket
    that provides NO WEAKER protection, i.e., k <= target_bucket.
    """

    feasible = sorted(
        set(
            int(x)
            for x in feasible
        )
    )

    if not feasible:
        raise RuntimeError(
            "Feasible execution-bucket set is empty."
        )

    if target_bucket in feasible:
        return int(
            target_bucket
        )

    no_weaker = [
        k
        for k in feasible
        if k <= target_bucket
    ]

    if not no_weaker:

        raise RuntimeError(
            "No feasible no-weaker bucket exists. "
            "This should be impossible under the global "
            "minimum-cost feasibility gate."
        )

    return int(
        max(
            no_weaker
        )
    )


# =============================================================================
# Streaming Phase-3 risk groups
# =============================================================================

def iter_risk_groups(
    risk_path: str,
) -> Iterator[
    Tuple[
        str,
        List[Dict[str, Any]],
    ]
]:

    current_sbs_id: Optional[
        str
    ] = None

    current_group: List[
        Dict[str, Any]
    ] = []

    for item in iter_jsonl(
        risk_path
    ):

        sbs_id = str(
            item.get(
                "sbs_id"
            )
        )

        if current_sbs_id is None:

            current_sbs_id = (
                sbs_id
            )

        if sbs_id != current_sbs_id:

            yield (
                current_sbs_id,
                current_group,
            )

            current_sbs_id = (
                sbs_id
            )

            current_group = []

        current_group.append(
            item
        )

    if current_sbs_id is not None:

        yield (
            current_sbs_id,
            current_group,
        )


# =============================================================================
# External exact random shuffling
# =============================================================================

class ExternalShuffledReportWriter:
    """
    Memory-bounded exact random-order writer.

    Each report receives an independent random 128-bit key.
    Chunks are sorted locally and merged by that random key.
    Private IDs are never written into server-visible reports.
    """

    def __init__(
        self,
        output_path: str,
        temp_parent: str,
        rng: random.Random,
        chunk_size: int,
    ) -> None:

        self.output_path = (
            output_path
        )

        self.rng = rng

        self.chunk_size = max(
            1000,
            int(chunk_size),
        )

        ensure_parent_dir(
            output_path
        )

        ensure_dir(
            temp_parent
        )

        self.temp_dir = (
            tempfile.mkdtemp(
                prefix=".shuffle_",
                dir=temp_parent,
            )
        )

        self.buffer: List[
            Tuple[
                int,
                int,
                str,
            ]
        ] = []

        self.chunk_paths: List[
            str
        ] = []

        self.sequence_no = 0
        self.num_reports = 0

    def add(
        self,
        report: Dict[str, Any],
    ) -> None:

        random_key = (
            self.rng.getrandbits(
                128
            )
        )

        payload = json.dumps(
            report,
            ensure_ascii=False,
            separators=(
                ",",
                ":",
            ),
        )

        self.buffer.append(
            (
                random_key,
                self.sequence_no,
                payload,
            )
        )

        self.sequence_no += 1
        self.num_reports += 1

        if (
            len(self.buffer)
            >= self.chunk_size
        ):
            self._flush_chunk()

    def _flush_chunk(
        self,
    ) -> None:

        if not self.buffer:
            return

        self.buffer.sort(
            key=lambda x: (
                x[0],
                x[1],
            )
        )

        chunk_path = os.path.join(
            self.temp_dir,
            f"chunk_{len(self.chunk_paths):06d}.txt",
        )

        with open(
            chunk_path,
            "w",
            encoding="utf-8",
        ) as f:

            for (
                key,
                seq_no,
                payload,
            ) in self.buffer:

                f.write(
                    f"{key:032x}\t"
                    f"{seq_no:020d}\t"
                    f"{payload}\n"
                )

        self.chunk_paths.append(
            chunk_path
        )

        self.buffer = []

    @staticmethod
    def _iter_chunk(
        path: str,
    ) -> Iterator[
        Tuple[
            int,
            int,
            str,
        ]
    ]:

        with open(
            path,
            "r",
            encoding="utf-8",
        ) as f:

            for line in f:

                key_hex, seq_text, payload = (
                    line.rstrip(
                        "\n"
                    ).split(
                        "\t",
                        2,
                    )
                )

                yield (
                    int(
                        key_hex,
                        16,
                    ),

                    int(
                        seq_text
                    ),

                    payload,
                )

    def finalize(
        self,
    ) -> int:

        self._flush_chunk()

        ensure_parent_dir(
            self.output_path
        )

        if not self.chunk_paths:

            open(
                self.output_path,
                "w",
                encoding="utf-8",
            ).close()

            shutil.rmtree(
                self.temp_dir,
                ignore_errors=True,
            )

            return 0

        iterators = [
            self._iter_chunk(
                path
            )
            for path in self.chunk_paths
        ]

        with open(
            self.output_path,
            "w",
            encoding="utf-8",
        ) as fout:

            for (
                _key,
                _seq_no,
                payload,
            ) in heapq.merge(
                *iterators
            ):

                fout.write(
                    payload
                    + "\n"
                )

        shutil.rmtree(
            self.temp_dir,
            ignore_errors=True,
        )

        return int(
            self.num_reports
        )


# =============================================================================
# Debug writer
# =============================================================================

class OptionalDebugWriter:

    def __init__(
        self,
        path: Optional[str],
    ) -> None:

        self.path = path
        self.file = None
        self.count = 0

        if path is not None:

            ensure_parent_dir(
                path
            )

            self.file = open(
                path,
                "w",
                encoding="utf-8",
            )

    def write(
        self,
        item: Dict[str, Any],
    ) -> None:

        if self.file is None:
            return

        self.file.write(
            json.dumps(
                item,
                ensure_ascii=False,
            )
            + "\n"
        )

        self.count += 1

    def close(
        self,
    ) -> None:

        if self.file is not None:

            self.file.close()
            self.file = None


# =============================================================================
# Running statistics
# =============================================================================

class RunningScalar:

    def __init__(self) -> None:

        self.count = 0
        self.total = 0.0
        self.minimum = None
        self.maximum = None

    def add(
        self,
        value: float,
    ) -> None:

        value = float(
            value
        )

        self.count += 1
        self.total += value

        if (
            self.minimum is None
            or value < self.minimum
        ):
            self.minimum = value

        if (
            self.maximum is None
            or value > self.maximum
        ):
            self.maximum = value

    def summary(
        self,
    ) -> Dict[str, float]:

        return {
            "count": int(
                self.count
            ),

            "avg": (
                float(
                    self.total
                    / self.count
                )
                if self.count
                else 0.0
            ),

            "min": (
                float(
                    self.minimum
                )
                if self.minimum
                is not None
                else 0.0
            ),

            "max": (
                float(
                    self.maximum
                )
                if self.maximum
                is not None
                else 0.0
            ),
        }


# =============================================================================
# One split
# =============================================================================

def privatize_split(
    split_name: str,
    event_path: str,
    risk_path: str,
    report_path: str,
    debug_path: Optional[str],

    successor_cache: Dict[
        str,
        List[str],
    ],

    start_domain: Sequence[str],

    count_domain: Sequence[int],

    privacy_cfg: Dict[str, Any],

    mechanism_rng: random.Random,

    shuffle_rng: random.Random,

    shuffle_chunk_size: int,

    progress_every: int,
) -> Dict[str, Any]:

    report_writer = (
        ExternalShuffledReportWriter(
            output_path=(
                report_path
            ),

            temp_parent=(
                os.path.dirname(
                    report_path
                )
            ),

            rng=shuffle_rng,

            chunk_size=(
                shuffle_chunk_size
            ),
        )
    )

    debug_writer = (
        OptionalDebugWriter(
            debug_path
        )
    )

    target_bucket_counts = (
        Counter()
    )

    exec_bucket_counts = (
        Counter()
    )

    keep_target = (
        Counter()
    )

    total_target = (
        Counter()
    )

    fallback_count = 0
    num_sbs = 0
    num_transitions = 0

    spend_stats = (
        RunningScalar()
    )

    remaining_stats = (
        RunningScalar()
    )

    transition_eps_stats = (
        RunningScalar()
    )

    num_feasible_empty = 0
    num_domain_violations = 0

    event_iter = iter_jsonl(
        event_path
    )

    risk_group_iter = (
        iter_risk_groups(
            risk_path
        )
    )

    try:

        for record_index, pair in enumerate(
            itertools.zip_longest(
                event_iter,
                risk_group_iter,
            ),
            start=1,
        ):

            event_record, risk_group = (
                pair
            )

            if event_record is None:

                raise RuntimeError(
                    "Risk file contains more SBS groups "
                    "than the event file."
                )

            if risk_group is None:

                raise RuntimeError(
                    "Event file contains more SBS records "
                    "than the risk file."
                )

            (
                risk_sbs_id,
                risk_items,
            ) = risk_group

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
                    "Event/risk SBS order mismatch: "
                    f"event={sbs_id}, "
                    f"risk={risk_sbs_id}"
                )

            num_sbs += 1

            true_start = str(
                event_record[
                    "start_event"
                ][
                    "st"
                ]
            )

            true_count = int(
                event_record[
                    "count_event"
                ][
                    "cnt"
                ]
            )

            transitions = (
                event_record[
                    "transition_events"
                ]
            )

            if (
                true_count
                != len(
                    event_record[
                        "segments"
                    ]
                )
            ):

                raise RuntimeError(
                    f"SBS count mismatch: {sbs_id}"
                )

            if (
                len(transitions)
                != true_count - 1
            ):

                raise RuntimeError(
                    f"Transition count mismatch: {sbs_id}"
                )

            if (
                len(risk_items)
                != len(transitions)
            ):

                raise RuntimeError(
                    f"Risk item count mismatch: {sbs_id}"
                )

            # -------------------------------------------------------------
            # 1. Start report
            # -------------------------------------------------------------

            if (
                true_start
                not in successor_cache
            ):

                num_domain_violations += 1

                raise RuntimeError(
                    "True SBS start is outside the fixed PUBLIC "
                    f"road-segment domain: {true_start}"
                )

            noisy_start = (
                grr_sample_fixed_domain(
                    true_value=(
                        true_start
                    ),

                    domain=(
                        start_domain
                    ),

                    epsilon=(
                        privacy_cfg[
                            "eps_start"
                        ]
                    ),

                    rng=(
                        mechanism_rng
                    ),

                    true_is_known_valid=True,
                )
            )

            report_writer.add(
                {
                    "event_type": "start",
                    "x_noisy": (
                        noisy_start
                    ),
                }
            )

            debug_writer.write(
                {
                    "event_type": "start",
                    "sbs_id": sbs_id,
                    "true_x": true_start,
                    "noisy_x": noisy_start,
                    "epsilon_used": (
                        privacy_cfg[
                            "eps_start"
                        ]
                    ),
                }
            )

            # -------------------------------------------------------------
            # 2. Exact segment-count report
            # -------------------------------------------------------------

            if (
                true_count < 1
                or true_count
                > privacy_cfg[
                    "L_max"
                ]
            ):

                num_domain_violations += 1

                raise RuntimeError(
                    "True SBS count is outside "
                    "PUBLIC count domain."
                )

            noisy_count = (
                grr_sample_fixed_domain(
                    true_value=(
                        true_count
                    ),

                    domain=(
                        count_domain
                    ),

                    epsilon=(
                        privacy_cfg[
                            "eps_count"
                        ]
                    ),

                    rng=(
                        mechanism_rng
                    ),
                )
            )

            report_writer.add(
                {
                    "event_type": "count",
                    "x_noisy": int(
                        noisy_count
                    ),
                }
            )

            debug_writer.write(
                {
                    "event_type": "count",
                    "sbs_id": sbs_id,
                    "true_x": true_count,
                    "noisy_x": int(
                        noisy_count
                    ),
                    "epsilon_used": (
                        privacy_cfg[
                            "eps_count"
                        ]
                    ),
                }
            )

            # -------------------------------------------------------------
            # 3. Transition reports
            # -------------------------------------------------------------

            remaining = (
                privacy_cfg[
                    "B_total"
                ]
                - privacy_cfg[
                    "eps_start"
                ]
                - privacy_cfg[
                    "eps_count"
                ]
            )

            transition_spend = 0.0

            for local_index, (
                transition,
                risk_item,
            ) in enumerate(
                zip(
                    transitions,
                    risk_items,
                ),
                start=1,
            ):

                t = int(
                    transition[
                        "t"
                    ]
                )

                risk_t = int(
                    risk_item[
                        "t"
                    ]
                )

                if t != risk_t:

                    raise RuntimeError(
                        "Transition/risk t mismatch: "
                        f"sbs_id={sbs_id}, "
                        f"event_t={t}, "
                        f"risk_t={risk_t}"
                    )

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
                        "Transition/risk value mismatch: "
                        f"sbs_id={sbs_id}, t={t}"
                    )

                public_domain = (
                    successor_cache.get(
                        u,
                        None,
                    )
                )

                if (
                    public_domain is None
                    or v
                    not in public_domain
                ):

                    num_domain_violations += 1

                    raise RuntimeError(
                        "True successor is outside fixed PUBLIC "
                        f"N(u): u={u}, v={v}"
                    )

                target_bucket = int(
                    risk_item.get(
                        "target_bucket",
                        risk_item.get(
                            "b_t"
                        ),
                    )
                )

                if not (
                    1
                    <= target_bucket
                    <= privacy_cfg[
                        "K"
                    ]
                ):

                    raise RuntimeError(
                        "Invalid target bucket."
                    )

                future_after = (
                    len(transitions)
                    - local_index
                )

                feasible = (
                    feasible_exec_buckets(
                        remaining_budget=(
                            remaining
                        ),

                        future_after_current=(
                            future_after
                        ),

                        cfg=(
                            privacy_cfg
                        ),
                    )
                )

                if not feasible:

                    num_feasible_empty += 1

                    raise RuntimeError(
                        "Empty feasible execution-bucket set."
                    )

                exec_bucket = (
                    choose_exec_bucket(
                        target_bucket=(
                            target_bucket
                        ),
                        feasible=feasible,
                    )
                )

                if (
                    exec_bucket
                    != target_bucket
                ):
                    fallback_count += 1

                eps_event = float(
                    privacy_cfg[
                        "eps_event_list"
                    ][
                        exec_bucket
                        - 1
                    ]
                )

                current_cost = (
                    eps_event
                    + privacy_cfg[
                        "eps_bucket"
                    ]
                )

                remaining_before = (
                    remaining
                )

                noisy_successor = (
                    grr_sample_fixed_domain(
                        true_value=v,

                        domain=(
                            public_domain
                        ),

                        epsilon=(
                            eps_event
                        ),

                        rng=(
                            mechanism_rng
                        ),
                    )
                )

                noisy_bucket = (
                    grr_sample_fixed_domain(
                        true_value=(
                            exec_bucket
                        ),

                        domain=tuple(
                            range(
                                1,
                                privacy_cfg[
                                    "K"
                                ]
                                + 1,
                            )
                        ),

                        epsilon=(
                            privacy_cfg[
                                "eps_bucket"
                            ]
                        ),

                        rng=(
                            mechanism_rng
                        ),
                    )
                )

                # Server-visible report.
                report_writer.add(
                    {
                        "event_type": (
                            "transition"
                        ),

                        "u": u,

                        "y": str(
                            noisy_successor
                        ),

                        "k_noisy": int(
                            noisy_bucket
                        ),
                    }
                )

                remaining -= (
                    current_cost
                )

                transition_spend += (
                    current_cost
                )

                if remaining < -1e-10:

                    raise RuntimeError(
                        "SBS reporting schedule exceeded B."
                    )

                keep_flag = int(
                    noisy_successor
                    == v
                )

                target_bucket_counts[
                    target_bucket
                ] += 1

                exec_bucket_counts[
                    exec_bucket
                ] += 1

                total_target[
                    target_bucket
                ] += 1

                keep_target[
                    target_bucket
                ] += keep_flag

                transition_eps_stats.add(
                    eps_event
                )

                num_transitions += 1

                debug_writer.write(
                    {
                        "event_type": (
                            "transition"
                        ),

                        "sbs_id": sbs_id,

                        "t": t,

                        "u": u,

                        "true_v": v,

                        "y": str(
                            noisy_successor
                        ),

                        "target_bucket": (
                            target_bucket
                        ),

                        "exec_bucket": (
                            exec_bucket
                        ),

                        "k_noisy": int(
                            noisy_bucket
                        ),

                        "epsilon_event": (
                            eps_event
                        ),

                        "epsilon_bucket": (
                            privacy_cfg[
                                "eps_bucket"
                            ]
                        ),

                        "transition_cost": (
                            current_cost
                        ),

                        "remaining_before": (
                            remaining_before
                        ),

                        "remaining_after": (
                            remaining
                        ),

                        "future_after": (
                            future_after
                        ),

                        "feasible_buckets": (
                            list(
                                feasible
                            )
                        ),

                        "keep_flag": (
                            keep_flag
                        ),

                        "risk_score": (
                            risk_item.get(
                                "risk_score"
                            )
                        ),
                    }
                )

            sbs_total_spend = (
                privacy_cfg[
                    "eps_start"
                ]
                + privacy_cfg[
                    "eps_count"
                ]
                + transition_spend
            )

            if (
                sbs_total_spend
                > privacy_cfg[
                    "B_total"
                ]
                + 1e-10
            ):

                raise RuntimeError(
                    "SBS schedule-cap violation: "
                    f"{sbs_total_spend} > "
                    f"{privacy_cfg['B_total']}"
                )

            spend_stats.add(
                sbs_total_spend
            )

            remaining_stats.add(
                privacy_cfg[
                    "B_total"
                ]
                - sbs_total_spend
            )

            if (
                progress_every > 0
                and record_index
                % progress_every
                == 0
            ):

                print(
                    f"[privatize][{split_name}] "
                    f"SBSs={record_index:,}, "
                    f"transitions={num_transitions:,}, "
                    f"fallbacks={fallback_count:,}"
                )

    finally:

        debug_writer.close()

    num_reports = (
        report_writer.finalize()
    )

    expected_reports = (
        2
        * num_sbs
        + num_transitions
    )

    if (
        num_reports
        != expected_reports
    ):

        raise RuntimeError(
            "Server-report count mismatch: "
            f"actual={num_reports}, "
            f"expected={expected_reports}"
        )

    keep_by_target = {}

    for bucket in range(
        1,
        privacy_cfg["K"] + 1,
    ):

        denominator = (
            total_target[
                bucket
            ]
        )

        keep_by_target[
            str(bucket)
        ] = (
            float(
                keep_target[
                    bucket
                ]
                / denominator
            )
            if denominator
            else 0.0
        )

    summary = {
        "split": split_name,

        "num_sbs": int(
            num_sbs
        ),

        "num_transition_events": int(
            num_transitions
        ),

        "num_start_reports": int(
            num_sbs
        ),

        "num_count_reports": int(
            num_sbs
        ),

        "num_transition_reports": int(
            num_transitions
        ),

        "num_server_reports": int(
            num_reports
        ),

        "expected_server_reports": int(
            expected_reports
        ),

        "target_bucket_counts": {
            str(k): int(
                target_bucket_counts[k]
            )
            for k in range(
                1,
                privacy_cfg["K"] + 1,
            )
        },

        "exec_bucket_counts": {
            str(k): int(
                exec_bucket_counts[k]
            )
            for k in range(
                1,
                privacy_cfg["K"] + 1,
            )
        },

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

        "keep_rate_by_target_bucket": (
            keep_by_target
        ),

        "sbs_schedule_spend": (
            spend_stats.summary()
        ),

        "sbs_remaining_margin": (
            remaining_stats.summary()
        ),

        "transition_epsilon": (
            transition_eps_stats.summary()
        ),

        "empty_feasible_set_count": int(
            num_feasible_empty
        ),

        "public_domain_violation_count": int(
            num_domain_violations
        ),

        "server_report_path": (
            report_path
        ),

        "debug_path": (
            debug_path
        ),
    }

    print(
        f"[privatize][{split_name}] DONE | "
        f"SBSs={num_sbs:,}, "
        f"transitions={num_transitions:,}, "
        f"reports={num_reports:,}, "
        f"fallback={fallback_count:,}, "
        f"max-spend="
        f"{summary['sbs_schedule_spend']['max']:.6f}"
    )

    return summary


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

    privacy_cfg = (
        extract_privacy_cfg(
            exp_cfg
        )
    )

    validate_canonical_privacy_cfg(
        exp_cfg,
        privacy_cfg,
    )

    budget_info = (
        validate_budget_feasibility(
            privacy_cfg
        )
    )

    print("=" * 90)

    print(
        "[privatize_riskaware] "
        "Canonical Phase-4 configuration"
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
                "privacy_cfg": (
                    privacy_cfg
                ),

                "budget_info": (
                    budget_info
                ),
            },
            indent=2,
            ensure_ascii=False,
        )
    )

    print("=" * 90)

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

    experiment_root = resolve_path(
        exp_cfg[
            "experiment_root"
        ]
    )

    privatized_dir = resolve_path(
        exp_cfg[
            "privatized_dir"
        ]
    )

    successor_cache_path = resolve_path(
        dataset_cfg[
            "successor_cache_path"
        ]
    )

    assert experiment_root is not None
    assert privatized_dir is not None
    assert successor_cache_path is not None

    risk_dir = os.path.join(
        experiment_root,
        "risk",
    )

    risk_paths = {
        split: os.path.join(
            risk_dir,
            f"{split}_risk.jsonl",
        )
        for split in [
            "train",
            "valid",
            "test",
        ]
    }

    for split in [
        "train",
        "valid",
        "test",
    ]:

        if (
            event_paths[split]
            is None
            or not os.path.exists(
                event_paths[split]
            )
        ):

            raise FileNotFoundError(
                f"Missing Phase-2 event file: "
                f"{event_paths[split]}"
            )

        if not os.path.exists(
            risk_paths[
                split
            ]
        ):

            raise FileNotFoundError(
                f"Missing Phase-3 risk file: "
                f"{risk_paths[split]}"
            )

    if not os.path.exists(
        successor_cache_path
    ):

        raise FileNotFoundError(
            "Missing public successor cache."
        )

    ensure_dir(
        privatized_dir
    )

    total_stages = 5

    # =========================================================================
    # Stage 1
    # =========================================================================

    start = log_stage(
        1,
        total_stages,
        "Loading PUBLIC domains...",
    )

    successor_cache = load_json(
        successor_cache_path
    )

    if not isinstance(
        successor_cache,
        dict,
    ):

        raise ValueError(
            "successor cache must be a JSON object."
        )

    # PUBLIC start domain = all fixed public road segments.
    start_domain = tuple(
        successor_cache.keys()
    )

    # PUBLIC exact count domain.
    count_domain = tuple(
        range(
            1,
            privacy_cfg[
                "L_max"
            ]
            + 1,
        )
    )

    if not start_domain:

        raise RuntimeError(
            "Public start domain is empty."
        )

    print(
        "[Info] public road-segment domain size = "
        f"{len(start_domain):,}"
    )

    print(
        "[Info] public exact-count domain = "
        f"1..{privacy_cfg['L_max']}"
    )

    log_done(
        start,
        "Public domains loaded",
    )

    save_debug = bool(
        args.save_debug
        or privacy_cfg[
            "save_debug_files"
        ]
    )

    split_seed_offsets = {
        "train": 101,
        "valid": 202,
        "test": 303,
    }

    summaries = {}

    # =========================================================================
    # Stages 2-4
    # =========================================================================

    for stage_idx, split in zip(
        [2, 3, 4],
        [
            "train",
            "valid",
            "test",
        ],
    ):

        stage_start = (
            log_stage(
                stage_idx,
                total_stages,
                f"Privatizing {split} split...",
            )
        )

        mechanism_seed = (
            privacy_cfg[
                "random_seed"
            ]
            * 10000
            + split_seed_offsets[
                split
            ]
        )

        shuffle_seed = (
            mechanism_seed
            + 5000003
        )

        mechanism_rng = (
            random.Random(
                mechanism_seed
            )
        )

        shuffle_rng = (
            random.Random(
                shuffle_seed
            )
        )

        report_path = os.path.join(
            privatized_dir,
            f"riskaware_{split}_reports.jsonl",
        )

        debug_path = (
            os.path.join(
                privatized_dir,
                f"riskaware_{split}_debug.jsonl",
            )
            if save_debug
            else None
        )

        summaries[
            split
        ] = privatize_split(
            split_name=split,

            event_path=(
                event_paths[
                    split
                ]
            ),

            risk_path=(
                risk_paths[
                    split
                ]
            ),

            report_path=(
                report_path
            ),

            debug_path=(
                debug_path
            ),

            successor_cache=(
                successor_cache
            ),

            start_domain=(
                start_domain
            ),

            count_domain=(
                count_domain
            ),

            privacy_cfg=(
                privacy_cfg
            ),

            mechanism_rng=(
                mechanism_rng
            ),

            shuffle_rng=(
                shuffle_rng
            ),

            shuffle_chunk_size=(
                args.shuffle_chunk_size
            ),

            progress_every=(
                args.progress_every
            ),
        )

        log_done(
            stage_start,
            f"{split} split privatized",
        )

    # =========================================================================
    # Stage 5
    # =========================================================================

    start = log_stage(
        5,
        total_stages,
        "Saving Phase-4 metadata and summary...",
    )

    eps_event_max = max(
        privacy_cfg[
            "eps_event_list"
        ]
    )

    eps_transition_envelope = (
        eps_event_max
        + privacy_cfg[
            "eps_bucket"
        ]
    )

    conditional_fixed_context_sbs_bound = (
        privacy_cfg[
            "eps_start"
        ]
        + privacy_cfg[
            "eps_count"
        ]
        + (
            privacy_cfg[
                "L_max"
            ]
            - 1
        )
        * eps_transition_envelope
    )

    meta = {
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

        "dataset_root": (
            dataset_cfg[
                "dataset_root"
            ]
        ),

        "experiment_root": (
            experiment_root
        ),

        "exp_tag": (
            exp_cfg[
                "exp_tag"
            ]
        ),

        "method": "riskaware",

        "privacy_scope": (
            privacy_cfg[
                "privacy_scope"
            ]
        ),

        "B_semantics": (
            privacy_cfg[
                "B_semantics"
            ]
        ),

        "B_total": (
            privacy_cfg[
                "B_total"
            ]
        ),

        "privacy_parameters": {
            "eps_start": (
                privacy_cfg[
                    "eps_start"
                ]
            ),

            "eps_count": (
                privacy_cfg[
                    "eps_count"
                ]
            ),

            "eps_bucket": (
                privacy_cfg[
                    "eps_bucket"
                ]
            ),

            "eps_event_list": (
                privacy_cfg[
                    "eps_event_list"
                ]
            ),
        },

        "public_domains": {
            "start": (
                "all road segments in fixed "
                "public successor cache"
            ),

            "start_domain_size": (
                len(
                    start_domain
                )
            ),

            "count": (
                f"1..{privacy_cfg['L_max']}"
            ),

            "successor": (
                "fixed public N(u)"
            ),

            "bucket": (
                f"1..{privacy_cfg['K']}"
            ),

            "secret_dependent_domain_expansion": (
                False
            ),
        },

        "formal_transition_guarantee": {
            "type": (
                "conditional_event_level_LDP"
            ),

            "public_context": (
                "U=current road segment"
            ),

            "protected_value": (
                "V=true successor in N(U)"
            ),

            "epsilon_bar_transition": (
                eps_transition_envelope
            ),

            "allows_private_data_dependent_exec_bucket": (
                True
            ),
        },

        "formal_full_sbs_B_ldp_claim": (
            False
        ),

        "conditional_fixed_context_sequence_composition_upper_bound": (
            conditional_fixed_context_sbs_bound
        ),

        "notes": {
            "B_is_schedule_cap_not_full_sbs_ldp": (
                True
            ),

            "public_current_segment_is_not_protected": (
                True
            ),

            "server_reports_are_deidentified": (
                True
            ),

            "server_reports_are_randomly_shuffled": (
                True
            ),

            "private_debug_is_server_visible": (
                False
            ),
        },

        "server_report_schema": {
            "start": [
                "event_type",
                "x_noisy",
            ],

            "count": [
                "event_type",
                "x_noisy",
            ],

            "transition": [
                "event_type",
                "u",
                "y",
                "k_noisy",
            ],
        },
    }

    summary = {
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

        "budget_info": (
            budget_info
        ),

        "save_debug_files": bool(
            save_debug
        ),

        "splits": summaries,
    }

    meta_path = os.path.join(
        privatized_dir,
        "riskaware_meta.json",
    )

    summary_path = os.path.join(
        privatized_dir,
        "riskaware_summary.json",
    )

    save_json(
        meta_path,
        meta,
    )

    save_json(
        summary_path,
        summary,
    )

    log_done(
        start,
        "Phase-4 metadata saved",
    )

    print("=" * 90)

    print(
        "[privatize_riskaware] PHASE 4A DONE"
    )

    print(
        f"Privatized directory:\n"
        f"  {privatized_dir}"
    )

    print(
        f"Metadata:\n"
        f"  {meta_path}"
    )

    print(
        f"Summary:\n"
        f"  {summary_path}"
    )

    print(
        "[Formal scope] "
        f"transition epsilon_bar = "
        f"{eps_transition_envelope:.6f}"
    )

    print(
        "[Important] "
        "B is an SBS reporting-schedule cap; "
        "no arbitrary full-SBS B-LDP claim is made."
    )

    print("=" * 90)


if __name__ == "__main__":
    main()