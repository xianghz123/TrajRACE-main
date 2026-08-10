#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

"""
event_utils.py

Canonical TrajRACE SBS/event construction utilities.

A complete map-matched road-segment trajectory

    T = <g_1.e, ..., g_n.e>

is decomposed into one or more Segment-Bounded Subtrajectories (SBSs).

For every SBS S_j, the local/private information units are

    I(S_j) = {st_j, cnt_j, X_j},

where

    st_j  : first road segment of S_j,
    cnt_j : exact number of road segments in S_j,
    X_j   : consecutive transition units inside S_j.

Important
---------
This module performs deterministic LOCAL preprocessing only.
It performs no privacy randomization and produces no server-visible report.
"""

import math
from collections import Counter
from statistics import mean, median
from typing import Any, Dict, List, Optional, Sequence


# =============================================================================
# Validation helpers
# =============================================================================

def _validate_segments(
    segments: Any,
    traj_id: Any,
) -> List[Any]:
    """
    Validate one complete map-matched road-segment sequence.

    Segment identifiers are intentionally preserved as-is because canonical
    preprocessing represents a public directed road segment as a string such as

        "u__v__key".
    """
    if not isinstance(segments, list):
        raise ValueError(
            f"segments must be a list for traj_id={traj_id}; "
            f"got {type(segments)}"
        )

    if len(segments) < 2:
        raise ValueError(
            f"traj_id={traj_id} contains fewer than two road segments"
        )

    return list(segments)


def _validate_segment_times(
    segment_times: Any,
    num_segments: int,
    traj_id: Any,
) -> Optional[List[Any]]:
    """
    Validate road-segment timestamps.

    Porto canonical preprocessing provides one timestamp per mapped segment.
    The None case is retained for datasets where timestamps are unavailable.
    """
    if segment_times is None:
        return None

    if not isinstance(segment_times, list):
        raise ValueError(
            f"segment_times must be list or None for traj_id={traj_id}"
        )

    if len(segment_times) != num_segments:
        raise ValueError(
            f"Timestamp/segment length mismatch for traj_id={traj_id}: "
            f"{len(segment_times)} vs {num_segments}"
        )

    return list(segment_times)


def _compute_delta_t(
    tau_u: Any,
    tau_v: Any,
) -> Optional[float]:
    """
    Compute the realized timestamp gap of one transition.
    """
    if tau_u is None or tau_v is None:
        return None

    try:
        delta_t = float(tau_v) - float(tau_u)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid timestamps: tau_u={tau_u}, tau_v={tau_v}"
        ) from exc

    if delta_t < 0:
        raise ValueError(
            f"Negative timestamp gap: tau_u={tau_u}, tau_v={tau_v}"
        )

    if float(delta_t).is_integer():
        return int(delta_t)

    return float(delta_t)


def _nearest_rank_percentile(
    values: Sequence[int],
    percentile: float,
) -> float:
    """
    Nearest-rank percentile without an external statistics dependency.
    """
    if not values:
        return 0.0

    xs = sorted(values)

    if percentile <= 0:
        return float(xs[0])

    if percentile >= 100:
        return float(xs[-1])

    rank = int(
        math.ceil(
            percentile / 100.0 * len(xs)
        )
    )

    rank = max(
        1,
        min(rank, len(xs)),
    )

    return float(xs[rank - 1])


# =============================================================================
# SBS decomposition
# =============================================================================

def partition_sequence_into_sbs(
    seq_record: Dict[str, Any],
    L_max: int = 30,
    mode: str = "transition_preserving_overlap",
) -> List[Dict[str, Any]]:
    """
    Decompose one complete mapped trajectory into SBSs.

    Canonical rule
    --------------
    Each SBS contains at most L_max road segments.

    Adjacent SBSs share exactly one boundary road segment so that every
    transition of the original trajectory appears in exactly one SBS.

    Example
    -------
    L_max = 4

        T  = [e1, e2, e3, e4, e5, e6, e7]

        S1 = [e1, e2, e3, e4]
        S2 = [e4, e5, e6, e7]

    Transition coverage

        S1: e1->e2, e2->e3, e3->e4
        S2: e4->e5, e5->e6, e6->e7

    Thus no original transition is lost or duplicated.

    For n >= 2 road segments,

        q = ceil((n - 1) / (L_max - 1)).
    """
    if int(L_max) < 2:
        raise ValueError(
            "L_max must be at least 2"
        )

    if mode != "transition_preserving_overlap":
        raise ValueError(
            "Canonical TrajRACE requires "
            "sbs_partition_mode='transition_preserving_overlap'"
        )

    traj_id = seq_record.get(
        "traj_id"
    )

    taxi_id = seq_record.get(
        "taxi_id",
        None,
    )

    parent_start_timestamp = seq_record.get(
        "start_timestamp",
        None,
    )

    segments = _validate_segments(
        seq_record.get("segments"),
        traj_id=traj_id,
    )

    segment_times = _validate_segment_times(
        seq_record.get(
            "segment_times",
            None,
        ),
        num_segments=len(segments),
        traj_id=traj_id,
    )

    n = len(segments)

    sbs_records: List[
        Dict[str, Any]
    ] = []

    # 0-based index of the first segment of the current SBS
    # in the complete parent trajectory.
    start_idx = 0
    sbs_index = 1

    while start_idx < n - 1:

        end_exclusive = min(
            start_idx + int(L_max),
            n,
        )

        sbs_segments = segments[
            start_idx:end_exclusive
        ]

        if segment_times is None:
            sbs_times = None
        else:
            sbs_times = segment_times[
                start_idx:end_exclusive
            ]

        # Paper-style 1-based parent trajectory indices.
        a_j = start_idx + 1
        r_j = end_exclusive

        sbs_records.append(
            {
                # -------------------------------------------------------------
                # Local-only identifiers.
                # -------------------------------------------------------------
                "traj_id": traj_id,
                "taxi_id": taxi_id,
                "sbs_id": (
                    f"{traj_id}::sbs{sbs_index}"
                ),
                "sbs_index": int(
                    sbs_index
                ),

                # -------------------------------------------------------------
                # Position in the complete mapped trajectory.
                # -------------------------------------------------------------
                "a_j": int(a_j),
                "r_j": int(r_j),

                "parent_num_segments": int(n),

                "parent_start_timestamp": (
                    parent_start_timestamp
                ),

                # -------------------------------------------------------------
                # Private SBS content.
                # -------------------------------------------------------------
                "segments": list(
                    sbs_segments
                ),

                "segment_times": (
                    None
                    if sbs_times is None
                    else list(sbs_times)
                ),
            }
        )

        if end_exclusive >= n:
            break

        # Reuse the final road segment as the first road segment
        # of the following SBS.
        start_idx = end_exclusive - 1

        sbs_index += 1

    expected_q = int(
        math.ceil(
            (n - 1)
            / (int(L_max) - 1)
        )
    )

    if len(sbs_records) != expected_q:
        raise RuntimeError(
            f"SBS decomposition mismatch for traj_id={traj_id}: "
            f"expected={expected_q}, got={len(sbs_records)}"
        )

    for item in sbs_records:
        item["num_sbs"] = int(
            len(sbs_records)
        )

    return sbs_records


# =============================================================================
# Private information-unit construction
# =============================================================================

def build_event_record_from_sbs(
    sbs_record: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Convert one SBS into the canonical TrajRACE private information units

        I(S_j) = {st_j, cnt_j, X_j}.

    Important
    ---------
    This is still client/local private data.

    traj_id, sbs_id, true successors, timestamps, etc. MUST NOT later appear
    in server-visible privatized reports.
    """
    traj_id = sbs_record.get(
        "traj_id"
    )

    segments = _validate_segments(
        sbs_record.get("segments"),
        traj_id=traj_id,
    )

    segment_times = _validate_segment_times(
        sbs_record.get(
            "segment_times",
            None,
        ),
        num_segments=len(segments),
        traj_id=traj_id,
    )

    cnt_j = int(
        len(segments)
    )

    R_j = int(
        cnt_j - 1
    )

    start_event = {
        "st": segments[0],
    }

    count_event = {
        "cnt": cnt_j,
    }

    transition_events: List[
        Dict[str, Any]
    ] = []

    a_j = int(
        sbs_record["a_j"]
    )

    for local_idx in range(
        cnt_j - 1
    ):
        u = segments[
            local_idx
        ]

        v = segments[
            local_idx + 1
        ]

        if segment_times is None:
            tau_u = None
            tau_v = None
        else:
            tau_u = segment_times[
                local_idx
            ]

            tau_v = segment_times[
                local_idx + 1
            ]

        delta_t = _compute_delta_t(
            tau_u,
            tau_v,
        )

        # 1-based transition position inside the current SBS.
        t = local_idx + 1

        # 1-based current-segment position in the complete parent trajectory.
        global_pos = (
            a_j
            + local_idx
        )

        transition_events.append(
            {
                "u": u,
                "v": v,

                "t": int(t),

                "global_pos": int(
                    global_pos
                ),

                "cnt": cnt_j,

                "tau_u": tau_u,
                "tau_v": tau_v,
                "delta_t": delta_t,
            }
        )

    if len(transition_events) != R_j:
        raise RuntimeError(
            f"Transition count mismatch for traj_id={traj_id}, "
            f"sbs_id={sbs_record.get('sbs_id')}"
        )

    return {
        # ---------------------------------------------------------------------
        # Local-only metadata.
        # ---------------------------------------------------------------------
        "traj_id": traj_id,

        "taxi_id": sbs_record.get(
            "taxi_id"
        ),

        "sbs_id": sbs_record.get(
            "sbs_id"
        ),

        "sbs_index": int(
            sbs_record[
                "sbs_index"
            ]
        ),

        "num_sbs": int(
            sbs_record[
                "num_sbs"
            ]
        ),

        "a_j": int(
            sbs_record[
                "a_j"
            ]
        ),

        "r_j": int(
            sbs_record[
                "r_j"
            ]
        ),

        "parent_num_segments": int(
            sbs_record[
                "parent_num_segments"
            ]
        ),

        "parent_start_timestamp": (
            sbs_record.get(
                "parent_start_timestamp"
            )
        ),

        # ---------------------------------------------------------------------
        # Local/private SBS content required by the later risk stage.
        # ---------------------------------------------------------------------
        "segments": list(
            segments
        ),

        "segment_times": (
            None
            if segment_times is None
            else list(segment_times)
        ),

        # ---------------------------------------------------------------------
        # Canonical private information units.
        # ---------------------------------------------------------------------
        "start_event": start_event,

        "count_event": count_event,

        "transition_events": (
            transition_events
        ),

        "num_transitions": R_j,
    }


def build_event_records_from_sequence(
    seq_record: Dict[str, Any],
    L_max: int = 30,
    partition_mode: str = "transition_preserving_overlap",
) -> List[Dict[str, Any]]:
    """
    Convert one complete mapped trajectory into one or more SBS event records.
    """
    sbs_records = (
        partition_sequence_into_sbs(
            seq_record=seq_record,
            L_max=L_max,
            mode=partition_mode,
        )
    )

    return [
        build_event_record_from_sbs(
            sbs_record
        )
        for sbs_record
        in sbs_records
    ]


def build_event_records_from_sequences(
    seq_records: List[
        Dict[str, Any]
    ],
    L_max: int = 30,
    partition_mode: str = "transition_preserving_overlap",
    skip_invalid: bool = False,
) -> List[Dict[str, Any]]:
    """
    Convert multiple complete trajectories into SBS event records.

    Invalid records are not silently skipped in canonical runs.
    """
    results: List[
        Dict[str, Any]
    ] = []

    for record in seq_records:

        try:
            results.extend(
                build_event_records_from_sequence(
                    seq_record=record,
                    L_max=L_max,
                    partition_mode=(
                        partition_mode
                    ),
                )
            )

        except Exception:
            if skip_invalid:
                continue

            raise

    return results


# =============================================================================
# Audit statistics
# =============================================================================

def summarize_event_records(
    event_records: List[
        Dict[str, Any]
    ],
) -> Dict[str, Any]:
    """
    Produce SBS statistics needed for sanity checking and R1-D2.
    """
    if not event_records:
        return {
            "num_parent_trajectories": 0,
            "num_sbs": 0,

            "num_single_sbs_trajectories": 0,
            "num_multi_sbs_trajectories": 0,
            "fraction_multi_sbs": 0.0,

            "avg_sbs_per_trajectory": 0.0,
            "median_sbs_per_trajectory": 0.0,
            "p95_sbs_per_trajectory": 0.0,
            "max_sbs_per_trajectory": 0,

            "avg_sbs_segments": 0.0,
            "min_sbs_segments": 0,
            "max_sbs_segments": 0,

            "num_start_events": 0,
            "num_count_events": 0,
            "num_transition_events": 0,

            "expected_parent_transitions": 0,
            "transition_coverage_ratio": 0.0,

            "num_shared_boundary_segments": 0,
        }

    sbs_per_traj: Counter = (
        Counter()
    )

    parent_num_segments: Dict[
        Any,
        int,
    ] = {}

    sbs_lengths: List[int] = []

    total_transitions = 0

    for record in event_records:

        traj_id = record[
            "traj_id"
        ]

        sbs_per_traj[
            traj_id
        ] += 1

        parent_num_segments.setdefault(
            traj_id,
            int(
                record[
                    "parent_num_segments"
                ]
            ),
        )

        cnt = int(
            record[
                "count_event"
            ]["cnt"]
        )

        sbs_lengths.append(
            cnt
        )

        total_transitions += len(
            record[
                "transition_events"
            ]
        )

    q_values = list(
        sbs_per_traj.values()
    )

    num_parent = len(
        sbs_per_traj
    )

    num_sbs = len(
        event_records
    )

    num_single = sum(
        1
        for q in q_values
        if q == 1
    )

    num_multi = sum(
        1
        for q in q_values
        if q > 1
    )

    expected_parent_transitions = sum(
        max(
            0,
            n - 1,
        )
        for n
        in parent_num_segments.values()
    )

    if expected_parent_transitions > 0:
        transition_coverage_ratio = (
            total_transitions
            / expected_parent_transitions
        )
    else:
        transition_coverage_ratio = 0.0

    return {
        "num_parent_trajectories": int(
            num_parent
        ),

        "num_sbs": int(
            num_sbs
        ),

        "num_single_sbs_trajectories": int(
            num_single
        ),

        "num_multi_sbs_trajectories": int(
            num_multi
        ),

        "fraction_multi_sbs": (
            float(
                num_multi
                / num_parent
            )
            if num_parent
            else 0.0
        ),

        "avg_sbs_per_trajectory": float(
            mean(q_values)
        ),

        "median_sbs_per_trajectory": float(
            median(q_values)
        ),

        "p95_sbs_per_trajectory": float(
            _nearest_rank_percentile(
                q_values,
                95.0,
            )
        ),

        "max_sbs_per_trajectory": int(
            max(q_values)
        ),

        "avg_sbs_segments": float(
            mean(
                sbs_lengths
            )
        ),

        "min_sbs_segments": int(
            min(
                sbs_lengths
            )
        ),

        "max_sbs_segments": int(
            max(
                sbs_lengths
            )
        ),

        "num_start_events": int(
            num_sbs
        ),

        "num_count_events": int(
            num_sbs
        ),

        "num_transition_events": int(
            total_transitions
        ),

        "expected_parent_transitions": int(
            expected_parent_transitions
        ),

        "transition_coverage_ratio": float(
            transition_coverage_ratio
        ),

        # One shared boundary segment for every additional SBS.
        "num_shared_boundary_segments": int(
            num_sbs
            - num_parent
        ),
    }