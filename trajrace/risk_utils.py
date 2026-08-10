#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

"""
risk_utils.py

Canonical TrajRACE Phase-3 risk scoring.

For every transition unit (g_i.e, g_{i+1}.e) in an SBS S_j:

    endpoint risk:
        r_i^e

    long-stay association risk:
        r_i^s

    low-out-degree risk:
        r_i^d

    composite risk:
        r_i = lambda_e * r_i^e
            + lambda_s * r_i^s
            + lambda_d * r_i^d

The risk score is then mapped LOCALLY to a target privacy bucket using
fixed PUBLIC thresholds.

Important privacy properties
----------------------------
1. No global ranking or private-data quantile is used.
2. Bucket thresholds are fixed public parameters.
3. Raw risk scores and true buckets remain client/local data.
4. The public legal-successor graph is the only topology input.
5. Road-network distance queries are cached only in local memory and are
   never written back to a public cache.
"""

import math
from collections import OrderedDict, deque
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


# =============================================================================
# Generic helpers
# =============================================================================

def _clip01(value: float) -> float:
    value = float(value)

    if value <= 0.0:
        return 0.0

    if value >= 1.0:
        return 1.0

    return value


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0

    return float(
        sum(values) / len(values)
    )


def _safe_float(
    value: Any,
    default: float = 0.0,
) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


# =============================================================================
# Risk configuration validation
# =============================================================================

def validate_risk_config(
    risk_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Validate and normalize canonical risk parameters.
    """
    required = [
        "lambda_e",
        "lambda_s",
        "lambda_d",
        "sigma_e",
        "sigma_st",
        "delta_st_sec",
        "K",
        "theta_list",
    ]

    for key in required:
        if key not in risk_cfg:
            raise KeyError(
                f"Missing canonical risk parameter: {key}"
            )

    lambda_e = float(
        risk_cfg["lambda_e"]
    )

    lambda_s = float(
        risk_cfg["lambda_s"]
    )

    lambda_d = float(
        risk_cfg["lambda_d"]
    )

    if (
        lambda_e < 0
        or lambda_s < 0
        or lambda_d < 0
    ):
        raise ValueError(
            "Risk weights must be non-negative."
        )

    weight_sum = (
        lambda_e
        + lambda_s
        + lambda_d
    )

    if abs(
        weight_sum - 1.0
    ) > 1e-6:
        raise ValueError(
            "Canonical risk weights must sum to 1. "
            f"Got {weight_sum}."
        )

    sigma_e = float(
        risk_cfg["sigma_e"]
    )

    sigma_st = float(
        risk_cfg["sigma_st"]
    )

    delta_st_sec = float(
        risk_cfg["delta_st_sec"]
    )

    if sigma_e <= 0:
        raise ValueError(
            "sigma_e must be > 0"
        )

    if sigma_st <= 0:
        raise ValueError(
            "sigma_st must be > 0"
        )

    if delta_st_sec <= 0:
        raise ValueError(
            "delta_st_sec must be > 0"
        )

    K = int(
        risk_cfg["K"]
    )

    if K < 2:
        raise ValueError(
            "K must be >= 2"
        )

    theta_list = [
        float(x)
        for x in risk_cfg[
            "theta_list"
        ]
    ]

    if len(theta_list) != K - 1:
        raise ValueError(
            f"theta_list must contain K-1={K-1} thresholds; "
            f"got {len(theta_list)}"
        )

    for theta in theta_list:
        if not (
            0.0 <= theta <= 1.0
        ):
            raise ValueError(
                "Every risk threshold must lie in [0,1]."
            )

    for idx in range(
        len(theta_list) - 1
    ):
        if not (
            theta_list[idx]
            > theta_list[idx + 1]
        ):
            raise ValueError(
                "theta_list must be strictly decreasing."
            )

    distance_mode = str(
        risk_cfg.get(
            "distance_mode",
            "road_network_shortest_path",
        )
    )

    if (
        distance_mode
        != "road_network_shortest_path"
    ):
        raise ValueError(
            "Canonical TrajRACE currently requires "
            "distance_mode='road_network_shortest_path'."
        )

    road_distance_definition = str(
        risk_cfg.get(
            "road_distance_definition",
            "symmetric_shortest_segment_hops",
        )
    )

    if (
        road_distance_definition
        != "symmetric_shortest_segment_hops"
    ):
        raise ValueError(
            "Canonical implementation currently supports only "
            "road_distance_definition="
            "'symmetric_shortest_segment_hops'."
        )

    return {
        "lambda_e": lambda_e,
        "lambda_s": lambda_s,
        "lambda_d": lambda_d,

        "sigma_e": sigma_e,
        "sigma_st": sigma_st,
        "delta_st_sec": delta_st_sec,

        "distance_mode": distance_mode,
        "road_distance_definition": (
            road_distance_definition
        ),

        "K": K,
        "theta_list": theta_list,
    }


# =============================================================================
# Public road-network distance oracle
# =============================================================================

class SegmentDistanceOracle:
    """
    Local shortest-path distance oracle over the PUBLIC legal-successor graph.

    Distance definition
    -------------------
    Each road segment is one graph state.

    One legal next-road-segment move has unit distance 1.

    For two road segments a and b:

        d(a,b) = min(
            directed_hops(a -> b),
            directed_hops(b -> a)
        ).

    The caller supplies an upper bound derived from the observed SBS order.
    Since both queried segments belong to the same valid SBS, at least one
    direction has a path no longer than that bound.

    Privacy note
    ------------
    The LRU cache is ephemeral/local only. Its query keys are never persisted
    or exposed to the server.
    """

    def __init__(
        self,
        successor_cache: Dict[str, List[str]],
        max_cache_entries: int = 250000,
    ) -> None:

        self.successor_cache = (
            successor_cache
        )

        self.max_cache_entries = max(
            0,
            int(max_cache_entries),
        )

        self._cache: OrderedDict[
            Tuple[str, str],
            int,
        ] = OrderedDict()

        self.num_queries = 0
        self.num_cache_hits = 0
        self.num_bfs_calls = 0
        self.num_failed_queries = 0

    @staticmethod
    def _canonical_pair(
        a: Any,
        b: Any,
    ) -> Tuple[str, str]:

        a = str(a)
        b = str(b)

        if a <= b:
            return a, b

        return b, a

    def _cache_get(
        self,
        key: Tuple[str, str],
    ) -> Optional[int]:

        if key not in self._cache:
            return None

        value = self._cache.pop(
            key
        )

        self._cache[
            key
        ] = value

        self.num_cache_hits += 1

        return int(value)

    def _cache_put(
        self,
        key: Tuple[str, str],
        value: int,
    ) -> None:

        if self.max_cache_entries <= 0:
            return

        if key in self._cache:
            self._cache.pop(
                key
            )

        self._cache[
            key
        ] = int(value)

        while (
            len(self._cache)
            > self.max_cache_entries
        ):
            self._cache.popitem(
                last=False
            )

    def _directed_shortest_hops(
        self,
        source: Any,
        target: Any,
        max_depth: int,
    ) -> Optional[int]:

        source = str(source)
        target = str(target)

        if source == target:
            return 0

        max_depth = int(
            max_depth
        )

        if max_depth <= 0:
            return None

        self.num_bfs_calls += 1

        queue = deque(
            [(source, 0)]
        )

        visited: Set[str] = {
            source
        }

        while queue:

            current, depth = (
                queue.popleft()
            )

            if depth >= max_depth:
                continue

            successors = (
                self.successor_cache.get(
                    current,
                    [],
                )
            )

            next_depth = (
                depth + 1
            )

            for nxt in successors:

                nxt = str(nxt)

                if nxt == target:
                    return next_depth

                if nxt in visited:
                    continue

                visited.add(
                    nxt
                )

                queue.append(
                    (
                        nxt,
                        next_depth,
                    )
                )

        return None

    def symmetric_shortest_hops(
        self,
        segment_a: Any,
        segment_b: Any,
        upper_bound: int,
        preferred_source: Optional[Any] = None,
        preferred_target: Optional[Any] = None,
    ) -> int:
        """
        Compute exact symmetric shortest segment-hop distance up to a known
        valid upper bound.

        preferred_source -> preferred_target should follow the observed SBS
        travel order. Therefore that direction is expected to contain a valid
        path within upper_bound.
        """
        self.num_queries += 1

        segment_a = str(
            segment_a
        )

        segment_b = str(
            segment_b
        )

        if segment_a == segment_b:
            return 0

        upper_bound = int(
            upper_bound
        )

        if upper_bound <= 0:
            self.num_failed_queries += 1

            raise ValueError(
                "Non-identical segments received "
                f"non-positive upper_bound={upper_bound}"
            )

        key = self._canonical_pair(
            segment_a,
            segment_b,
        )

        cached = self._cache_get(
            key
        )

        if cached is not None:
            return cached

        if preferred_source is None:
            preferred_source = (
                segment_a
            )

        if preferred_target is None:
            preferred_target = (
                segment_b
            )

        preferred_source = str(
            preferred_source
        )

        preferred_target = str(
            preferred_target
        )

        best = (
            self._directed_shortest_hops(
                source=preferred_source,
                target=preferred_target,
                max_depth=upper_bound,
            )
        )

        # If a preferred forward path was found with distance d, the reverse
        # direction only needs to be searched up to d-1 to determine whether
        # a strictly shorter symmetric route exists.
        if best is not None:

            if best > 1:
                reverse = (
                    self._directed_shortest_hops(
                        source=preferred_target,
                        target=preferred_source,
                        max_depth=best - 1,
                    )
                )

                if (
                    reverse is not None
                    and reverse < best
                ):
                    best = reverse

        else:

            reverse = (
                self._directed_shortest_hops(
                    source=preferred_target,
                    target=preferred_source,
                    max_depth=upper_bound,
                )
            )

            if reverse is not None:
                best = reverse

        if best is None:
            self.num_failed_queries += 1

            raise RuntimeError(
                "Unable to find a public-road path between "
                f"{segment_a} and {segment_b} within "
                f"upper_bound={upper_bound}."
            )

        self._cache_put(
            key,
            int(best),
        )

        return int(best)

    def summary(
        self,
    ) -> Dict[str, Any]:

        hit_rate = (
            self.num_cache_hits
            / self.num_queries
            if self.num_queries
            else 0.0
        )

        return {
            "distance_definition": (
                "symmetric_shortest_segment_hops"
            ),
            "num_queries": int(
                self.num_queries
            ),
            "num_cache_hits": int(
                self.num_cache_hits
            ),
            "cache_hit_rate": float(
                hit_rate
            ),
            "num_bfs_calls": int(
                self.num_bfs_calls
            ),
            "num_failed_queries": int(
                self.num_failed_queries
            ),
            "current_cache_entries": int(
                len(self._cache)
            ),
            "max_cache_entries": int(
                self.max_cache_entries
            ),
        }


# =============================================================================
# Stay-anchor construction
# =============================================================================

def detect_stay_anchor_positions(
    event_record: Dict[str, Any],
    delta_st_sec: float,
) -> List[int]:
    """
    Eq. (5):

    If adjacent road-segment timestamps satisfy

        tau_{i+1} - tau_i >= Delta_st,

    both road segments become stay anchors.

    Returns 0-based positions inside the current SBS.
    """
    transitions = event_record.get(
        "transition_events",
        [],
    )

    anchor_positions: Set[int] = (
        set()
    )

    for idx, transition in enumerate(
        transitions
    ):

        delta_t = transition.get(
            "delta_t",
            None,
        )

        if delta_t is None:
            continue

        delta_t = _safe_float(
            delta_t,
            0.0,
        )

        if (
            delta_t
            >= float(delta_st_sec)
        ):
            anchor_positions.add(
                idx
            )

            anchor_positions.add(
                idx + 1
            )

    return sorted(
        anchor_positions
    )


# =============================================================================
# Risk components
# =============================================================================

def _endpoint_risk(
    segments: Sequence[Any],
    current_index: int,
    sigma_e: float,
    distance_oracle: SegmentDistanceOracle,
) -> Tuple[float, int, int]:

    current_index = int(
        current_index
    )

    last_index = (
        len(segments) - 1
    )

    current_segment = (
        segments[current_index]
    )

    start_segment = (
        segments[0]
    )

    end_segment = (
        segments[-1]
    )

    # start -> current follows observed SBS order
    d_start = (
        distance_oracle.symmetric_shortest_hops(
            segment_a=current_segment,
            segment_b=start_segment,
            upper_bound=max(
                1,
                current_index,
            )
            if current_index > 0
            else 0,
            preferred_source=start_segment,
            preferred_target=current_segment,
        )
        if current_index > 0
        else 0
    )

    # current -> end follows observed SBS order
    remaining = (
        last_index
        - current_index
    )

    d_end = (
        distance_oracle.symmetric_shortest_hops(
            segment_a=current_segment,
            segment_b=end_segment,
            upper_bound=max(
                1,
                remaining,
            ),
            preferred_source=current_segment,
            preferred_target=end_segment,
        )
        if remaining > 0
        else 0
    )

    sigma_e = max(
        float(sigma_e),
        1e-12,
    )

    start_score = math.exp(
        -float(d_start)
        / sigma_e
    )

    end_score = math.exp(
        -float(d_end)
        / sigma_e
    )

    risk = max(
        start_score,
        end_score,
    )

    return (
        _clip01(risk),
        int(d_start),
        int(d_end),
    )


def _stay_risk(
    segments: Sequence[Any],
    current_index: int,
    stay_anchor_positions: Sequence[int],
    sigma_st: float,
    distance_oracle: SegmentDistanceOracle,
) -> Tuple[float, Optional[int]]:

    if not stay_anchor_positions:
        return 0.0, None

    current_segment = (
        segments[current_index]
    )

    min_distance: Optional[int] = (
        None
    )

    for anchor_index in (
        stay_anchor_positions
    ):

        anchor_index = int(
            anchor_index
        )

        anchor_segment = (
            segments[
                anchor_index
            ]
        )

        if (
            anchor_index
            == current_index
        ):
            min_distance = 0
            break

        upper_bound = abs(
            anchor_index
            - current_index
        )

        if current_index < anchor_index:

            preferred_source = (
                current_segment
            )

            preferred_target = (
                anchor_segment
            )

        else:

            preferred_source = (
                anchor_segment
            )

            preferred_target = (
                current_segment
            )

        distance = (
            distance_oracle.symmetric_shortest_hops(
                segment_a=current_segment,
                segment_b=anchor_segment,
                upper_bound=upper_bound,
                preferred_source=(
                    preferred_source
                ),
                preferred_target=(
                    preferred_target
                ),
            )
        )

        if (
            min_distance is None
            or distance < min_distance
        ):
            min_distance = int(
                distance
            )

    if min_distance is None:
        return 0.0, None

    sigma_st = max(
        float(sigma_st),
        1e-12,
    )

    risk = math.exp(
        -float(min_distance)
        / sigma_st
    )

    return (
        _clip01(risk),
        int(min_distance),
    )


def _degree_risk(
    current_segment: Any,
    successor_cache: Dict[
        str,
        List[str],
    ],
) -> Tuple[float, int]:

    current_segment = str(
        current_segment
    )

    successors = (
        successor_cache.get(
            current_segment,
            [],
        )
    )

    candidate_size = len(
        successors
    )

    if candidate_size <= 0:
        raise RuntimeError(
            "Current road segment has no public legal successor: "
            f"{current_segment}"
        )

    risk = (
        1.0
        / float(candidate_size)
    )

    return (
        _clip01(risk),
        int(candidate_size),
    )


# =============================================================================
# Fixed-public-threshold bucket mapping
# =============================================================================

def assign_target_bucket(
    risk_score: float,
    theta_list: Sequence[float],
    K: int,
) -> int:
    """
    Eq. (9): fixed PUBLIC threshold mapping.

    theta_list must be strictly decreasing.

    Example K=3:
        r >= theta_1                 -> 1
        theta_2 <= r < theta_1      -> 2
        r < theta_2                  -> 3
    """
    risk_score = float(
        risk_score
    )

    for idx, threshold in enumerate(
        theta_list
    ):
        if (
            risk_score
            >= float(threshold)
        ):
            return idx + 1

    return int(K)


# =============================================================================
# One-SBS risk computation
# =============================================================================

def compute_transition_risks_for_record(
    event_record: Dict[str, Any],
    successor_cache: Dict[
        str,
        List[str],
    ],
    risk_cfg: Dict[str, Any],
    distance_oracle: SegmentDistanceOracle,
) -> List[Dict[str, Any]]:
    """
    Compute canonical risk scores and fixed-threshold target buckets
    for every transition in one SBS.
    """
    cfg = validate_risk_config(
        risk_cfg
    )

    traj_id = event_record.get(
        "traj_id"
    )

    sbs_id = event_record.get(
        "sbs_id"
    )

    segments = event_record.get(
        "segments",
        [],
    )

    transitions = event_record.get(
        "transition_events",
        [],
    )

    if not isinstance(
        segments,
        list,
    ):
        raise ValueError(
            f"segments must be a list: sbs_id={sbs_id}"
        )

    if len(segments) < 2:
        raise ValueError(
            f"SBS contains fewer than two segments: sbs_id={sbs_id}"
        )

    if len(transitions) != (
        len(segments) - 1
    ):
        raise ValueError(
            "Transition/SBS length mismatch: "
            f"sbs_id={sbs_id}, "
            f"segments={len(segments)}, "
            f"transitions={len(transitions)}"
        )

    stay_anchor_positions = (
        detect_stay_anchor_positions(
            event_record=event_record,
            delta_st_sec=cfg[
                "delta_st_sec"
            ],
        )
    )

    out: List[
        Dict[str, Any]
    ] = []

    for local_index, transition in enumerate(
        transitions
    ):

        u = str(
            transition.get(
                "u"
            )
        )

        v = str(
            transition.get(
                "v"
            )
        )

        expected_u = str(
            segments[
                local_index
            ]
        )

        expected_v = str(
            segments[
                local_index + 1
            ]
        )

        if (
            u != expected_u
            or v != expected_v
        ):
            raise RuntimeError(
                "Transition does not match SBS segment order: "
                f"sbs_id={sbs_id}, "
                f"local_index={local_index}"
            )

        public_successors = (
            successor_cache.get(
                u,
                [],
            )
        )

        # Hard public-domain check.
        if v not in public_successors:
            raise RuntimeError(
                "True successor is not contained in the PUBLIC "
                f"legal-successor domain N(u): u={u}, v={v}"
            )

        (
            risk_endpoint,
            d_start,
            d_end,
        ) = _endpoint_risk(
            segments=segments,
            current_index=local_index,
            sigma_e=cfg[
                "sigma_e"
            ],
            distance_oracle=(
                distance_oracle
            ),
        )

        (
            risk_stay,
            d_stay,
        ) = _stay_risk(
            segments=segments,
            current_index=local_index,
            stay_anchor_positions=(
                stay_anchor_positions
            ),
            sigma_st=cfg[
                "sigma_st"
            ],
            distance_oracle=(
                distance_oracle
            ),
        )

        (
            risk_degree,
            candidate_size,
        ) = _degree_risk(
            current_segment=u,
            successor_cache=(
                successor_cache
            ),
        )

        risk_score = (
            cfg["lambda_e"]
            * risk_endpoint
            + cfg["lambda_s"]
            * risk_stay
            + cfg["lambda_d"]
            * risk_degree
        )

        risk_score = _clip01(
            risk_score
        )

        target_bucket = (
            assign_target_bucket(
                risk_score=(
                    risk_score
                ),
                theta_list=cfg[
                    "theta_list"
                ],
                K=cfg["K"],
            )
        )

        delta_t = transition.get(
            "delta_t",
            None,
        )

        item = {
            # -------------------------------------------------------------
            # Local identifiers only.
            # -------------------------------------------------------------
            "traj_id": traj_id,
            "sbs_id": sbs_id,
            "sbs_index": event_record.get(
                "sbs_index"
            ),
            "num_sbs": event_record.get(
                "num_sbs"
            ),

            # -------------------------------------------------------------
            # True local transition.
            # -------------------------------------------------------------
            "t": int(
                transition.get(
                    "t",
                    local_index + 1,
                )
            ),
            "global_pos": transition.get(
                "global_pos"
            ),

            "u": u,
            "v": v,

            "cnt": int(
                event_record[
                    "count_event"
                ]["cnt"]
            ),

            "delta_t": delta_t,

            # -------------------------------------------------------------
            # Canonical Eq. (4)-(8) risk components.
            # -------------------------------------------------------------
            "risk_endpoint": float(
                risk_endpoint
            ),
            "risk_stay": float(
                risk_stay
            ),
            "risk_degree": float(
                risk_degree
            ),

            "risk_score": float(
                risk_score
            ),

            # -------------------------------------------------------------
            # Eq. (9) fixed-public-threshold target bucket.
            # -------------------------------------------------------------
            "target_bucket": int(
                target_bucket
            ),

            # -------------------------------------------------------------
            # Diagnostics used only for local audit/rebuttal analysis.
            # -------------------------------------------------------------
            "endpoint_distance_start": int(
                d_start
            ),
            "endpoint_distance_end": int(
                d_end
            ),

            "stay_min_distance": (
                None
                if d_stay is None
                else int(d_stay)
            ),

            "num_stay_anchor_segments": int(
                len(
                    stay_anchor_positions
                )
            ),

            "public_successor_domain_size": int(
                candidate_size
            ),

            # -------------------------------------------------------------
            # Compatibility aliases for later migration of old scripts.
            # These remain LOCAL and may be removed after Phase 4 is frozen.
            # -------------------------------------------------------------
            "phi_endpoint": float(
                risk_endpoint
            ),
            "phi_stay": float(
                risk_stay
            ),
            "phi_deg": float(
                risk_degree
            ),
            "R_t": float(
                risk_score
            ),
            "b_t": int(
                target_bucket
            ),
        }

        out.append(
            item
        )

    return out


# =============================================================================
# Risk/bucket summary
# =============================================================================

def summarize_target_buckets(
    risk_items: Sequence[
        Dict[str, Any]
    ],
    K: int,
) -> Dict[str, Any]:
    """
    Summarize ALREADY ASSIGNED fixed-threshold buckets.

    This function never changes bucket assignments.
    """
    K = int(
        K
    )

    total = len(
        risk_items
    )

    result: Dict[
        str,
        Any,
    ] = {}

    for bucket in range(
        1,
        K + 1,
    ):

        subset = [
            item
            for item
            in risk_items
            if int(
                item.get(
                    "target_bucket",
                    item.get(
                        "b_t",
                        0,
                    ),
                )
            )
            == bucket
        ]

        count = len(
            subset
        )

        result[
            str(bucket)
        ] = {
            "count": int(
                count
            ),

            "fraction": (
                float(
                    count / total
                )
                if total
                else 0.0
            ),

            "avg_risk_score": _mean(
                [
                    _safe_float(
                        x.get(
                            "risk_score",
                            x.get(
                                "R_t",
                                0.0,
                            ),
                        )
                    )
                    for x in subset
                ]
            ),

            "avg_risk_endpoint": _mean(
                [
                    _safe_float(
                        x.get(
                            "risk_endpoint",
                            0.0,
                        )
                    )
                    for x in subset
                ]
            ),

            "avg_risk_stay": _mean(
                [
                    _safe_float(
                        x.get(
                            "risk_stay",
                            0.0,
                        )
                    )
                    for x in subset
                ]
            ),

            "avg_risk_degree": _mean(
                [
                    _safe_float(
                        x.get(
                            "risk_degree",
                            0.0,
                        )
                    )
                    for x in subset
                ]
            ),
        }

    return result