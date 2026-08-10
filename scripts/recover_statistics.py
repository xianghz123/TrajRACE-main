#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

"""
recover_statistics.py

Canonical TrajRACE Phase-5 server-side statistical recovery.

This script performs risk-aware recovery only.

Recovery paths
--------------
A. Main / paper recovery:
       context bucket-mixture recovery

       pi_u(k) = P(K=k | U=u)

       Qbar_u = sum_k pi_u(k) Q_u^(k)

       recover P(V | U=u) from Y.

B. Diagnostic Plan-B recovery:
       joint latent recovery

       latent state  : (V, K)
       observed state: (Y, K_noisy)

       A_u[(y,k_tilde),(v,k)]
         = Q_u^(k)(v,y) * M(k,k_tilde)

       recover P(V,K | U=u)
       and marginalize over K.

C. Recovery-dependence diagnostic:
       I(K;V | U)
       weighted conditional total variation.

Important privacy boundary
--------------------------
The actual server recovery uses ONLY privatized reports and PUBLIC domains.

True event/risk records are accessed ONLY by the diagnostic section to
measure recovery error and K-V dependence for rebuttal/ablation analysis.

They are never inputs to the released recovery estimator itself.
"""

import argparse
import itertools
import json
import math
import os
import sys
import time

from collections import Counter, defaultdict
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np


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


# =============================================================================
# Logging
# =============================================================================

def fmt_sec(
    sec: float,
) -> str:

    if sec < 60:
        return f"{sec:.1f}s"

    minutes = int(
        sec // 60
    )

    seconds = (
        sec
        - 60
        * minutes
    )

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
            "Canonical TrajRACE Phase-5 "
            "risk-aware statistical recovery."
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
        "--joint_min_reports",
        type=int,
        default=20,
        help=(
            "Minimum number of reports for a public context u "
            "before solving the full joint (V,K) system. "
            "Sparser contexts use context-mixture backoff."
        ),
    )

    parser.add_argument(
        "--ridge",
        type=float,
        default=1e-8,
        help=(
            "Small numerical ridge used by constrained "
            "least-squares recovery."
        ),
    )

    parser.add_argument(
        "--skip_joint",
        action="store_true",
        help=(
            "Skip Plan-B joint latent recovery. "
            "Normally leave this disabled for the Recovery Gate."
        ),
    )

    return parser.parse_args()


# =============================================================================
# Canonical configuration
# =============================================================================

def extract_recovery_cfg(
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
        "tau_shrinkage",
        "recovery_mode",
    ]

    for key in required:

        if key not in exp_cfg:

            raise KeyError(
                f"Missing recovery config: {key}"
            )

    K = int(
        exp_cfg[
            "K"
        ]
    )

    eps_event_list = [
        float(x)
        for x in exp_cfg[
            "eps_event_list"
        ]
    ]

    if len(
        eps_event_list
    ) != K:

        raise ValueError(
            "eps_event_list length must equal K."
        )

    cfg = {
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

        "eps_event_list": (
            eps_event_list
        ),

        "K": K,

        "L_max": int(
            exp_cfg[
                "L_max"
            ]
        ),

        "tau_shrinkage": float(
            exp_cfg[
                "tau_shrinkage"
            ]
        ),

        "recovery_mode": str(
            exp_cfg[
                "recovery_mode"
            ]
        ),

        "bucket_recovery_solver": str(
            exp_cfg.get(
                "bucket_recovery_solver",
                "constrained_least_squares",
            )
        ),

        "enable_recovery_dependence_diagnostic": bool(
            exp_cfg.get(
                "enable_recovery_dependence_diagnostic",
                True,
            )
        ),

        "use_global_backoff": bool(
            exp_cfg.get(
                "use_global_backoff",
                True,
            )
        ),
    }

    if (
        cfg[
            "recovery_mode"
        ]
        != "context_bucket_mixture"
    ):

        raise ValueError(
            "Current configured main recovery must be "
            "'context_bucket_mixture'. "
            "Joint recovery is evaluated as the Plan-B diagnostic."
        )

    if (
        cfg[
            "bucket_recovery_solver"
        ]
        != "constrained_least_squares"
    ):

        raise ValueError(
            "Canonical recovery requires "
            "bucket_recovery_solver="
            "'constrained_least_squares'."
        )

    return cfg


# =============================================================================
# RR channels
# =============================================================================

def rr_params(
    epsilon: float,
    domain_size: int,
) -> Tuple[
    float,
    float,
]:

    d = int(
        domain_size
    )

    if d <= 0:

        raise ValueError(
            "RR domain size must be positive."
        )

    if d == 1:

        return 1.0, 0.0

    exp_eps = math.exp(
        float(
            epsilon
        )
    )

    p = (
        exp_eps
        / (
            exp_eps
            + d
            - 1
        )
    )

    q = (
        1.0
        / (
            exp_eps
            + d
            - 1
        )
    )

    return (
        float(p),
        float(q),
    )


def symmetric_rr_channel(
    epsilon: float,
    domain_size: int,
) -> np.ndarray:

    d = int(
        domain_size
    )

    p, q = rr_params(
        epsilon,
        d,
    )

    matrix = np.full(
        (
            d,
            d,
        ),
        q,
        dtype=float,
    )

    np.fill_diagonal(
        matrix,
        p,
    )

    return matrix


# =============================================================================
# Constrained probability recovery
# =============================================================================

def project_simplex(
    vector: np.ndarray,
) -> np.ndarray:
    """
    Euclidean projection onto the probability simplex.
    """

    x = np.asarray(
        vector,
        dtype=float,
    )

    if x.size == 0:
        return x

    u = np.sort(
        x
    )[::-1]

    cssv = (
        np.cumsum(u)
        - 1.0
    )

    indices = np.arange(
        1,
        x.size + 1,
        dtype=float,
    )

    condition = (
        u
        - cssv
        / indices
        > 0
    )

    if not np.any(
        condition
    ):

        return np.full(
            x.shape,
            1.0
            / x.size,
        )

    rho = np.nonzero(
        condition
    )[0][-1]

    theta = (
        cssv[rho]
        / float(
            rho + 1
        )
    )

    projected = np.maximum(
        x - theta,
        0.0,
    )

    total = float(
        projected.sum()
    )

    if total <= 0:

        return np.full(
            x.shape,
            1.0
            / x.size,
        )

    return (
        projected
        / total
    )


def constrained_distribution_ls(
    channel: np.ndarray,
    observed_frequency: np.ndarray,
    ridge: float,
) -> np.ndarray:
    """
    Non-negative constrained least-squares recovery followed by
    simplex normalization.
    """

    A = np.asarray(
        channel,
        dtype=float,
    )

    b = np.asarray(
        observed_frequency,
        dtype=float,
    )

    n_latent = (
        A.shape[1]
    )

    ridge = max(
        0.0,
        float(
            ridge
        ),
    )

    if ridge > 0:

        sqrt_ridge = math.sqrt(
            ridge
        )

        A_aug = np.vstack(
            [
                A,
                sqrt_ridge
                * np.eye(
                    n_latent
                ),
            ]
        )

        b_aug = np.concatenate(
            [
                b,
                np.zeros(
                    n_latent,
                    dtype=float,
                ),
            ]
        )

    else:

        A_aug = A
        b_aug = b

    try:

        from scipy.optimize import (
            lsq_linear
        )

        result = lsq_linear(
            A_aug,
            b_aug,
            bounds=(
                0.0,
                np.inf,
            ),
            lsmr_tol="auto",
            verbose=0,
        )

        estimate = (
            result.x
        )

    except Exception:

        estimate, *_ = (
            np.linalg.lstsq(
                A_aug,
                b_aug,
                rcond=None,
            )
        )

        estimate = np.maximum(
            estimate,
            0.0,
        )

    return project_simplex(
        estimate
    )


# =============================================================================
# Scalar start/count recovery
# =============================================================================

def recover_sparse_symmetric_rr(
    observed: Counter,
    public_domain_size: int,
    epsilon: float,
) -> Dict[str, Any]:
    """
    Sparse equivalent of the clipped symmetric-RR inversion.

    For an unobserved public-domain item:
        observed frequency = 0,
    hence its raw inverse estimate is negative when q>0 and becomes
    zero after non-negative clipping.

    Therefore the complete 3.1M-road-segment start domain does not need
    to be materialized in the output JSON.
    """

    n = int(
        sum(
            observed.values()
        )
    )

    d = int(
        public_domain_size
    )

    p, q = rr_params(
        epsilon,
        d,
    )

    positive: Dict[
        str,
        float,
    ] = {}

    if n <= 0:

        return {
            "domain_size": d,
            "num_reports": 0,
            "support_size_after_projection": 0,
            "distribution": {},
        }

    denominator = (
        p - q
    )

    if abs(
        denominator
    ) < 1e-15:

        for key, count in (
            observed.items()
        ):

            positive[
                str(key)
            ] = float(
                count
                / n
            )

    else:

        for key, count in (
            observed.items()
        ):

            freq = (
                float(count)
                / n
            )

            estimate = (
                freq
                - q
            ) / denominator

            if estimate > 0:

                positive[
                    str(key)
                ] = float(
                    estimate
                )

    total = sum(
        positive.values()
    )

    if total <= 0:

        positive = {
            str(key): (
                float(count)
                / n
            )
            for key, count
            in observed.items()
        }

        total = sum(
            positive.values()
        )

    distribution = {
        key: float(
            value
            / total
        )
        for key, value
        in positive.items()
    }

    return {
        "domain_size": d,

        "num_reports": n,

        "support_size_after_projection": int(
            len(
                distribution
            )
        ),

        "rr_p": p,

        "rr_q": q,

        "distribution": (
            distribution
        ),
    }


def recover_dense_symmetric_rr(
    observed: Counter,
    domain: Sequence[Any],
    epsilon: float,
    ridge: float,
) -> Dict[str, Any]:

    domain = list(
        domain
    )

    d = len(
        domain
    )

    n = int(
        sum(
            observed.values()
        )
    )

    if n <= 0:

        return {
            "domain_size": d,
            "num_reports": 0,
            "distribution": {},
        }

    channel = (
        symmetric_rr_channel(
            epsilon,
            d,
        )
    )

    obs_freq = np.array(
        [
            observed.get(
                x,
                0,
            )
            / n
            for x in domain
        ],
        dtype=float,
    )

    estimate = (
        constrained_distribution_ls(
            channel,
            obs_freq,
            ridge,
        )
    )

    return {
        "domain_size": d,

        "num_reports": n,

        "distribution": {
            str(
                domain[i]
            ): float(
                estimate[i]
            )
            for i in range(d)
        },
    }


# =============================================================================
# Server-report aggregation
# =============================================================================

def aggregate_server_reports(
    report_path: str,
) -> Dict[str, Any]:

    start_counter = Counter()
    count_counter = Counter()

    per_u_y = defaultdict(
        Counter
    )

    per_u_noisy_bucket = defaultdict(
        Counter
    )

    per_u_joint_obs = defaultdict(
        Counter
    )

    global_noisy_bucket = Counter()

    type_counts = Counter()

    for report in iter_jsonl(
        report_path
    ):

        event_type = report[
            "event_type"
        ]

        type_counts[
            event_type
        ] += 1

        if event_type == "start":

            start_counter[
                str(
                    report[
                        "x_noisy"
                    ]
                )
            ] += 1

        elif event_type == "count":

            count_counter[
                int(
                    report[
                        "x_noisy"
                    ]
                )
            ] += 1

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

            k_noisy = int(
                report[
                    "k_noisy"
                ]
            )

            per_u_y[
                u
            ][
                y
            ] += 1

            per_u_noisy_bucket[
                u
            ][
                k_noisy
            ] += 1

            per_u_joint_obs[
                u
            ][
                (
                    y,
                    k_noisy,
                )
            ] += 1

            global_noisy_bucket[
                k_noisy
            ] += 1

        else:

            raise RuntimeError(
                f"Unknown server report type: "
                f"{event_type}"
            )

    return {
        "start_counter": (
            start_counter
        ),

        "count_counter": (
            count_counter
        ),

        "per_u_y": (
            per_u_y
        ),

        "per_u_noisy_bucket": (
            per_u_noisy_bucket
        ),

        "per_u_joint_obs": (
            per_u_joint_obs
        ),

        "global_noisy_bucket": (
            global_noisy_bucket
        ),

        "type_counts": (
            type_counts
        ),
    }


# =============================================================================
# Bucket-mixture recovery
# =============================================================================

def recover_bucket_mix(
    noisy_counter: Counter,
    K: int,
    epsilon_bucket: float,
    ridge: float,
) -> Dict[int, float]:

    domain = list(
        range(
            1,
            K + 1,
        )
    )

    n = int(
        sum(
            noisy_counter.values()
        )
    )

    if n <= 0:

        return {
            k: (
                1.0
                / K
            )
            for k in domain
        }

    channel = (
        symmetric_rr_channel(
            epsilon_bucket,
            K,
        )
    )

    obs_freq = np.array(
        [
            noisy_counter.get(
                k,
                0,
            )
            / n
            for k in domain
        ],
        dtype=float,
    )

    estimate = (
        constrained_distribution_ls(
            channel,
            obs_freq,
            ridge,
        )
    )

    return {
        k: float(
            estimate[
                k - 1
            ]
        )
        for k in domain
    }


def shrink_bucket_mix(
    local_mix: Dict[int, float],
    global_mix: Dict[int, float],
    num_reports: int,
    tau: float,
    K: int,
) -> Tuple[
    Dict[int, float],
    float,
]:

    denominator = (
        num_reports
        + tau
    )

    alpha = (
        float(
            num_reports
        )
        / denominator
        if denominator > 0
        else 0.0
    )

    mixed = {
        k: (
            alpha
            * float(
                local_mix.get(
                    k,
                    0.0,
                )
            )
            + (
                1.0
                - alpha
            )
            * float(
                global_mix.get(
                    k,
                    0.0,
                )
            )
        )
        for k in range(
            1,
            K + 1,
        )
    }

    total = sum(
        mixed.values()
    )

    if total <= 0:

        mixed = {
            k: (
                1.0
                / K
            )
            for k in range(
                1,
                K + 1,
            )
        }

    else:

        mixed = {
            k: float(
                value
                / total
            )
            for k, value
            in mixed.items()
        }

    return (
        mixed,
        float(alpha),
    )


# =============================================================================
# Main context bucket-mixture recovery
# =============================================================================

def recover_context_bucket_mixture(
    aggregate: Dict[str, Any],
    successor_cache: Dict[
        str,
        List[str],
    ],
    cfg: Dict[str, Any],
    ridge: float,
) -> Tuple[
    Dict[str, Any],
    Dict[str, Any],
]:

    global_mix = (
        recover_bucket_mix(
            noisy_counter=(
                aggregate[
                    "global_noisy_bucket"
                ]
            ),
            K=cfg[
                "K"
            ],
            epsilon_bucket=(
                cfg[
                    "eps_bucket"
                ]
            ),
            ridge=ridge,
        )
    )

    transition_result: Dict[
        str,
        Any,
    ] = {}

    bucket_result: Dict[
        str,
        Any,
    ] = {
        "global_mix": {
            str(k): float(v)
            for k, v
            in global_mix.items()
        },

        "per_u_mix": {},
    }

    for u, y_counter in (
        aggregate[
            "per_u_y"
        ].items()
    ):

        if u not in successor_cache:

            raise RuntimeError(
                "Server report contains unknown public "
                f"context u={u}"
            )

        domain = list(
            successor_cache[
                u
            ]
        )

        if not domain:

            raise RuntimeError(
                f"Public N(u) is empty for "
                f"reported context u={u}"
            )

        N_u = int(
            sum(
                y_counter.values()
            )
        )

        local_mix = (
            recover_bucket_mix(
                noisy_counter=(
                    aggregate[
                        "per_u_noisy_bucket"
                    ][
                        u
                    ]
                ),
                K=cfg[
                    "K"
                ],
                epsilon_bucket=(
                    cfg[
                        "eps_bucket"
                    ]
                ),
                ridge=ridge,
            )
        )

        if cfg[
            "use_global_backoff"
        ]:

            pi_u, alpha = (
                shrink_bucket_mix(
                    local_mix=(
                        local_mix
                    ),

                    global_mix=(
                        global_mix
                    ),

                    num_reports=(
                        N_u
                    ),

                    tau=cfg[
                        "tau_shrinkage"
                    ],

                    K=cfg[
                        "K"
                    ],
                )
            )

        else:

            pi_u = (
                local_mix
            )

            alpha = 1.0

        d = len(
            domain
        )

        pbar = 0.0
        qbar = 0.0

        for k in range(
            1,
            cfg["K"] + 1,
        ):

            p_k, q_k = (
                rr_params(
                    cfg[
                        "eps_event_list"
                    ][
                        k - 1
                    ],
                    d,
                )
            )

            weight = float(
                pi_u[
                    k
                ]
            )

            pbar += (
                weight
                * p_k
            )

            qbar += (
                weight
                * q_k
            )

        channel = np.full(
            (
                d,
                d,
            ),
            qbar,
            dtype=float,
        )

        np.fill_diagonal(
            channel,
            pbar,
        )

        obs_freq = np.array(
            [
                y_counter.get(
                    v,
                    0,
                )
                / N_u
                for v in domain
            ],
            dtype=float,
        )

        estimate = (
            constrained_distribution_ls(
                channel,
                obs_freq,
                ridge,
            )
        )

        transition_result[
            u
        ] = {
            "domain_size": int(
                d
            ),

            "num_reports": (
                N_u
            ),

            "pbar": float(
                pbar
            ),

            "qbar": float(
                qbar
            ),

            "distribution": {
                str(
                    domain[i]
                ): float(
                    estimate[i]
                )
                for i in range(d)
            },
        }

        bucket_result[
            "per_u_mix"
        ][
            u
        ] = {
            "num_reports": (
                N_u
            ),

            "alpha_u": float(
                alpha
            ),

            "pi_u_local": {
                str(k): float(
                    local_mix[
                        k
                    ]
                )
                for k in range(
                    1,
                    cfg["K"] + 1,
                )
            },

            "pi_u_shrunk": {
                str(k): float(
                    pi_u[
                        k
                    ]
                )
                for k in range(
                    1,
                    cfg["K"] + 1,
                )
            },
        }

    return (
        transition_result,
        bucket_result,
    )


# =============================================================================
# Joint latent (V,K) recovery
# =============================================================================

def build_joint_channel(
    domain: Sequence[str],
    cfg: Dict[str, Any],
) -> np.ndarray:

    domain = list(
        domain
    )

    d = len(
        domain
    )

    K = cfg[
        "K"
    ]

    bucket_channel = (
        symmetric_rr_channel(
            cfg[
                "eps_bucket"
            ],
            K,
        )
    )

    dim = (
        d
        * K
    )

    A = np.zeros(
        (
            dim,
            dim,
        ),
        dtype=float,
    )

    def latent_index(
        v_idx: int,
        k_idx: int,
    ) -> int:

        return (
            v_idx
            * K
            + k_idx
        )

    def observed_index(
        y_idx: int,
        noisy_k_idx: int,
    ) -> int:

        return (
            y_idx
            * K
            + noisy_k_idx
        )

    for k_idx in range(
        K
    ):

        successor_channel = (
            symmetric_rr_channel(
                cfg[
                    "eps_event_list"
                ][
                    k_idx
                ],
                d,
            )
        )

        for v_idx in range(
            d
        ):

            col = latent_index(
                v_idx,
                k_idx,
            )

            for y_idx in range(
                d
            ):

                q_prob = (
                    successor_channel[
                        y_idx,
                        v_idx
                    ]
                )

                for noisy_k_idx in (
                    range(K)
                ):

                    m_prob = (
                        bucket_channel[
                            noisy_k_idx,
                            k_idx
                        ]
                    )

                    row = observed_index(
                        y_idx,
                        noisy_k_idx,
                    )

                    A[
                        row,
                        col
                    ] = (
                        q_prob
                        * m_prob
                    )

    return A


def recover_joint_latent(
    aggregate: Dict[str, Any],
    successor_cache: Dict[
        str,
        List[str],
    ],
    context_result: Dict[
        str,
        Any,
    ],
    cfg: Dict[str, Any],
    joint_min_reports: int,
    ridge: float,
) -> Dict[str, Any]:

    result: Dict[
        str,
        Any,
    ] = {}

    K = cfg[
        "K"
    ]

    for u, joint_counter in (
        aggregate[
            "per_u_joint_obs"
        ].items()
    ):

        domain = list(
            successor_cache[
                u
            ]
        )

        d = len(
            domain
        )

        N_u = int(
            sum(
                joint_counter.values()
            )
        )

        if (
            N_u
            < int(
                joint_min_reports
            )
        ):

            result[
                u
            ] = {
                "num_reports": (
                    N_u
                ),

                "domain_size": (
                    d
                ),

                "source": (
                    "context_backoff_sparse"
                ),

                "distribution": (
                    context_result[
                        u
                    ][
                        "distribution"
                    ]
                ),
            }

            continue

        A = (
            build_joint_channel(
                domain,
                cfg,
            )
        )

        obs = np.zeros(
            d * K,
            dtype=float,
        )

        v_index = {
            str(v): idx
            for idx, v
            in enumerate(
                domain
            )
        }

        for (
            y,
            noisy_k
        ), count in (
            joint_counter.items()
        ):

            y = str(y)

            if y not in v_index:

                raise RuntimeError(
                    "Observed privatized successor "
                    "outside fixed public N(u)."
                )

            y_idx = (
                v_index[
                    y
                ]
            )

            noisy_k_idx = (
                int(
                    noisy_k
                )
                - 1
            )

            row = (
                y_idx
                * K
                + noisy_k_idx
            )

            obs[
                row
            ] += (
                count
                / N_u
            )

        latent = (
            constrained_distribution_ls(
                A,
                obs,
                ridge,
            )
        )

        v_marginal = np.zeros(
            d,
            dtype=float,
        )

        k_marginal = np.zeros(
            K,
            dtype=float,
        )

        joint_distribution = {}

        for v_idx, v in enumerate(
            domain
        ):

            for k_idx in range(
                K
            ):

                idx = (
                    v_idx
                    * K
                    + k_idx
                )

                probability = float(
                    latent[
                        idx
                    ]
                )

                v_marginal[
                    v_idx
                ] += probability

                k_marginal[
                    k_idx
                ] += probability

                if probability > 0:

                    joint_distribution[
                        f"{v}|||{k_idx + 1}"
                    ] = probability

        v_marginal = (
            project_simplex(
                v_marginal
            )
        )

        k_marginal = (
            project_simplex(
                k_marginal
            )
        )

        result[
            u
        ] = {
            "num_reports": (
                N_u
            ),

            "domain_size": (
                d
            ),

            "source": (
                "joint_constrained_least_squares"
            ),

            "distribution": {
                str(
                    domain[i]
                ): float(
                    v_marginal[i]
                )
                for i in range(d)
            },

            "latent_bucket_marginal": {
                str(k + 1): float(
                    k_marginal[k]
                )
                for k in range(K)
            },

            "joint_distribution": (
                joint_distribution
            ),
        }

    return result


# =============================================================================
# Recompute true execution K for LOCAL diagnostics
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

    feasible = []

    for k in range(
        1,
        cfg["K"] + 1,
    ):

        current_cost = (
            cfg[
                "eps_event_list"
            ][
                k - 1
            ]
            + cfg[
                "eps_bucket"
            ]
        )

        needed = (
            current_cost
            + future_after_current
            * min_cost
        )

        if (
            needed
            <= remaining_budget
            + 1e-12
        ):

            feasible.append(
                k
            )

    return feasible


def choose_exec_bucket(
    target_bucket: int,
    feasible: Sequence[int],
) -> int:

    feasible = sorted(
        set(
            int(x)
            for x in feasible
        )
    )

    if target_bucket in feasible:

        return int(
            target_bucket
        )

    candidates = [
        k
        for k in feasible
        if k <= target_bucket
    ]

    if not candidates:

        raise RuntimeError(
            "No feasible no-weaker execution bucket."
        )

    return int(
        max(
            candidates
        )
    )


def iter_risk_groups(
    risk_path: str,
) -> Iterator[
    Tuple[
        str,
        List[Dict[str, Any]],
    ]
]:

    current_sbs_id = None
    current_group = []

    for item in iter_jsonl(
        risk_path
    ):

        sbs_id = str(
            item[
                "sbs_id"
            ]
        )

        if current_sbs_id is None:

            current_sbs_id = (
                sbs_id
            )

        if (
            sbs_id
            != current_sbs_id
        ):

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


def build_true_local_diagnostic_counts(
    event_path: str,
    risk_path: str,
    successor_cache: Dict[
        str,
        List[str],
    ],
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """
    LOCAL DIAGNOSTIC ONLY.

    This recomputes the deterministic execution bucket from the true
    Phase-2/3 records. It is never used by the server estimator.
    """

    joint_counts = defaultdict(
        Counter
    )

    v_counts = defaultdict(
        Counter
    )

    k_counts = defaultdict(
        Counter
    )

    num_transitions = 0
    fallback_count = 0

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
                "Event/risk diagnostic length mismatch."
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
                "Event/risk SBS mismatch."
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
                "Event/risk transition-count mismatch."
            )

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

            if v not in (
                successor_cache.get(
                    u,
                    []
                )
            ):

                raise RuntimeError(
                    "True transition outside public N(u)."
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
                    remaining_budget=(
                        remaining
                    ),

                    future_after_current=(
                        future_after
                    ),

                    cfg=cfg,
                )
            )

            if not feasible:

                raise RuntimeError(
                    "Empty diagnostic feasible set."
                )

            execution = (
                choose_exec_bucket(
                    target_bucket=(
                        target
                    ),
                    feasible=(
                        feasible
                    ),
                )
            )

            if execution != target:

                fallback_count += 1

            cost = (
                cfg[
                    "eps_event_list"
                ][
                    execution - 1
                ]
                + cfg[
                    "eps_bucket"
                ]
            )

            remaining -= cost

            joint_counts[
                u
            ][
                (
                    v,
                    execution,
                )
            ] += 1

            v_counts[
                u
            ][
                v
            ] += 1

            k_counts[
                u
            ][
                execution
            ] += 1

            num_transitions += 1

    return {
        "joint_counts": (
            joint_counts
        ),

        "v_counts": (
            v_counts
        ),

        "k_counts": (
            k_counts
        ),

        "num_transitions": int(
            num_transitions
        ),

        "fallback_count": int(
            fallback_count
        ),

        "fallback_rate": (
            float(
                fallback_count
                / num_transitions
            )
            if num_transitions
            else 0.0
        ),
    }


# =============================================================================
# K-V dependence
# =============================================================================

def compute_conditional_dependence(
    true_counts: Dict[str, Any],
) -> Dict[str, Any]:

    joint_counts = (
        true_counts[
            "joint_counts"
        ]
    )

    total_all = int(
        true_counts[
            "num_transitions"
        ]
    )

    conditional_mi_nats = 0.0
    weighted_tv = 0.0

    max_tv = 0.0
    max_tv_u = None

    num_contexts = 0

    per_u_summary = {}

    for u, joint in (
        joint_counts.items()
    ):

        N_u = int(
            sum(
                joint.values()
            )
        )

        if N_u <= 0:
            continue

        num_contexts += 1

        v_counter = Counter()
        k_counter = Counter()

        for (
            v,
            k
        ), count in (
            joint.items()
        ):

            v_counter[
                v
            ] += count

            k_counter[
                k
            ] += count

        mi_u = 0.0
        tv_u = 0.0

        v_values = list(
            v_counter.keys()
        )

        k_values = list(
            k_counter.keys()
        )

        for v in v_values:

            p_v = (
                v_counter[
                    v
                ]
                / N_u
            )

            for k in k_values:

                p_k = (
                    k_counter[
                        k
                    ]
                    / N_u
                )

                p_vk = (
                    joint.get(
                        (
                            v,
                            k,
                        ),
                        0,
                    )
                    / N_u
                )

                product = (
                    p_v
                    * p_k
                )

                tv_u += (
                    abs(
                        p_vk
                        - product
                    )
                )

                if (
                    p_vk > 0
                    and product > 0
                ):

                    mi_u += (
                        p_vk
                        * math.log(
                            p_vk
                            / product
                        )
                    )

        tv_u *= 0.5

        weight = (
            N_u
            / total_all
            if total_all
            else 0.0
        )

        conditional_mi_nats += (
            weight
            * mi_u
        )

        weighted_tv += (
            weight
            * tv_u
        )

        if tv_u > max_tv:

            max_tv = (
                tv_u
            )

            max_tv_u = (
                u
            )

        per_u_summary[
            u
        ] = {
            "num_reports": (
                N_u
            ),

            "mutual_information_nats": (
                float(
                    mi_u
                )
            ),

            "conditional_tv": float(
                tv_u
            ),
        }

    return {
        "num_contexts": int(
            num_contexts
        ),

        "conditional_mutual_information_nats": float(
            conditional_mi_nats
        ),

        "conditional_mutual_information_bits": float(
            conditional_mi_nats
            / math.log(2.0)
        ),

        "weighted_conditional_tv": float(
            weighted_tv
        ),

        "max_context_tv": float(
            max_tv
        ),

        "max_context_tv_u": (
            max_tv_u
        ),

        "per_u": (
            per_u_summary
        ),
    }


# =============================================================================
# Recovery error against LOCAL truth
# =============================================================================

def counter_to_distribution(
    counter: Counter,
) -> Dict[str, float]:

    total = int(
        sum(
            counter.values()
        )
    )

    if total <= 0:
        return {}

    return {
        str(key): float(
            count
            / total
        )
        for key, count
        in counter.items()
    }


def tv_distance(
    p: Dict[str, float],
    q: Dict[str, float],
    domain: Sequence[str],
) -> float:

    return float(
        0.5
        * sum(
            abs(
                float(
                    p.get(
                        str(x),
                        0.0,
                    )
                )
                - float(
                    q.get(
                        str(x),
                        0.0,
                    )
                )
            )
            for x in domain
        )
    )


def compare_recovery_to_truth(
    true_counts: Dict[str, Any],
    successor_cache: Dict[
        str,
        List[str],
    ],
    context_result: Dict[
        str,
        Any,
    ],
    joint_result: Optional[
        Dict[str, Any]
    ],
) -> Dict[str, Any]:

    total_all = int(
        true_counts[
            "num_transitions"
        ]
    )

    weighted_context_tv = 0.0
    weighted_joint_tv = 0.0

    weighted_context_tv_joint_solved = 0.0
    weighted_joint_tv_joint_solved = 0.0

    solved_weight = 0.0
    solved_contexts = 0

    per_u = {}

    for u, true_counter in (
        true_counts[
            "v_counts"
        ].items()
    ):

        if u not in (
            context_result
        ):

            continue

        domain = list(
            successor_cache[
                u
            ]
        )

        N_u = int(
            sum(
                true_counter.values()
            )
        )

        weight = (
            N_u
            / total_all
            if total_all
            else 0.0
        )

        true_dist = (
            counter_to_distribution(
                true_counter
            )
        )

        context_dist = (
            context_result[
                u
            ][
                "distribution"
            ]
        )

        context_tv = (
            tv_distance(
                true_dist,
                context_dist,
                domain,
            )
        )

        weighted_context_tv += (
            weight
            * context_tv
        )

        joint_tv = None
        joint_source = None

        if (
            joint_result is not None
            and u in joint_result
        ):

            joint_dist = (
                joint_result[
                    u
                ][
                    "distribution"
                ]
            )

            joint_source = (
                joint_result[
                    u
                ][
                    "source"
                ]
            )

            joint_tv = (
                tv_distance(
                    true_dist,
                    joint_dist,
                    domain,
                )
            )

            weighted_joint_tv += (
                weight
                * joint_tv
            )

            if (
                joint_source
                == "joint_constrained_least_squares"
            ):

                solved_contexts += 1

                solved_weight += (
                    weight
                )

                weighted_context_tv_joint_solved += (
                    weight
                    * context_tv
                )

                weighted_joint_tv_joint_solved += (
                    weight
                    * joint_tv
                )

        per_u[
            u
        ] = {
            "num_true_transitions": (
                N_u
            ),

            "context_tv": float(
                context_tv
            ),

            "joint_tv": (
                None
                if joint_tv is None
                else float(
                    joint_tv
                )
            ),

            "joint_source": (
                joint_source
            ),
        }

    if solved_weight > 0:

        solved_context_avg = (
            weighted_context_tv_joint_solved
            / solved_weight
        )

        solved_joint_avg = (
            weighted_joint_tv_joint_solved
            / solved_weight
        )

    else:

        solved_context_avg = None
        solved_joint_avg = None

    return {
        "weighted_context_recovery_tv": float(
            weighted_context_tv
        ),

        "weighted_joint_hybrid_recovery_tv": (
            None
            if joint_result is None
            else float(
                weighted_joint_tv
            )
        ),

        "num_joint_solved_contexts": int(
            solved_contexts
        ),

        "joint_solved_transition_weight": float(
            solved_weight
        ),

        "joint_solved_context_only": {
            "context_tv": (
                solved_context_avg
            ),

            "joint_tv": (
                solved_joint_avg
            ),

            "absolute_improvement": (
                None
                if (
                    solved_context_avg
                    is None
                    or solved_joint_avg
                    is None
                )
                else float(
                    solved_context_avg
                    - solved_joint_avg
                )
            ),
        },

        "per_u": per_u,
    }


# =============================================================================
# Preliminary Recovery Gate
# =============================================================================

def build_recovery_gate(
    configured_mode: str,
    dependence: Dict[str, Any],
    comparison: Dict[str, Any],
) -> Dict[str, Any]:

    context_error = (
        comparison[
            "weighted_context_recovery_tv"
        ]
    )

    joint_error = (
        comparison[
            "weighted_joint_hybrid_recovery_tv"
        ]
    )

    solved = (
        comparison[
            "joint_solved_context_only"
        ]
    )

    recommendation = (
        configured_mode
    )

    reason = (
        "Configured context mixture retained."
    )

    if (
        joint_error is not None
        and solved[
            "absolute_improvement"
        ] is not None
    ):

        absolute_improvement = float(
            solved[
                "absolute_improvement"
            ]
        )

        # Deliberately conservative preliminary gate.
        # Final decision should use multi-seed/full-data results.
        if (
            absolute_improvement
            >= 0.005
            and solved[
                "joint_tv"
            ]
            < solved[
                "context_tv"
            ]
        ):

            recommendation = (
                "joint_latent_recovery"
            )

            reason = (
                "Joint recovery materially improves TV "
                "on contexts with enough reports."
            )

        else:

            recommendation = (
                "context_bucket_mixture"
            )

            reason = (
                "No material joint-recovery improvement "
                "on sufficiently supported contexts."
            )

    return {
        "configured_mode": (
            configured_mode
        ),

        "preliminary_recommended_mode": (
            recommendation
        ),

        "configured_mode_matches_preliminary_recommendation": (
            configured_mode
            == recommendation
        ),

        "weighted_conditional_tv_K_V_given_U": (
            dependence[
                "weighted_conditional_tv"
            ]
        ),

        "conditional_mutual_information_nats": (
            dependence[
                "conditional_mutual_information_nats"
            ]
        ),

        "weighted_context_recovery_tv": (
            context_error
        ),

        "weighted_joint_hybrid_recovery_tv": (
            joint_error
        ),

        "reason": reason,

        "important_note": (
            "This is a smoke/development Recovery Gate. "
            "Freeze the final recovery mode only after "
            "full-data multi-seed confirmation."
        ),
    }


# =============================================================================
# One split
# =============================================================================

def recover_one_split(
    split: str,
    report_path: str,
    event_path: str,
    risk_path: str,
    successor_cache: Dict[
        str,
        List[str],
    ],
    cfg: Dict[str, Any],
    joint_min_reports: int,
    ridge: float,
    skip_joint: bool,
) -> Dict[str, Any]:

    # -------------------------------------------------------------------------
    # Server-only recovery inputs
    # -------------------------------------------------------------------------

    aggregate = (
        aggregate_server_reports(
            report_path
        )
    )

    start_rec = (
        recover_sparse_symmetric_rr(
            observed=(
                aggregate[
                    "start_counter"
                ]
            ),

            public_domain_size=(
                len(
                    successor_cache
                )
            ),

            epsilon=(
                cfg[
                    "eps_start"
                ]
            ),
        )
    )

    count_domain = list(
        range(
            1,
            cfg[
                "L_max"
            ]
            + 1,
        )
    )

    count_rec = (
        recover_dense_symmetric_rr(
            observed=(
                aggregate[
                    "count_counter"
                ]
            ),

            domain=(
                count_domain
            ),

            epsilon=(
                cfg[
                    "eps_count"
                ]
            ),

            ridge=ridge,
        )
    )

    (
        context_transition,
        bucket_mix,
    ) = recover_context_bucket_mixture(
        aggregate=aggregate,
        successor_cache=(
            successor_cache
        ),
        cfg=cfg,
        ridge=ridge,
    )

    if skip_joint:

        joint_transition = None

    else:

        joint_transition = (
            recover_joint_latent(
                aggregate=aggregate,

                successor_cache=(
                    successor_cache
                ),

                context_result=(
                    context_transition
                ),

                cfg=cfg,

                joint_min_reports=(
                    joint_min_reports
                ),

                ridge=ridge,
            )
        )

    # -------------------------------------------------------------------------
    # LOCAL diagnostic only
    # -------------------------------------------------------------------------

    if cfg[
        "enable_recovery_dependence_diagnostic"
    ]:

        true_counts = (
            build_true_local_diagnostic_counts(
                event_path=(
                    event_path
                ),

                risk_path=(
                    risk_path
                ),

                successor_cache=(
                    successor_cache
                ),

                cfg=cfg,
            )
        )

        dependence = (
            compute_conditional_dependence(
                true_counts
            )
        )

        comparison = (
            compare_recovery_to_truth(
                true_counts=(
                    true_counts
                ),

                successor_cache=(
                    successor_cache
                ),

                context_result=(
                    context_transition
                ),

                joint_result=(
                    joint_transition
                ),
            )
        )

        gate = (
            build_recovery_gate(
                configured_mode=(
                    cfg[
                        "recovery_mode"
                    ]
                ),

                dependence=(
                    dependence
                ),

                comparison=(
                    comparison
                ),
            )
        )

    else:

        true_counts = None
        dependence = None
        comparison = None
        gate = None

    summary = {
        "split": split,

        "server_report_counts": {
            str(k): int(v)
            for k, v
            in aggregate[
                "type_counts"
            ].items()
        },

        "num_recovered_transition_contexts": int(
            len(
                context_transition
            )
        ),

        "joint_recovery_enabled": bool(
            not skip_joint
        ),

        "joint_min_reports": int(
            joint_min_reports
        ),

        "local_diagnostic_enabled": bool(
            cfg[
                "enable_recovery_dependence_diagnostic"
            ]
        ),

        "recomputed_scheduler_fallback_rate": (
            None
            if true_counts is None
            else true_counts[
                "fallback_rate"
            ]
        ),

        "recovery_gate": (
            gate
        ),
    }

    return {
        "start": (
            start_rec
        ),

        "count": (
            count_rec
        ),

        "transition_context": (
            context_transition
        ),

        "bucket_mix": (
            bucket_mix
        ),

        "transition_joint": (
            joint_transition
        ),

        "dependence": (
            dependence
        ),

        "comparison": (
            comparison
        ),

        "summary": (
            summary
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

    cfg = extract_recovery_cfg(
        exp_cfg
    )

    print("=" * 90)

    print(
        "[recover_statistics] "
        "Canonical Phase-5 Recovery Closure"
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
                "recovery_cfg": cfg,

                "joint_min_reports": (
                    args.joint_min_reports
                ),

                "ridge": (
                    args.ridge
                ),

                "joint_enabled": (
                    not args.skip_joint
                ),
            },
            indent=2,
            ensure_ascii=False,
        )
    )

    print("=" * 90)

    successor_cache_path = (
        resolve_path(
            dataset_cfg[
                "successor_cache_path"
            ]
        )
    )

    privatized_dir = (
        resolve_path(
            exp_cfg[
                "privatized_dir"
            ]
        )
    )

    recovered_dir = (
        resolve_path(
            exp_cfg[
                "recovered_dir"
            ]
        )
    )

    experiment_root = (
        resolve_path(
            exp_cfg[
                "experiment_root"
            ]
        )
    )

    assert successor_cache_path
    assert privatized_dir
    assert recovered_dir
    assert experiment_root

    ensure_dir(
        recovered_dir
    )

    # =========================================================================
    # Stage 1
    # =========================================================================

    total_stages = 5

    start = log_stage(
        1,
        total_stages,
        "Loading PUBLIC successor topology...",
    )

    successor_cache = (
        load_json(
            successor_cache_path
        )
    )

    if not isinstance(
        successor_cache,
        dict,
    ):

        raise ValueError(
            "Successor cache must be a JSON object."
        )

    print(
        "[Info] PUBLIC road segments = "
        f"{len(successor_cache):,}"
    )

    log_done(
        start,
        "Public topology loaded",
    )

    split_outputs = {}

    risk_dir = os.path.join(
        experiment_root,
        "risk",
    )

    # =========================================================================
    # Stages 2-4
    # =========================================================================

    for stage_idx, split in zip(
        [
            2,
            3,
            4,
        ],
        [
            "train",
            "valid",
            "test",
        ],
    ):

        start = log_stage(
            stage_idx,
            total_stages,
            f"Recovering {split} statistics...",
        )

        report_path = os.path.join(
            privatized_dir,
            f"riskaware_{split}_reports.jsonl",
        )

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

        for path in [
            report_path,
            event_path,
            risk_path,
        ]:

            if not os.path.exists(
                path
            ):

                raise FileNotFoundError(
                    f"Required Phase-5 input missing: "
                    f"{path}"
                )

        result = (
            recover_one_split(
                split=split,

                report_path=(
                    report_path
                ),

                event_path=(
                    event_path
                ),

                risk_path=(
                    risk_path
                ),

                successor_cache=(
                    successor_cache
                ),

                cfg=cfg,

                joint_min_reports=(
                    args.joint_min_reports
                ),

                ridge=args.ridge,

                skip_joint=(
                    args.skip_joint
                ),
            )
        )

        split_outputs[
            split
        ] = result[
            "summary"
        ]

        save_json(
            os.path.join(
                recovered_dir,
                f"riskaware_{split}_start.json",
            ),
            result[
                "start"
            ],
        )

        save_json(
            os.path.join(
                recovered_dir,
                f"riskaware_{split}_count.json",
            ),
            result[
                "count"
            ],
        )

        save_json(
            os.path.join(
                recovered_dir,
                f"riskaware_{split}_transition_context.json",
            ),
            result[
                "transition_context"
            ],
        )

        save_json(
            os.path.join(
                recovered_dir,
                f"riskaware_{split}_bucket_mix.json",
            ),
            result[
                "bucket_mix"
            ],
        )

        if (
            result[
                "transition_joint"
            ]
            is not None
        ):

            save_json(
                os.path.join(
                    recovered_dir,
                    f"riskaware_{split}_transition_joint.json",
                ),
                result[
                    "transition_joint"
                ],
            )

        if (
            result[
                "dependence"
            ]
            is not None
        ):

            diagnostic = {
                "split": split,

                "dependence": (
                    result[
                        "dependence"
                    ]
                ),

                "recovery_comparison": (
                    result[
                        "comparison"
                    ]
                ),

                "recovery_gate": (
                    result[
                        "summary"
                    ][
                        "recovery_gate"
                    ]
                ),
            }

            save_json(
                os.path.join(
                    recovered_dir,
                    f"riskaware_{split}_recovery_diagnostic.json",
                ),
                diagnostic,
            )

        gate = (
            result[
                "summary"
            ][
                "recovery_gate"
            ]
        )

        if gate is not None:

            print(
                f"[Recovery Gate][{split}] "
                f"I(K;V|U)="
                f"{gate['conditional_mutual_information_nats']:.6f}, "
                f"weighted-TV="
                f"{gate['weighted_conditional_tv_K_V_given_U']:.6f}, "
                f"context-TV="
                f"{gate['weighted_context_recovery_tv']:.6f}, "
                f"joint-TV="
                f"{gate['weighted_joint_hybrid_recovery_tv']}, "
                f"recommend="
                f"{gate['preliminary_recommended_mode']}"
            )

        log_done(
            start,
            f"{split} recovery completed",
        )

    # =========================================================================
    # Stage 5
    # =========================================================================

    start = log_stage(
        5,
        total_stages,
        "Saving Recovery Gate overview...",
    )

    recommendations = []

    for split in [
        "train",
        "valid",
        "test",
    ]:

        gate = (
            split_outputs[
                split
            ][
                "recovery_gate"
            ]
        )

        if gate is not None:

            recommendations.append(
                gate[
                    "preliminary_recommended_mode"
                ]
            )

    if recommendations:

        recommendation_counts = Counter(
            recommendations
        )

        preliminary_overall = (
            recommendation_counts.most_common(
                1
            )[0][0]
        )

    else:

        recommendation_counts = (
            Counter()
        )

        preliminary_overall = (
            cfg[
                "recovery_mode"
            ]
        )

    overview = {
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

        "configured_main_recovery": (
            cfg[
                "recovery_mode"
            ]
        ),

        "joint_plan_b_evaluated": bool(
            not args.skip_joint
        ),

        "joint_min_reports": int(
            args.joint_min_reports
        ),

        "splits": (
            split_outputs
        ),

        "preliminary_overall_recommendation": (
            preliminary_overall
        ),

        "recommendation_counts": {
            str(k): int(v)
            for k, v
            in recommendation_counts.items()
        },

        "method_freeze_status": (
            "PRELIMINARY_SMOKE_ONLY"
        ),

        "next_gate": (
            "Confirm the selected recovery mode "
            "on larger/full data and multiple seeds "
            "before freezing the paper mechanism."
        ),
    }

    overview_path = os.path.join(
        recovered_dir,
        "riskaware_recovery_summary.json",
    )

    save_json(
        overview_path,
        overview,
    )

    log_done(
        start,
        "Recovery overview saved",
    )

    print("=" * 90)

    print(
        "[recover_statistics] "
        "PHASE 5 SMOKE DONE"
    )

    print(
        f"Recovered directory:\n"
        f"  {recovered_dir}"
    )

    print(
        f"Recovery overview:\n"
        f"  {overview_path}"
    )

    print(
        f"Configured recovery = "
        f"{cfg['recovery_mode']}"
    )

    print(
        f"Preliminary recommendation = "
        f"{preliminary_overall}"
    )

    print(
        "\nIMPORTANT:"
        "\n  Server estimators use only privatized reports "
        "and public topology."
        "\n  True events/risk records are used only for "
        "the LOCAL Recovery Gate diagnostic."
    )

    print("=" * 90)


if __name__ == "__main__":
    main()