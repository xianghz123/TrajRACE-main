#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
versioning_utils.py

Canonical TrajRACE version/path management.

Responsibilities
----------------
1. Derive a DATASET variant from parameters that really change the
   raw/preprocessed/map-matched dataset.
2. Derive an EXPERIMENT tag from SBS/risk/privacy/recovery parameters.
3. Generate canonical paths for every pipeline stage.
4. Prevent different rebuttal runs from overwriting one another.

Important separation
--------------------
Dataset version MUST NOT depend on:
    - L_max
    - privacy budget B
    - epsilon allocation
    - K / risk thresholds
    - recovery parameters
    - experiment random seed

Those parameters affect downstream experiments, not raw map matching.

Therefore:
    changing B/K/L_max does NOT trigger map matching again;
    changing raw sample size / split seed / map matcher DOES.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from typing import Any, Dict, Optional, Tuple


# =============================================================================
# Generic helpers
# =============================================================================

def _stable_hash(obj: Any, n: int = 8) -> str:
    """
    Stable short hash for configuration signatures.
    """
    text = json.dumps(
        obj,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:n]


def _fmt_num_for_tag(x: Any) -> str:
    """
    Examples:
        1.0   -> 1
        1.5   -> 1p5
        0.005 -> 0p005
    """
    if x is None:
        return "none"

    try:
        value = float(x)
    except (TypeError, ValueError):
        return str(x).replace(".", "p").replace("-", "m")

    if abs(value - int(value)) < 1e-12:
        return str(int(value))

    s = f"{value:.12g}"

    if "." in s:
        s = s.rstrip("0").rstrip(".")

    return s.replace(".", "p").replace("-", "m")


def _safe_tag(text: Any) -> str:
    s = str(text)
    s = s.replace(" ", "")
    s = s.replace("/", "-")
    s = s.replace("\\", "-")
    s = s.replace(".", "p")
    return s


def _dataset_name(dataset_cfg: Dict[str, Any]) -> str:
    return str(dataset_cfg.get("dataset_name", "dataset"))


def _raw_sample_size(dataset_cfg: Dict[str, Any]) -> int:
    """
    Canonical key:
        raw_sample_size

    The old Porto-specific key is supported only for backward compatibility.
    """
    if dataset_cfg.get("raw_sample_size") is not None:
        return int(dataset_cfg["raw_sample_size"])

    if dataset_cfg.get("porto_raw_sample_size") is not None:
        return int(dataset_cfg["porto_raw_sample_size"])

    return 0


def _active_privacy_profile(
    exp_cfg: Dict[str, Any],
) -> Tuple[str, Dict[str, Any]]:
    """
    Return:
        profile_name, profile_dict
    """
    profile_name = str(
        exp_cfg.get(
            "active_budget_profile",
            exp_cfg.get("default_budget_profile", "B1p0"),
        )
    )

    profiles = exp_cfg.get("privacy_profiles", {})

    if not isinstance(profiles, dict):
        raise ValueError("privacy_profiles must be a mapping")

    if profile_name not in profiles:
        raise KeyError(
            f"Active privacy profile '{profile_name}' not found in "
            f"privacy_profiles={list(profiles.keys())}"
        )

    profile = profiles[profile_name]

    if not isinstance(profile, dict):
        raise ValueError(
            f"privacy profile '{profile_name}' must be a mapping"
        )

    return profile_name, copy.deepcopy(profile)


# =============================================================================
# Dataset-side versioning
# =============================================================================

def build_dataset_signature(
    dataset_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Dataset signature contains ONLY parameters that change the
    raw/preprocessed/map-matched trajectory collection.
    """
    return {
        "dataset_name": _dataset_name(dataset_cfg),

        "raw_csv": str(dataset_cfg.get("raw_csv", "")),
        "raw_sample_size": _raw_sample_size(dataset_cfg),

        "sampling_interval_sec": dataset_cfg.get(
            "sampling_interval_sec",
            None,
        ),
        "segment_time_mode": dataset_cfg.get(
            "segment_time_mode",
            None,
        ),

        "train_ratio": dataset_cfg.get("train_ratio", 0.8),
        "valid_ratio": dataset_cfg.get("valid_ratio", 0.1),
        "test_ratio": dataset_cfg.get("test_ratio", 0.1),
        "split_seed": dataset_cfg.get("split_seed", 42),

        "drop_missing_data": dataset_cfg.get(
            "drop_missing_data",
            True,
        ),
        "min_points_per_traj": dataset_cfg.get(
            "min_points_per_traj",
            0,
        ),
        "max_points_per_traj_before_map_matching": dataset_cfg.get(
            "max_points_per_traj_before_map_matching",
            None,
        ),

        "enable_point_downsample_before_map_matching": dataset_cfg.get(
            "enable_point_downsample_before_map_matching",
            False,
        ),
        "max_points_after_downsample_before_map_matching": dataset_cfg.get(
            "max_points_after_downsample_before_map_matching",
            None,
        ),

        "map_matching_method": dataset_cfg.get(
            "map_matching_method",
            "nearest_node_shortest_path",
        ),

        "road_graph_path": str(
            dataset_cfg.get("road_graph_path", "")
        ),

        "deduplicate_consecutive_segments": dataset_cfg.get(
            "deduplicate_consecutive_segments",
            True,
        ),

        "min_segments_per_traj": dataset_cfg.get(
            "min_segments_per_traj",
            2,
        ),

        "preserve_full_mapped_trajectory": dataset_cfg.get(
            "preserve_full_mapped_trajectory",
            True,
        ),
    }


def derive_dataset_variant(
    dataset_cfg: Dict[str, Any],
    exp_cfg: Optional[Dict[str, Any]] = None,
) -> str:
    """
    exp_cfg is accepted for backward compatibility but intentionally
    does NOT affect the dataset variant.
    """
    del exp_cfg

    name = _dataset_name(dataset_cfg)
    sample_size = _raw_sample_size(dataset_cfg)
    seed = int(dataset_cfg.get("split_seed", 42))
    dt = int(dataset_cfg.get("sampling_interval_sec", 0))

    mm = _safe_tag(
        dataset_cfg.get(
            "map_matching_method",
            "nearest_node_shortest_path",
        )
    )

    signature = build_dataset_signature(dataset_cfg)
    short_hash = _stable_hash(signature, n=8)

    return (
        f"{name}"
        f"_s{sample_size}"
        f"_seed{seed}"
        f"_dt{dt}"
        f"_mm{mm}"
        f"_{short_hash}"
    )


def derive_dataset_root(
    dataset_cfg: Dict[str, Any],
    exp_cfg: Optional[Dict[str, Any]] = None,
) -> str:
    del exp_cfg

    base_dir = str(
        dataset_cfg.get(
            "variants_base_dir",
            "data/variants",
        )
    )

    variant = derive_dataset_variant(dataset_cfg)

    return os.path.join(base_dir, variant)


def _assign_dataset_paths(
    dataset_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Add all automatically generated paths to an already versioned
    dataset configuration.
    """
    out = copy.deepcopy(dataset_cfg)

    root = str(out["dataset_root"])
    name = _dataset_name(out)

    # -------------------------------------------------------------------------
    # Stage 1: preprocessing
    # -------------------------------------------------------------------------
    out["cleaned_path"] = os.path.join(
        root,
        "intermediate",
        "cleaned",
        f"{name}_cleaned.jsonl",
    )

    out["segment_sequence_dir"] = os.path.join(
        root,
        "intermediate",
        "segment_sequences",
    )

    out["segment_seq_train"] = os.path.join(
        out["segment_sequence_dir"],
        "train.jsonl",
    )

    out["segment_seq_valid"] = os.path.join(
        out["segment_sequence_dir"],
        "valid.jsonl",
    )

    out["segment_seq_test"] = os.path.join(
        out["segment_sequence_dir"],
        "test.jsonl",
    )

    out["preprocess_summary_path"] = os.path.join(
        out["segment_sequence_dir"],
        "preprocess_summary.json",
    )

    # -------------------------------------------------------------------------
    # Stage 2: SBS / events
    # -------------------------------------------------------------------------
    out["event_output_dir"] = os.path.join(
        root,
        "processed",
        "events",
    )

    out["event_train"] = os.path.join(
        out["event_output_dir"],
        "train_events.jsonl",
    )

    out["event_valid"] = os.path.join(
        out["event_output_dir"],
        "valid_events.jsonl",
    )

    out["event_test"] = os.path.join(
        out["event_output_dir"],
        "test_events.jsonl",
    )

    out["event_summary_path"] = os.path.join(
        out["event_output_dir"],
        "event_summary.json",
    )

    # -------------------------------------------------------------------------
    # Stage 3: risk
    # -------------------------------------------------------------------------
    out["risk_output_dir"] = os.path.join(
        root,
        "processed",
        "risk",
    )

    out["risk_train"] = os.path.join(
        out["risk_output_dir"],
        "train_risk.jsonl",
    )

    out["risk_valid"] = os.path.join(
        out["risk_output_dir"],
        "valid_risk.jsonl",
    )

    out["risk_test"] = os.path.join(
        out["risk_output_dir"],
        "test_risk.jsonl",
    )

    return out


def apply_dataset_versioning(
    dataset_cfg: Dict[str, Any],
    exp_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Apply canonical dataset versioning.

    The experiment configuration is intentionally ignored for the
    dataset variant itself.
    """
    del exp_cfg

    out = copy.deepcopy(dataset_cfg)

    out["raw_sample_size"] = _raw_sample_size(out)

    out["dataset_variant"] = derive_dataset_variant(out)
    out["dataset_root"] = derive_dataset_root(out)

    return _assign_dataset_paths(out)


# =============================================================================
# Experiment-side versioning
# =============================================================================

def build_experiment_signature(
    dataset_cfg: Dict[str, Any],
    exp_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Experiment signature contains parameters that affect SBS/risk/privacy/
    recovery/synthesis/evaluation outputs.
    """
    profile_name, profile = _active_privacy_profile(exp_cfg)

    return {
        "dataset_variant": dataset_cfg.get("dataset_variant", ""),

        "experiment_name": exp_cfg.get(
            "experiment_name",
            "trajrace",
        ),

        # SBS
        "L_max": exp_cfg.get("L_max", 30),
        "sbs_partition_mode": exp_cfg.get(
            "sbs_partition_mode",
            "transition_preserving_overlap",
        ),
        "count_mode": exp_cfg.get(
            "count_mode",
            "exact_num_segments",
        ),

        # Risk
        "lambda_e": exp_cfg.get("lambda_e"),
        "lambda_s": exp_cfg.get("lambda_s"),
        "lambda_d": exp_cfg.get("lambda_d"),
        "sigma_e": exp_cfg.get("sigma_e"),
        "sigma_st": exp_cfg.get("sigma_st"),
        "delta_st_sec": exp_cfg.get("delta_st_sec"),
        "distance_mode": exp_cfg.get("distance_mode"),
        "bucket_mapping_mode": exp_cfg.get(
            "bucket_mapping_mode",
        ),
        "K": exp_cfg.get("K"),
        "theta_list": exp_cfg.get("theta_list", []),

        # Privacy
        "privacy_scope": exp_cfg.get("privacy_scope"),
        "B_semantics": exp_cfg.get("B_semantics"),
        "active_budget_profile": profile_name,
        "privacy_profile": profile,

        # Recovery
        "recovery_mode": exp_cfg.get("recovery_mode"),
        "tau_shrinkage": exp_cfg.get(
            "tau_shrinkage",
            20.0,
        ),
        "use_global_backoff": exp_cfg.get(
            "use_global_backoff",
            True,
        ),

        # Synthesis / randomness
        "synthetic_num_mode": exp_cfg.get(
            "synthetic_num_mode",
            "match_test_size",
        ),
        "random_seed": exp_cfg.get(
            "random_seed",
            42,
        ),

        # Evaluation
        "shared_grid_size": exp_cfg.get(
            "shared_grid_size",
            6,
        ),
    }


def derive_exp_tag(
    dataset_cfg: Dict[str, Any],
    exp_cfg: Dict[str, Any],
) -> str:
    profile_name, profile = _active_privacy_profile(exp_cfg)

    B = _fmt_num_for_tag(profile.get("B"))
    L_max = int(exp_cfg.get("L_max", 30))
    K = int(exp_cfg.get("K", 3))
    seed = int(exp_cfg.get("random_seed", 42))

    recovery = _safe_tag(
        exp_cfg.get(
            "recovery_mode",
            "context_bucket_mixture",
        )
    )

    signature = build_experiment_signature(
        dataset_cfg,
        exp_cfg,
    )

    short_hash = _stable_hash(signature, n=8)

    return (
        f"Lm{L_max}"
        f"_K{K}"
        f"_{profile_name}"
        f"_B{B}"
        f"_rec{recovery}"
        f"_seed{seed}"
        f"_{short_hash}"
    )


def _assign_experiment_paths(
    dataset_cfg: Dict[str, Any],
    exp_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    out = copy.deepcopy(exp_cfg)

    dataset_root = str(dataset_cfg["dataset_root"])
    exp_tag = str(out["exp_tag"])

    experiment_root = os.path.join(
        dataset_root,
        "experiments",
        exp_tag,
    )

    out["experiment_root"] = experiment_root

    out["privatized_dir"] = os.path.join(
        experiment_root,
        "privatized",
    )

    out["recovered_dir"] = os.path.join(
        experiment_root,
        "recovered",
    )

    out["synthetic_dir"] = os.path.join(
        experiment_root,
        "synthetic",
    )

    out["evaluation_dir"] = os.path.join(
        experiment_root,
        "evaluation",
    )

    out["audit_dir"] = os.path.join(
        experiment_root,
        "audit",
    )

    out["logs_dir"] = os.path.join(
        experiment_root,
        "logs",
    )

    return out


def apply_exp_versioning(
    dataset_cfg: Dict[str, Any],
    exp_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    out = copy.deepcopy(exp_cfg)

    profile_name, profile = _active_privacy_profile(out)

    out["active_budget_profile"] = profile_name

    # Flatten the active profile for downstream scripts.
    out["B_total"] = float(profile["B"])
    out["eps_start"] = float(profile["eps_start"])
    out["eps_count"] = float(profile["eps_count"])
    out["eps_bucket"] = float(profile["eps_bucket"])
    out["eps_event_list"] = [
        float(x)
        for x in profile["eps_event_list"]
    ]

    out["exp_tag"] = derive_exp_tag(
        dataset_cfg,
        out,
    )

    return _assign_experiment_paths(
        dataset_cfg,
        out,
    )


# =============================================================================
# Optional fixed-dataset reuse
# =============================================================================

def force_fixed_dataset_variant(
    dataset_cfg: Dict[str, Any],
    exp_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Optional expert mode.

    Allows downstream experiments to reuse an already-preprocessed dataset
    root explicitly.

    Unlike the legacy implementation, normal changes to B/K/L_max/risk
    parameters already reuse the same dataset automatically, so this option
    is rarely necessary.
    """
    fixed_variant = exp_cfg.get(
        "fixed_dataset_variant",
        None,
    )

    fixed_root = exp_cfg.get(
        "fixed_dataset_root",
        None,
    )

    if fixed_variant is None and fixed_root is None:
        return dataset_cfg

    out = copy.deepcopy(dataset_cfg)

    if fixed_variant is not None:
        out["dataset_variant"] = str(
            fixed_variant
        )

    if fixed_root is not None:
        out["dataset_root"] = str(
            fixed_root
        )
    else:
        base_dir = str(
            out.get(
                "variants_base_dir",
                "data/variants",
            )
        )

        out["dataset_root"] = os.path.join(
            base_dir,
            str(out["dataset_variant"]),
        )

    return _assign_dataset_paths(out)


# =============================================================================
# Public API
# =============================================================================

def apply_versioning(
    raw_dataset_cfg: Dict[str, Any],
    raw_exp_cfg: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Canonical entry point.

    1. Dataset-side versioning:
       raw/preprocessing/map-matching parameters only.

    2. Optional explicit fixed-dataset reuse.

    3. Experiment-side versioning:
       SBS/risk/privacy/recovery/synthesis/evaluation parameters.
    """
    dataset_cfg = apply_dataset_versioning(
        raw_dataset_cfg
    )

    dataset_cfg = force_fixed_dataset_variant(
        dataset_cfg,
        raw_exp_cfg,
    )

    exp_cfg = apply_exp_versioning(
        dataset_cfg,
        raw_exp_cfg,
    )

    return dataset_cfg, exp_cfg


def pretty_version_summary(
    raw_dataset_cfg: Dict[str, Any],
    raw_exp_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    dataset_cfg, exp_cfg = apply_versioning(
        raw_dataset_cfg,
        raw_exp_cfg,
    )

    profile_name, profile = _active_privacy_profile(
        exp_cfg
    )

    return {
        "dataset_name": dataset_cfg["dataset_name"],
        "dataset_variant": dataset_cfg["dataset_variant"],
        "dataset_root": dataset_cfg["dataset_root"],

        "raw_sample_size": dataset_cfg[
            "raw_sample_size"
        ],
        "split_seed": dataset_cfg.get(
            "split_seed",
            42,
        ),
        "sampling_interval_sec": dataset_cfg.get(
            "sampling_interval_sec",
            0,
        ),
        "map_matching_method": dataset_cfg.get(
            "map_matching_method",
        ),

        "exp_tag": exp_cfg["exp_tag"],
        "experiment_root": exp_cfg[
            "experiment_root"
        ],

        "L_max": int(
            exp_cfg.get(
                "L_max",
                30,
            )
        ),

        "active_budget_profile": profile_name,
        "B": float(profile["B"]),

        "K": int(
            exp_cfg.get(
                "K",
                3,
            )
        ),

        "recovery_mode": exp_cfg.get(
            "recovery_mode",
        ),

        "random_seed": int(
            exp_cfg.get(
                "random_seed",
                42,
            )
        ),
    }