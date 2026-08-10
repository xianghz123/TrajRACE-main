#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

"""
run_rebuttal_experiments.py

Final TrajRACE rebuttal experiment orchestrator.

Goals
-----
1. Provide one reproducible entry point for the canonical pipeline.
2. Reuse shared deterministic outputs when safe.
3. Run budget/seed grids without manually editing exp_main.yaml.
4. Fail fast on any correctness/privacy gate.
5. Save commands, logs, generated configs, Git commit, durations, and
   aggregate results in a run manifest.
6. Keep the fixed structural configuration (L_max/SBS mode/risk definition)
   unchanged within one sweep; only the active privacy profile and experiment
   random seed are changed automatically.

Canonical pipeline
------------------
Shared dataset stages:
    preprocess_dataset.py
    build_events.py

Per-experiment stages:
    compute_transition_risk.py
    privatize_riskaware.py
    audit_privacy.py
    recover_statistics.py
    privatize_uniform.py
    recover_uniform_statistics.py
    synthesize_trajectories.py
    evaluate_compare.py

Important
---------
- Existing outputs are reused by default when their marker file exists.
  Use --rerun to force recomputation.
- For grid runs, the expensive joint recovery diagnostic is executed only
  for the canonical profile/seed by default. Other runs use --skip_joint.
- This runner does not change the formal privacy scope. B remains an SBS
  reporting-schedule cap; the formal transition guarantee is conditional
  event-level LDP given public U.
"""

import argparse
import csv
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from trajrace.versioning_utils import apply_versioning, pretty_version_summary


DEFAULT_DATASET_CONFIG = "configs/dataset.yaml"
DEFAULT_EXP_CONFIG = "configs/exp_main.yaml"

ALL_STAGES = [
    "preprocess",
    "events",
    "risk",
    "riskaware",
    "privacy_audit",
    "recovery",
    "uniform",
    "uniform_recovery",
    "synthesis",
    "evaluation",
]


# =============================================================================
# I/O helpers
# =============================================================================

def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def load_yaml(path: Path) -> Dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)

    if not isinstance(obj, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")

    return obj


def save_yaml(path: Path, obj: Dict[str, Any]) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            obj,
            f,
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
        )


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(
            obj,
            f,
            ensure_ascii=False,
            indent=2,
        )


def save_csv(
    path: Path,
    rows: List[Dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames: List[str] = []

    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def utc_now_iso() -> str:
    return dt.datetime.now(
        dt.timezone.utc
    ).isoformat()


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run canonical TrajRACE rebuttal experiments "
            "with fail-fast gates and manifests."
        )
    )

    parser.add_argument(
        "--dataset_config",
        default=DEFAULT_DATASET_CONFIG,
    )

    parser.add_argument(
        "--exp_config",
        default=DEFAULT_EXP_CONFIG,
    )

    parser.add_argument(
        "--mode",
        choices=[
            "single",
            "grid",
        ],
        default="single",
        help=(
            "single: current active privacy profile/random_seed; "
            "grid: all rebuttal_budget_profiles x multiseed_values."
        ),
    )

    parser.add_argument(
        "--profiles",
        nargs="*",
        default=None,
        help=(
            "Override privacy profiles for grid mode, e.g. "
            "--profiles B0p5 B1p0 B1p5."
        ),
    )

    parser.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=None,
        help=(
            "Override experiment seeds for grid mode, e.g. "
            "--seeds 42 43 44 45 46."
        ),
    )

    parser.add_argument(
        "--raw_sample_size_override",
        type=int,
        default=None,
        help=(
            "Create a generated dataset config with this raw sample size. "
            "Useful for smoke/development runs. The value participates in "
            "dataset versioning and therefore cannot overwrite canonical 100K."
        ),
    )

    parser.add_argument(
        "--stages",
        nargs="*",
        choices=ALL_STAGES,
        default=None,
        help=(
            "Run only selected stages. Default: the complete pipeline."
        ),
    )

    parser.add_argument(
        "--rerun",
        action="store_true",
        help=(
            "Force selected stages even when their marker output exists."
        ),
    )

    parser.add_argument(
        "--force_restart_map_matching",
        action="store_true",
        help=(
            "Pass --force_restart_map_matching to preprocessing."
        ),
    )

    parser.add_argument(
        "--allow_download_osm",
        action="store_true",
        help=(
            "Pass --allow_download_osm to preprocessing. "
            "Do not use in frozen canonical runs unless intentionally setting "
            "up the public graph for the first time."
        ),
    )

    parser.add_argument(
        "--save_debug",
        action="store_true",
        help=(
            "Save LOCAL private debug traces for risk-aware/uniform "
            "privatization. Recommended only for smoke/debug runs."
        ),
    )

    parser.add_argument(
        "--joint_mode",
        choices=[
            "canonical",
            "all",
            "none",
        ],
        default="canonical",
        help=(
            "Control expensive joint recovery diagnostic. "
            "canonical: full joint only on base active profile/base seed; "
            "all: all runs; none: always --skip_joint."
        ),
    )

    parser.add_argument(
        "--joint_min_reports",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--python",
        default=sys.executable,
        help=(
            "Python interpreter used for child scripts."
        ),
    )

    parser.add_argument(
        "--run_root",
        default="rebuttal_runs",
        help=(
            "Directory under project root for generated configs/logs/manifests."
        ),
    )

    return parser.parse_args()


# =============================================================================
# Configuration generation
# =============================================================================

def make_dataset_config(
    raw_dataset_cfg: Dict[str, Any],
    run_dir: Path,
    raw_sample_size_override: Optional[int],
) -> Path:

    cfg = dict(raw_dataset_cfg)

    if raw_sample_size_override is not None:

        if raw_sample_size_override <= 0:
            raise ValueError(
                "--raw_sample_size_override must be > 0"
            )

        cfg[
            "raw_sample_size"
        ] = int(
            raw_sample_size_override
        )

    path = (
        run_dir
        / "configs"
        / "dataset.generated.yaml"
    )

    save_yaml(
        path,
        cfg,
    )

    return path


def make_exp_config(
    raw_exp_cfg: Dict[str, Any],
    run_dir: Path,
    profile: str,
    seed: int,
) -> Path:

    cfg = dict(
        raw_exp_cfg
    )

    privacy_profiles = cfg.get(
        "privacy_profiles",
        {},
    )

    if profile not in privacy_profiles:
        raise KeyError(
            f"Privacy profile '{profile}' not found in exp config."
        )

    cfg[
        "active_budget_profile"
    ] = str(
        profile
    )

    cfg[
        "default_budget_profile"
    ] = str(
        profile
    )

    cfg[
        "random_seed"
    ] = int(
        seed
    )

    path = (
        run_dir
        / "configs"
        / f"exp_{profile}_seed{seed}.generated.yaml"
    )

    save_yaml(
        path,
        cfg,
    )

    return path


def resolve_single_and_grid(
    raw_exp_cfg: Dict[str, Any],
    args: argparse.Namespace,
) -> Tuple[
    List[str],
    List[int],
]:

    base_profile = str(
        raw_exp_cfg.get(
            "active_budget_profile",
            raw_exp_cfg.get(
                "default_budget_profile",
                "B1p0",
            ),
        )
    )

    base_seed = int(
        raw_exp_cfg.get(
            "random_seed",
            42,
        )
    )

    if args.mode == "single":
        return [
            base_profile
        ], [
            base_seed
        ]

    profiles = (
        args.profiles
        if args.profiles
        else list(
            raw_exp_cfg.get(
                "rebuttal_budget_profiles",
                [
                    base_profile
                ],
            )
        )
    )

    seeds = (
        args.seeds
        if args.seeds
        else [
            int(x)
            for x in raw_exp_cfg.get(
                "multiseed_values",
                [
                    base_seed
                ],
            )
        ]
    )

    return (
        [
            str(x)
            for x in profiles
        ],
        [
            int(x)
            for x in seeds
        ],
    )


def validate_structural_sweep(
    raw_exp_cfg: Dict[str, Any],
) -> None:
    """
    The runner changes only privacy profile and random_seed.

    Structural parameters are therefore fixed by construction. This guard
    documents the assumption that dataset-root SBS events can be reused
    safely within one sweep.
    """

    if str(
        raw_exp_cfg.get(
            "sbs_partition_mode",
            ""
        )
    ) != "transition_preserving_overlap":

        raise ValueError(
            "Canonical runner requires "
            "sbs_partition_mode=transition_preserving_overlap."
        )

    if str(
        raw_exp_cfg.get(
            "count_mode",
            ""
        )
    ) != "exact_num_segments":

        raise ValueError(
            "Canonical runner requires count_mode=exact_num_segments."
        )


# =============================================================================
# Git / environment metadata
# =============================================================================

def git_metadata() -> Dict[str, Any]:

    result: Dict[str, Any] = {
        "commit": None,
        "dirty": None,
    }

    try:

        commit = subprocess.check_output(
            [
                "git",
                "rev-parse",
                "HEAD",
            ],
            cwd=PROJECT_ROOT,
            text=True,
        ).strip()

        result[
            "commit"
        ] = commit

        status = subprocess.check_output(
            [
                "git",
                "status",
                "--porcelain",
            ],
            cwd=PROJECT_ROOT,
            text=True,
        )

        result[
            "dirty"
        ] = bool(
            status.strip()
        )

    except Exception:
        pass

    return result


# =============================================================================
# Process runner
# =============================================================================

def run_command(
    command: Sequence[str],
    log_path: Path,
) -> Dict[str, Any]:

    log_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    command = [
        str(x)
        for x in command
    ]

    print(
        "\n"
        + "="
        * 100
    )

    print(
        "[RUN] "
        + " ".join(
            command
        )
    )

    print(
        "[LOG] "
        + str(
            log_path
        )
    )

    print(
        "="
        * 100
    )

    start = time.perf_counter()
    started_at = utc_now_iso()

    with log_path.open(
        "w",
        encoding="utf-8",
    ) as log_file:

        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        assert process.stdout is not None

        for line in process.stdout:

            print(
                line,
                end="",
            )

            log_file.write(
                line
            )

        return_code = (
            process.wait()
        )

    elapsed = (
        time.perf_counter()
        - start
    )

    result = {
        "command": command,
        "started_at_utc": started_at,
        "finished_at_utc": utc_now_iso(),
        "elapsed_sec": float(
            elapsed
        ),
        "return_code": int(
            return_code
        ),
        "log_path": str(
            log_path
        ),
    }

    if return_code != 0:

        raise RuntimeError(
            "Command failed with "
            f"return code {return_code}: "
            + " ".join(
                command
            )
        )

    return result


# =============================================================================
# Marker resolution
# =============================================================================

def resolved_configs(
    dataset_config_path: Path,
    exp_config_path: Path,
) -> Tuple[
    Dict[str, Any],
    Dict[str, Any],
]:

    raw_d = load_yaml(
        dataset_config_path
    )

    raw_e = load_yaml(
        exp_config_path
    )

    return apply_versioning(
        raw_d,
        raw_e,
    )


def marker_paths(
    dcfg: Dict[str, Any],
    ecfg: Dict[str, Any],
) -> Dict[str, Path]:

    experiment_root = resolve_path(
        str(
            ecfg[
                "experiment_root"
            ]
        )
    )

    privatized_dir = resolve_path(
        str(
            ecfg[
                "privatized_dir"
            ]
        )
    )

    recovered_dir = resolve_path(
        str(
            ecfg[
                "recovered_dir"
            ]
        )
    )

    synthetic_dir = resolve_path(
        str(
            ecfg[
                "synthetic_dir"
            ]
        )
    )

    audit_dir = resolve_path(
        str(
            ecfg[
                "audit_dir"
            ]
        )
    )

    return {
        "preprocess": resolve_path(
            str(
                dcfg[
                    "preprocess_summary_path"
                ]
            )
        ),

        "events": resolve_path(
            str(
                dcfg[
                    "event_summary_path"
                ]
            )
        ),

        "risk": (
            experiment_root
            / "risk"
            / "risk_summary.json"
        ),

        "riskaware": (
            privatized_dir
            / "riskaware_summary.json"
        ),

        "privacy_audit": (
            audit_dir
            / "privacy_audit.json"
        ),

        "recovery": (
            recovered_dir
            / "riskaware_recovery_summary.json"
        ),

        "uniform": (
            privatized_dir
            / "uniform_summary.json"
        ),

        "uniform_recovery": (
            recovered_dir
            / "uniform_recovery_summary.json"
        ),

        "synthesis": (
            synthetic_dir
            / "synthetic_overview.json"
        ),

        "evaluation": (
            experiment_root
            / "evaluation"
            / "main_compare.json"
        ),
    }


def marker_is_valid(
    stage: str,
    marker: Path,
) -> bool:

    if not marker.exists():
        return False

    if stage == "privacy_audit":

        try:
            audit = load_json(
                marker
            )

            return (
                audit.get(
                    "overall_status"
                )
                == "PASS"
                and int(
                    audit.get(
                        "num_hard_failures",
                        -1,
                    )
                )
                == 0
            )

        except Exception:
            return False

    return True


# =============================================================================
# Stage commands
# =============================================================================

def stage_command(
    stage: str,
    python_exe: str,
    dataset_config_path: Path,
    exp_config_path: Path,
    args: argparse.Namespace,
    do_joint: bool,
) -> List[str]:

    base = [
        python_exe,
    ]

    common = [
        "--dataset_config",
        str(
            dataset_config_path
        ),
        "--exp_config",
        str(
            exp_config_path
        ),
    ]

    if stage == "preprocess":

        cmd = base + [
            "scripts/preprocess_dataset.py",
        ] + common

        if args.force_restart_map_matching:
            cmd.append(
                "--force_restart_map_matching"
            )

        if args.allow_download_osm:
            cmd.append(
                "--allow_download_osm"
            )

        return cmd

    if stage == "events":

        return (
            base
            + [
                "scripts/build_events.py"
            ]
            + common
        )

    if stage == "risk":

        return (
            base
            + [
                "scripts/compute_transition_risk.py"
            ]
            + common
        )

    if stage == "riskaware":

        cmd = (
            base
            + [
                "scripts/privatize_riskaware.py"
            ]
            + common
        )

        if args.save_debug:
            cmd.append(
                "--save_debug"
            )

        return cmd

    if stage == "privacy_audit":

        return (
            base
            + [
                "scripts/audit_privacy.py"
            ]
            + common
        )

    if stage == "recovery":

        cmd = (
            base
            + [
                "scripts/recover_statistics.py"
            ]
            + common
            + [
                "--joint_min_reports",
                str(
                    args.joint_min_reports
                ),
            ]
        )

        if not do_joint:
            cmd.append(
                "--skip_joint"
            )

        return cmd

    if stage == "uniform":

        cmd = (
            base
            + [
                "scripts/privatize_uniform.py"
            ]
            + common
        )

        if args.save_debug:
            cmd.append(
                "--save_debug"
            )

        return cmd

    if stage == "uniform_recovery":

        return (
            base
            + [
                "scripts/recover_uniform_statistics.py"
            ]
            + common
        )

    if stage == "synthesis":

        return (
            base
            + [
                "scripts/synthesize_trajectories.py"
            ]
            + common
        )

    if stage == "evaluation":

        return (
            base
            + [
                "scripts/evaluate_compare.py"
            ]
            + common
        )

    raise KeyError(
        f"Unknown stage: {stage}"
    )


# =============================================================================
# Result collection
# =============================================================================

def get_nested(
    obj: Any,
    path: Sequence[str],
    default: Any = None,
) -> Any:

    cur = obj

    for key in path:

        if not isinstance(
            cur,
            dict,
        ):
            return default

        if key not in cur:
            return default

        cur = cur[
            key
        ]

    return cur


def collect_experiment_result(
    profile: str,
    seed: int,
    dcfg: Dict[str, Any],
    ecfg: Dict[str, Any],
    markers: Dict[str, Path],
) -> List[Dict[str, Any]]:

    rows: List[
        Dict[str, Any]
    ] = []

    eval_obj = (
        load_json(
            markers[
                "evaluation"
            ]
        )
        if markers[
            "evaluation"
        ].exists()
        else {}
    )

    audit_obj = (
        load_json(
            markers[
                "privacy_audit"
            ]
        )
        if markers[
            "privacy_audit"
        ].exists()
        else {}
    )

    recovery_obj = (
        load_json(
            markers[
                "recovery"
            ]
        )
        if markers[
            "recovery"
        ].exists()
        else {}
    )

    riskaware_summary_path = markers[
        "riskaware"
    ]

    riskaware_summary = (
        load_json(
            riskaware_summary_path
        )
        if riskaware_summary_path.exists()
        else {}
    )

    B_value = ecfg.get(
        "B_total",
        ecfg.get(
            "B",
            None,
        ),
    )

    for result in eval_obj.get(
        "results",
        [],
    ):

        method = str(
            result.get(
                "method"
            )
        )

        row = {
            "dataset_name": dcfg.get(
                "dataset_name"
            ),

            "dataset_variant": dcfg.get(
                "dataset_variant"
            ),

            "profile": profile,

            "B": B_value,

            "seed": int(
                seed
            ),

            "exp_tag": ecfg.get(
                "exp_tag"
            ),

            "method": method,

            "privacy_audit_status": audit_obj.get(
                "overall_status"
            ),

            "privacy_hard_failures": audit_obj.get(
                "num_hard_failures"
            ),

            "recovery_configured": recovery_obj.get(
                "configured_main_recovery"
            ),

            "recovery_preliminary_recommendation": recovery_obj.get(
                "preliminary_overall_recommendation"
            ),
        }

        for key, value in result.items():

            if key == "method":
                continue

            row[
                key
            ] = value

        if method == "riskaware":

            split_summary = get_nested(
                riskaware_summary,
                [
                    "splits",
                    "test",
                ],
                {},
            )

            row[
                "scheduler_fallback_rate_test"
            ] = split_summary.get(
                "scheduler_fallback_rate"
            )

            row[
                "riskaware_max_schedule_spend_test"
            ] = get_nested(
                split_summary,
                [
                    "sbs_schedule_spend",
                    "max",
                ],
                None,
            )

            target_counts = split_summary.get(
                "target_bucket_counts",
                {},
            )

            row[
                "target_bucket1_count_test"
            ] = target_counts.get(
                "1"
            )

        rows.append(
            row
        )

    return rows


# =============================================================================
# Main
# =============================================================================

def main() -> None:

    args = parse_args()

    base_dataset_path = resolve_path(
        args.dataset_config
    )

    base_exp_path = resolve_path(
        args.exp_config
    )

    raw_dataset_cfg = load_yaml(
        base_dataset_path
    )

    raw_exp_cfg = load_yaml(
        base_exp_path
    )

    validate_structural_sweep(
        raw_exp_cfg
    )

    profiles, seeds = (
        resolve_single_and_grid(
            raw_exp_cfg,
            args,
        )
    )

    base_profile = str(
        raw_exp_cfg.get(
            "active_budget_profile",
            raw_exp_cfg.get(
                "default_budget_profile",
                profiles[0],
            ),
        )
    )

    base_seed = int(
        raw_exp_cfg.get(
            "random_seed",
            seeds[0],
        )
    )

    timestamp = dt.datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    run_dir = (
        resolve_path(
            args.run_root
        )
        / (
            f"rebuttal_{args.mode}_"
            f"{timestamp}"
        )
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    generated_dataset_path = (
        make_dataset_config(
            raw_dataset_cfg,
            run_dir,
            args.raw_sample_size_override,
        )
    )

    selected_stages = (
        list(
            args.stages
        )
        if args.stages
        else list(
            ALL_STAGES
        )
    )

    manifest: Dict[
        str,
        Any,
    ] = {
        "created_at_utc": utc_now_iso(),

        "project_root": str(
            PROJECT_ROOT
        ),

        "git": git_metadata(),

        "base_dataset_config": str(
            base_dataset_path
        ),

        "base_exp_config": str(
            base_exp_path
        ),

        "generated_dataset_config": str(
            generated_dataset_path
        ),

        "mode": args.mode,

        "profiles": profiles,

        "seeds": seeds,

        "selected_stages": (
            selected_stages
        ),

        "raw_sample_size_override": (
            args.raw_sample_size_override
        ),

        "joint_mode": (
            args.joint_mode
        ),

        "rerun": bool(
            args.rerun
        ),

        "runs": [],
    }

    manifest_path = (
        run_dir
        / "run_manifest.json"
    )

    save_json(
        manifest_path,
        manifest,
    )

    # -------------------------------------------------------------------------
    # Shared Phase 1-2 are executed at most once because profile/seed do not
    # change the dataset variant or SBS construction.
    # -------------------------------------------------------------------------

    first_exp_path = (
        make_exp_config(
            raw_exp_cfg,
            run_dir,
            profiles[0],
            seeds[0],
        )
    )

    first_dcfg, first_ecfg = (
        resolved_configs(
            generated_dataset_path,
            first_exp_path,
        )
    )

    first_markers = marker_paths(
        first_dcfg,
        first_ecfg,
    )

    shared_records: List[
        Dict[str, Any]
    ] = []

    for stage in [
        "preprocess",
        "events",
    ]:

        if stage not in selected_stages:
            continue

        marker = first_markers[
            stage
        ]

        if (
            not args.rerun
            and marker_is_valid(
                stage,
                marker,
            )
        ):

            print(
                f"[SKIP] {stage}: "
                f"existing marker {marker}"
            )

            shared_records.append(
                {
                    "stage": stage,
                    "status": "SKIPPED_EXISTING",
                    "marker": str(
                        marker
                    ),
                }
            )

            continue

        command = stage_command(
            stage,
            args.python,
            generated_dataset_path,
            first_exp_path,
            args,
            do_joint=False,
        )

        result = run_command(
            command,
            run_dir
            / "logs"
            / f"shared_{stage}.log",
        )

        if not marker_is_valid(
            stage,
            marker,
        ):

            raise RuntimeError(
                f"Stage {stage} finished but "
                f"marker was not created: {marker}"
            )

        result.update(
            {
                "stage": stage,
                "status": "DONE",
                "marker": str(
                    marker
                ),
            }
        )

        shared_records.append(
            result
        )

    manifest[
        "shared_stages"
    ] = shared_records

    save_json(
        manifest_path,
        manifest,
    )

    # -------------------------------------------------------------------------
    # Per-profile x seed experiments.
    # -------------------------------------------------------------------------

    aggregate_rows: List[
        Dict[str, Any]
    ] = []

    per_experiment_stages = [
        stage
        for stage in ALL_STAGES
        if stage not in {
            "preprocess",
            "events",
        }
        and stage in selected_stages
    ]

    for profile in profiles:

        for seed in seeds:

            exp_path = make_exp_config(
                raw_exp_cfg,
                run_dir,
                profile,
                seed,
            )

            dcfg, ecfg = (
                resolved_configs(
                    generated_dataset_path,
                    exp_path,
                )
            )

            markers = marker_paths(
                dcfg,
                ecfg,
            )

            exp_tag = str(
                ecfg[
                    "exp_tag"
                ]
            )

            experiment_record: Dict[
                str,
                Any,
            ] = {
                "profile": profile,
                "seed": int(
                    seed
                ),
                "exp_tag": exp_tag,
                "generated_exp_config": str(
                    exp_path
                ),
                "resolved_summary": pretty_version_summary(
                    load_yaml(
                        generated_dataset_path
                    ),
                    load_yaml(
                        exp_path
                    ),
                ),
                "stages": [],
            }

            print(
                "\n"
                + "#"
                * 100
            )

            print(
                "# EXPERIMENT "
                f"profile={profile}, "
                f"seed={seed}, "
                f"exp_tag={exp_tag}"
            )

            print(
                "#"
                * 100
            )

            for stage in (
                per_experiment_stages
            ):

                marker = markers[
                    stage
                ]

                if (
                    not args.rerun
                    and marker_is_valid(
                        stage,
                        marker,
                    )
                ):

                    print(
                        f"[SKIP] {stage}: "
                        f"existing marker {marker}"
                    )

                    experiment_record[
                        "stages"
                    ].append(
                        {
                            "stage": stage,
                            "status": (
                                "SKIPPED_EXISTING"
                            ),
                            "marker": str(
                                marker
                            ),
                        }
                    )

                    continue

                if (
                    args.joint_mode
                    == "all"
                ):

                    do_joint = True

                elif (
                    args.joint_mode
                    == "none"
                ):

                    do_joint = False

                else:

                    do_joint = (
                        profile
                        == base_profile
                        and seed
                        == base_seed
                    )

                command = (
                    stage_command(
                        stage,
                        args.python,
                        generated_dataset_path,
                        exp_path,
                        args,
                        do_joint=(
                            do_joint
                        ),
                    )
                )

                log_name = (
                    f"{profile}_seed{seed}_"
                    f"{stage}.log"
                )

                result = run_command(
                    command,
                    run_dir
                    / "logs"
                    / log_name,
                )

                if not marker_is_valid(
                    stage,
                    marker,
                ):

                    raise RuntimeError(
                        f"Stage {stage} finished "
                        f"but marker is invalid/missing: "
                        f"{marker}"
                    )

                result.update(
                    {
                        "stage": stage,
                        "status": "DONE",
                        "marker": str(
                            marker
                        ),
                    }
                )

                experiment_record[
                    "stages"
                ].append(
                    result
                )

                # A privacy-audit failure normally exits non-zero. This is an
                # additional explicit semantic check for reused or modified
                # files.
                if stage == "privacy_audit":

                    audit = load_json(
                        marker
                    )

                    if (
                        audit.get(
                            "overall_status"
                        )
                        != "PASS"
                        or int(
                            audit.get(
                                "num_hard_failures",
                                -1,
                            )
                        )
                        != 0
                    ):

                        raise RuntimeError(
                            "Privacy audit did not PASS."
                        )

            manifest[
                "runs"
            ].append(
                experiment_record
            )

            save_json(
                manifest_path,
                manifest,
            )

            if markers[
                "evaluation"
            ].exists():

                aggregate_rows.extend(
                    collect_experiment_result(
                        profile,
                        seed,
                        dcfg,
                        ecfg,
                        markers,
                    )
                )

    # -------------------------------------------------------------------------
    # Aggregate outputs for rebuttal tables.
    # -------------------------------------------------------------------------

    aggregate_csv_path = (
        run_dir
        / "rebuttal_aggregate.csv"
    )

    aggregate_json_path = (
        run_dir
        / "rebuttal_aggregate.json"
    )

    save_csv(
        aggregate_csv_path,
        aggregate_rows,
    )

    save_json(
        aggregate_json_path,
        {
            "created_at_utc": utc_now_iso(),
            "rows": aggregate_rows,
        },
    )

    manifest[
        "finished_at_utc"
    ] = utc_now_iso()

    manifest[
        "aggregate_csv"
    ] = str(
        aggregate_csv_path
    )

    manifest[
        "aggregate_json"
    ] = str(
        aggregate_json_path
    )

    save_json(
        manifest_path,
        manifest,
    )

    print(
        "\n"
        + "="
        * 100
    )

    print(
        "[run_rebuttal_experiments] DONE"
    )

    print(
        f"Run directory:\n  {run_dir}"
    )

    print(
        f"Manifest:\n  {manifest_path}"
    )

    print(
        f"Aggregate CSV:\n  "
        f"{aggregate_csv_path}"
    )

    print(
        f"Aggregate JSON:\n  "
        f"{aggregate_json_path}"
    )

    print(
        "="
        * 100
    )


if __name__ == "__main__":
    main()
