#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
scripts/evaluate_independent_attack.py

R1-W3/D3: mechanism-independent auxiliary-data next-hop attack.

Purpose
-------
Reviewer 1 correctly noted that HR-Keep is mechanism-targeted because the
"high-risk" subset is defined using the same TrajRACE risk score that controls
the protection level.  This script therefore evaluates a separate inference
attacker whose prediction features do NOT contain TrajRACE's private risk
score or bucket-selection internals.

Attacker protocol
-----------------
Training/calibration:
    - disjoint auxiliary split, default=train;
    - the attacker may have auxiliary labeled trajectories, hence V is known
      only on the auxiliary split;
    - for each method separately, it learns empirical next-hop posteriors from
      (U, Y, K_noisy) -> V, with hierarchical shrinkage/backoff.

Test-time attacker inputs:
    - public current segment U;
    - server-visible privatized successor report Y;
    - server-visible noisy bucket label K~ (= k_noisy);
    - public successor domain N(U).

Explicitly excluded:
    - TrajRACE composite risk score r_i / R_t;
    - target-risk bucket b_i;
    - true execution bucket k_i;
    - HR-Keep labels/subsets;
    - any test-set true successor except for final offline scoring.

Primary evaluation:
    branching contexts |N(U)| > 1.

Metrics
-------
context_prior_top1:
    Top-1 accuracy using only the auxiliary P(V|U) prior.

posterior_top1:
    Top-1 accuracy using the auxiliary-data model P(V|U,Y,K~).

attack_advantage:
    posterior_top1 - context_prior_top1.

The model is deliberately trained only on the auxiliary split.  The test
release is used only for inference/scoring, so there is no test-release
calibration step.

Required local debug files
--------------------------
The canonical privatizers must be run with --save_debug.  Debug truth is local
and is used only to train the auxiliary attacker or score the test attacker.
The server report schema remains unchanged.

Expected aligned debug transition fields:
    event_type, u, true_v, y, k_noisy

IMPORTANT:
Never align shuffled server reports with debug files by row order.
Each debug row must already contain its own aligned (u,true_v,y,k_noisy).
"""

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from trajrace.versioning_utils import apply_versioning


# =============================================================================
# I/O and configuration helpers
# =============================================================================

def resolve_path(pathlike: str | Path) -> Path:
    p = Path(pathlike)
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def load_yaml(path: Path) -> Dict[str, Any]:
    import yaml
    with path.open("r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return obj


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_no}")
            yield obj


def activate_privacy_profile(raw_exp: Mapping[str, Any]) -> Dict[str, Any]:
    """Merge active privacy profile into top-level aliases used by old/new code."""
    exp = dict(raw_exp)
    profiles = exp.get("privacy_profiles") or {}
    profile = str(
        exp.get(
            "active_budget_profile",
            exp.get("default_budget_profile", "B1p0"),
        )
    )
    if isinstance(profiles, dict) and profile in profiles:
        prof = profiles[profile]
        if not isinstance(prof, dict):
            raise TypeError(f"privacy_profiles[{profile!r}] must be a mapping")
        exp.update(prof)

    alias_sets = {
        "B_total": ("B_total", "B", "budget"),
        "eps_start": ("eps_start", "epsilon_s"),
        "eps_count": ("eps_count", "eps_len", "epsilon_c"),
        "eps_bucket": ("eps_bucket", "eps_buck", "epsilon_b"),
        "eps_event_list": ("eps_event_list", "eps_evt_list", "epsilon_e_list"),
        "K": ("K",),
        "L_max": ("L_max", "L_m"),
    }
    for canonical, names in alias_sets.items():
        for name in names:
            if name in exp and exp[name] is not None:
                exp[canonical] = exp[name]
                break

    if "eps_count" in exp:
        exp.setdefault("eps_len", exp["eps_count"])
    if "eps_bucket" in exp:
        exp.setdefault("eps_buck", exp["eps_bucket"])
    if "eps_event_list" in exp:
        exp.setdefault("eps_evt_list", exp["eps_event_list"])

    exp["active_budget_profile"] = profile
    exp["default_budget_profile"] = profile
    return exp


def resolved_configs(dataset_config: Path, exp_config: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    raw_d = load_yaml(dataset_config)
    raw_e = activate_privacy_profile(load_yaml(exp_config))
    dcfg, ecfg = apply_versioning(raw_d, raw_e)

    # Compatibility fallbacks for older versioning_utils.py.
    if "experiment_root" not in ecfg:
        dataset_root = resolve_path(dcfg["dataset_root"])
        exp_root = dataset_root / "experiments" / str(ecfg["exp_tag"])
        ecfg["experiment_root"] = str(exp_root)
        ecfg.setdefault("privatized_dir", str(exp_root / "privatized"))
    return dcfg, ecfg


# =============================================================================
# Debug schema validation
# =============================================================================

REQUIRED_FIELDS = ("u", "true_v", "y", "k_noisy")
BANNED_ATTACK_FIELDS = (
    "risk",
    "risk_score",
    "r_i",
    "R_t",
    "target_bucket",
    "target_k",
    "true_bucket",
    "execution_bucket",
    "k_true",
    "high_risk",
    "hr_keep",
)


def transition_debug_rows(path: Path) -> Iterable[Dict[str, Any]]:
    for rec in iter_jsonl(path):
        if rec.get("event_type") != "transition":
            continue
        missing = [k for k in REQUIRED_FIELDS if k not in rec]
        if missing:
            raise KeyError(
                f"Aligned transition debug row in {path} is missing fields: {missing}"
            )
        # We intentionally do not read any risk/bucket-internal fields.
        yield {
            "u": str(rec["u"]),
            "true_v": str(rec["true_v"]),
            "y": str(rec["y"]),
            "k_noisy": int(rec["k_noisy"]),
        }


# =============================================================================
# Auxiliary attacker
# =============================================================================

class AuxiliaryAttackModel:
    """Hierarchical empirical posterior with context-prior shrinkage.

    Levels:
        (U,Y,K~) -> (U,Y) -> U

    The full conditional automatically captures any dependence between
    the protected successor and the noisy bucket observation without assuming
    K independent of V.  Every level is learned only from the auxiliary split.
    """

    def __init__(self, shrinkage_mass: float = 5.0, min_support: int = 3):
        self.shrinkage_mass = float(shrinkage_mass)
        self.min_support = int(min_support)

        self.count_u: Dict[str, Counter[str]] = defaultdict(Counter)
        self.count_uy: Dict[Tuple[str, str], Counter[str]] = defaultdict(Counter)
        self.count_uyk: Dict[Tuple[str, str, int], Counter[str]] = defaultdict(Counter)
        self.global_v: Counter[str] = Counter()
        self.n_aux = 0

    def fit(self, rows: Iterable[Mapping[str, Any]]) -> None:
        for r in rows:
            u = str(r["u"])
            v = str(r["true_v"])
            y = str(r["y"])
            kn = int(r["k_noisy"])
            self.count_u[u][v] += 1
            self.count_uy[(u, y)][v] += 1
            self.count_uyk[(u, y, kn)][v] += 1
            self.global_v[v] += 1
            self.n_aux += 1

        if self.n_aux <= 0:
            raise RuntimeError("Auxiliary attack split contains no transition debug rows.")

    @staticmethod
    def _normalize(values: Mapping[str, float], domain: Sequence[str]) -> Dict[str, float]:
        z = sum(max(0.0, float(values.get(v, 0.0))) for v in domain)
        if z <= 0:
            return {v: 1.0 / len(domain) for v in domain}
        return {v: max(0.0, float(values.get(v, 0.0))) / z for v in domain}

    def context_prior(self, u: str, domain: Sequence[str]) -> Dict[str, float]:
        if not domain:
            return {}
        c = self.count_u.get(u, Counter())
        # Laplace smoothing only inside the public successor domain N(U).
        vals = {v: float(c.get(v, 0)) + 1.0 for v in domain}
        return self._normalize(vals, domain)

    def _shrink(
        self,
        local_counts: Counter[str],
        backoff: Mapping[str, float],
        domain: Sequence[str],
    ) -> Dict[str, float]:
        n = sum(local_counts.get(v, 0) for v in domain)
        mass = self.shrinkage_mass
        vals = {
            v: float(local_counts.get(v, 0)) + mass * float(backoff.get(v, 0.0))
            for v in domain
        }
        return self._normalize(vals, domain)

    def posterior(self, u: str, y: str, kn: int, domain: Sequence[str]) -> Tuple[Dict[str, float], str, int]:
        prior = self.context_prior(u, domain)

        c_uy = self.count_uy.get((u, y), Counter())
        n_uy = sum(c_uy.get(v, 0) for v in domain)
        if n_uy >= self.min_support:
            p_uy = self._shrink(c_uy, prior, domain)
            level_uy = "U,Y"
        else:
            p_uy = prior
            level_uy = "U"

        c_uyk = self.count_uyk.get((u, y, int(kn)), Counter())
        n_uyk = sum(c_uyk.get(v, 0) for v in domain)
        if n_uyk >= self.min_support:
            p = self._shrink(c_uyk, p_uy, domain)
            return p, "U,Y,K~", n_uyk

        return p_uy, level_uy, n_uy


def entropy(prob: Mapping[str, float]) -> float:
    return -sum(p * math.log(p) for p in prob.values() if p > 0)


def argmax_label(prob: Mapping[str, float], domain: Sequence[str]) -> str:
    # Stable deterministic tie break using the public domain order.
    best = domain[0]
    best_p = float(prob.get(best, 0.0))
    for v in domain[1:]:
        p = float(prob.get(v, 0.0))
        if p > best_p:
            best, best_p = v, p
    return best


def evaluate_one_method(
    method: str,
    aux_debug: Path,
    test_debug: Path,
    successor_domains: Mapping[str, Sequence[str]],
    shrinkage_mass: float,
    min_support: int,
) -> List[Dict[str, Any]]:

    model = AuxiliaryAttackModel(
        shrinkage_mass=shrinkage_mass,
        min_support=min_support,
    )
    model.fit(transition_debug_rows(aux_debug))

    # Separate all and branching scopes.
    stats = {
        "all": Counter(),
        "branching": Counter(),
    }
    sums = {
        "all": defaultdict(float),
        "branching": defaultdict(float),
    }
    backoff_usage = {
        "all": Counter(),
        "branching": Counter(),
    }

    for r in transition_debug_rows(test_debug):
        u = str(r["u"])
        true_v = str(r["true_v"])
        y = str(r["y"])
        kn = int(r["k_noisy"])

        domain = [str(v) for v in successor_domains.get(u, [])]
        if not domain:
            continue
        if true_v not in domain:
            raise ValueError(
                f"Test true successor {true_v!r} not in public N({u})"
            )
        if y not in domain:
            raise ValueError(
                f"Server-visible report y={y!r} not in public N({u})"
            )

        prior = model.context_prior(u, domain)
        post, level, support = model.posterior(u, y, kn, domain)

        prior_pred = argmax_label(prior, domain)
        post_pred = argmax_label(post, domain)

        scopes = ["all"]
        if len(domain) > 1:
            scopes.append("branching")

        for scope in scopes:
            stats[scope]["n"] += 1
            stats[scope]["prior_correct"] += int(prior_pred == true_v)
            stats[scope]["post_correct"] += int(post_pred == true_v)

            p_true = max(float(post.get(true_v, 0.0)), 1e-15)
            h = entropy(post)
            h_norm = h / math.log(len(domain)) if len(domain) > 1 else 0.0

            sums[scope]["posterior_entropy"] += h
            sums[scope]["normalized_entropy"] += h_norm
            sums[scope]["nll"] += -math.log(p_true)
            sums[scope]["true_posterior_prob"] += p_true
            sums[scope]["local_support"] += float(support)
            backoff_usage[scope][level] += 1

    rows: List[Dict[str, Any]] = []
    for scope in ("all", "branching"):
        n = int(stats[scope]["n"])
        if n <= 0:
            continue
        prior_acc = stats[scope]["prior_correct"] / n
        post_acc = stats[scope]["post_correct"] / n

        rows.append(
            {
                "method": method,
                "scope": scope,
                "n_test_transitions": n,
                "n_aux_transitions": model.n_aux,
                "context_prior_top1": prior_acc,
                "posterior_top1": post_acc,
                "attack_advantage": post_acc - prior_acc,
                "mean_posterior_entropy": sums[scope]["posterior_entropy"] / n,
                "mean_normalized_entropy": sums[scope]["normalized_entropy"] / n,
                "mean_nll": sums[scope]["nll"] / n,
                "mean_true_posterior_prob": sums[scope]["true_posterior_prob"] / n,
                "mean_aux_local_support": sums[scope]["local_support"] / n,
                "backoff_UYK_count": int(backoff_usage[scope]["U,Y,K~"]),
                "backoff_UY_count": int(backoff_usage[scope]["U,Y"]),
                "backoff_U_count": int(backoff_usage[scope]["U"]),
                "uses_test_release_for_calibration": False,
                "uses_risk_score": False,
                "uses_target_bucket": False,
                "uses_true_execution_bucket": False,
                "uses_hr_keep_label": False,
            }
        )
    return rows


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Disjoint auxiliary-data next-hop attack for TrajRACE R1-W3/D3."
    )
    p.add_argument("--dataset_config", default="configs/dataset.yaml")
    p.add_argument("--exp_config", default="configs/exp_main.yaml")
    p.add_argument("--aux_split", default="train")
    p.add_argument("--test_split", default="test")
    p.add_argument(
        "--methods",
        nargs="+",
        default=["riskaware", "uniform"],
        choices=["riskaware", "uniform"],
    )
    p.add_argument(
        "--shrinkage_mass",
        type=float,
        default=5.0,
        help="Dirichlet-style shrinkage mass toward the context prior.",
    )
    p.add_argument(
        "--min_support",
        type=int,
        default=3,
        help="Minimum auxiliary support before using a finer conditional level.",
    )
    p.add_argument("--output_dir", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.aux_split == args.test_split:
        raise ValueError(
            "Auxiliary and test splits must be disjoint for the rebuttal attack."
        )

    dataset_cfg_path = resolve_path(args.dataset_config)
    exp_cfg_path = resolve_path(args.exp_config)
    dcfg, ecfg = resolved_configs(dataset_cfg_path, exp_cfg_path)

    succ_path = resolve_path(dcfg["successor_cache_path"])
    successor_domains = {
        str(u): [str(v) for v in vs]
        for u, vs in load_json(succ_path).items()
    }

    priv_dir_raw = ecfg.get("privatized_dir", dcfg.get("privatized_dir"))
    if not priv_dir_raw:
        raise KeyError("Resolved config has no privatized_dir")
    priv_dir = resolve_path(priv_dir_raw)

    out_dir = (
        resolve_path(args.output_dir)
        if args.output_dir
        else resolve_path(ecfg["experiment_root"]) / "w3_attack"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict[str, Any]] = []

    for method in args.methods:
        aux_debug = priv_dir / f"{method}_{args.aux_split}_debug.jsonl"
        test_debug = priv_dir / f"{method}_{args.test_split}_debug.jsonl"

        for pth in (aux_debug, test_debug):
            if not pth.exists():
                raise FileNotFoundError(
                    f"Missing local debug trace: {pth}\n"
                    "Rerun canonical privatization with --save_debug. "
                    "Do not publish *_debug.jsonl files in the final artifact."
                )

        rows = evaluate_one_method(
            method=method,
            aux_debug=aux_debug,
            test_debug=test_debug,
            successor_domains=successor_domains,
            shrinkage_mass=args.shrinkage_mass,
            min_support=args.min_support,
        )
        for row in rows:
            row["dataset_name"] = dcfg.get("dataset_name")
            row["dataset_variant"] = dcfg.get("dataset_variant")
            row["exp_tag"] = ecfg.get("exp_tag")
            row["aux_split"] = args.aux_split
            row["test_split"] = args.test_split
        all_rows.extend(rows)

    save_csv(out_dir / "attack_results.csv", all_rows)
    save_json(
        out_dir / "attack_results.json",
        {
            "dataset_name": dcfg.get("dataset_name"),
            "dataset_variant": dcfg.get("dataset_variant"),
            "exp_tag": ecfg.get("exp_tag"),
            "aux_split": args.aux_split,
            "test_split": args.test_split,
            "primary_scope": "branching",
            "attack_advantage_definition": (
                "posterior Top-1 P(V|U,Y,K~) minus context-only Top-1 P(V|U)"
            ),
            "test_time_inputs": [
                "public U",
                "server-visible Y",
                "server-visible K~ (k_noisy)",
                "public N(U)",
            ],
            "auxiliary_training_labels": [
                "true successor V on disjoint auxiliary split only"
            ],
            "explicitly_excluded": list(BANNED_ATTACK_FIELDS),
            "test_release_used_for_calibration": False,
            "results": all_rows,
        },
    )

    print(json.dumps(all_rows, ensure_ascii=False, indent=2))
    print(f"[DONE] independent attack -> {out_dir}")


if __name__ == "__main__":
    main()
