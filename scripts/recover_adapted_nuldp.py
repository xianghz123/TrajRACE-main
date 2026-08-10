#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""Server-only recovery for the adapted non-uniform LDP comparator.

Consumes ONLY server-visible adapted_nuldp reports and public domains.
For each public current road segment U=u, it jointly estimates the latent
P(V,K | U=u) from observed (Y,K_tilde) using the known product channel

  A_u[(y,k~),(v,k)] = Q_u^k(v,y) * M(k,k~),

where Q is GRR over public N(u) with epsilon_event[k], and M is GRR over
bucket labels with epsilon_bucket.  The recovered next-hop distribution is
P(V|U)=sum_k P(V,K|U).

This avoids assuming independence between execution bucket and successor and
makes the adapted heterogeneous comparator compatible with the same shuffled
server-report schema used by TrajRACE.
"""

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
try:
    from scipy.optimize import nnls
except Exception as exc:  # pragma: no cover
    raise RuntimeError("scipy is required: pip install scipy") from exc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from trajrace.versioning_utils import apply_versioning, pretty_version_summary

DEFAULT_DATASET_CONFIG = "configs/dataset.yaml"
DEFAULT_EXP_CONFIG = "configs/exp_main.yaml"
METHOD = "adapted_nuldp"


def resolve_path(path: Optional[str]) -> Optional[Path]:
    if path is None:
        return None
    p = Path(path)
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def load_yaml(path: Path) -> Dict[str, Any]:
    import yaml
    with path.open("r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"YAML root must be mapping: {path}")
    return obj


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON: {path}:{line_no}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Expected JSON object: {path}:{line_no}")
            yield obj


def alias(exp: Mapping[str, Any], names: Sequence[str], default: Any = None, required: bool = True) -> Any:
    for name in names:
        if name in exp and exp[name] is not None:
            return exp[name]
    if required:
        raise KeyError(f"Missing config key; expected one of {list(names)}")
    return default


def extract_cfg(exp: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "eps_start": float(alias(exp, ["eps_start", "epsilon_s"])),
        "eps_count": float(alias(exp, ["eps_count", "eps_len", "epsilon_c"])),
        "eps_bucket": float(alias(exp, ["eps_bucket", "eps_buck", "epsilon_b"])),
        "eps_event_list": [float(x) for x in alias(exp, ["eps_event_list", "eps_evt_list", "epsilon_e_list"])],
        "K": int(alias(exp, ["K"])),
        "L_max": int(alias(exp, ["L_max", "L_m"])),
    }


def rr_params(eps: float, d: int) -> Tuple[float, float]:
    if d <= 0:
        raise ValueError("GRR domain empty")
    if d == 1:
        return 1.0, 0.0
    ee = math.exp(float(eps))
    return ee / (ee + d - 1), 1.0 / (ee + d - 1)


def rr_prob(true_idx: int, out_idx: int, eps: float, d: int) -> float:
    if d == 1:
        return 1.0
    p, q = rr_params(eps, d)
    return p if true_idx == out_idx else q


def clipped_rr_inverse(counter: Counter, domain: Sequence[Any], eps: float) -> Dict[str, float]:
    domain = list(domain)
    n = sum(counter.values())
    d = len(domain)
    if n <= 0:
        return {}
    p, q = rr_params(eps, d)
    den = p - q
    est: Dict[str, float] = {}
    for x in domain:
        obs = counter.get(x, 0) / n
        raw = (obs - q) / den if abs(den) > 1e-15 else obs
        if raw > 0:
            est[str(x)] = float(raw)
    s = sum(est.values())
    if s <= 0:
        est = {str(x): counter.get(x, 0) / n for x in domain if counter.get(x, 0) > 0}
        s = sum(est.values())
    return {k: v / s for k, v in est.items()} if s > 0 else {}


def sparse_start_inverse(counter: Counter, public_domain_size: int, eps: float) -> Dict[str, float]:
    n = sum(counter.values())
    if n <= 0:
        return {}
    p, q = rr_params(eps, public_domain_size)
    den = p - q
    est: Dict[str, float] = {}
    for x, c in counter.items():
        obs = c / n
        raw = (obs - q) / den if abs(den) > 1e-15 else obs
        if raw > 0:
            est[str(x)] = float(raw)
    s = sum(est.values())
    if s <= 0:
        est = {str(x): c / n for x, c in counter.items()}
        s = sum(est.values())
    return {k: v / s for k, v in est.items()} if s > 0 else {}


def build_channel(domain: Sequence[str], eps_event_list: Sequence[float], eps_bucket: float) -> np.ndarray:
    d = len(domain)
    K = len(eps_event_list)
    # row = observed (y_idx, ktilde_idx), col = latent (v_idx, k_idx)
    A = np.zeros((d * K, d * K), dtype=float)
    for y_idx in range(d):
        for kt_idx in range(K):
            row = y_idx * K + kt_idx
            for v_idx in range(d):
                for k_idx in range(K):
                    col = v_idx * K + k_idx
                    qv = rr_prob(v_idx, y_idx, float(eps_event_list[k_idx]), d)
                    qb = rr_prob(k_idx, kt_idx, float(eps_bucket), K)
                    A[row, col] = qv * qb
    return A


def recover_joint(
    counter: Counter,
    domain: Sequence[str],
    eps_event_list: Sequence[float],
    eps_bucket: float,
    mass_weight: float,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    domain = [str(x) for x in domain]
    d = len(domain)
    K = len(eps_event_list)
    n = int(sum(counter.values()))
    if n <= 0:
        return {}, {"num_reports": 0, "nnls_residual": None}
    if d == 1:
        return {domain[0]: 1.0}, {"num_reports": n, "nnls_residual": 0.0, "singleton": True}

    A = build_channel(domain, eps_event_list, eps_bucket)
    b = np.zeros(d * K, dtype=float)
    for y_idx, y in enumerate(domain):
        for kt_idx in range(K):
            b[y_idx * K + kt_idx] = float(counter.get((y, kt_idx + 1), 0)) / n

    # NNLS with an explicit soft sum-to-one constraint.  The final vector is
    # normalized again, so nonnegativity is the important hard constraint.
    mw = max(1.0, float(mass_weight))
    A_aug = np.vstack([A, mw * np.ones((1, d * K), dtype=float)])
    b_aug = np.concatenate([b, np.array([mw], dtype=float)])
    x, residual = nnls(A_aug, b_aug)
    s = float(x.sum())
    if s <= 1e-15:
        # Stable fallback: ignore noisy bucket labels and invert the y marginal
        y_counter = Counter()
        for (y, _k), c in counter.items():
            y_counter[y] += c
        # Use the strongest channel as a conservative fallback; this branch
        # should be rare and is explicitly counted in diagnostics.
        dist = clipped_rr_inverse(y_counter, domain, float(eps_event_list[0]))
        return dist, {
            "num_reports": n,
            "nnls_residual": float(residual),
            "fallback": "y_marginal_strongest_channel",
        }
    x /= s
    p_v = np.zeros(d, dtype=float)
    p_k = np.zeros(K, dtype=float)
    for v_idx in range(d):
        for k_idx in range(K):
            z = float(x[v_idx * K + k_idx])
            p_v[v_idx] += z
            p_k[k_idx] += z
    p_v /= max(float(p_v.sum()), 1e-15)
    cond = float(np.linalg.cond(A)) if A.size else float("nan")
    return (
        {domain[i]: float(p_v[i]) for i in range(d) if p_v[i] > 0},
        {
            "num_reports": n,
            "nnls_residual": float(residual),
            "channel_condition_number": cond,
            "latent_bucket_mass": {str(k + 1): float(p_k[k]) for k in range(K)},
        },
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Recover adapted non-uniform LDP reports")
    p.add_argument("--dataset_config", default=DEFAULT_DATASET_CONFIG)
    p.add_argument("--exp_config", default=DEFAULT_EXP_CONFIG)
    p.add_argument("--mass_weight", type=float, default=10.0)
    p.add_argument("--progress_every", type=int, default=10000)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dpath = resolve_path(args.dataset_config)
    epath = resolve_path(args.exp_config)
    assert dpath is not None and epath is not None
    raw_d, raw_e = load_yaml(dpath), load_yaml(epath)
    dcfg, ecfg = apply_versioning(raw_d, raw_e)
    cfg = extract_cfg(ecfg)
    if len(cfg["eps_event_list"]) != cfg["K"]:
        raise ValueError("eps_event_list length must equal K")

    succ_path = resolve_path(str(dcfg["successor_cache_path"]))
    priv_dir = resolve_path(str(ecfg["privatized_dir"]))
    rec_dir = resolve_path(str(ecfg["recovered_dir"]))
    assert succ_path is not None and priv_dir is not None and rec_dir is not None
    rec_dir.mkdir(parents=True, exist_ok=True)
    succ = load_json(succ_path)

    print("=" * 100)
    print("[recover_adapted_nuldp]")
    try:
        print(json.dumps(pretty_version_summary(raw_d, raw_e), ensure_ascii=False, indent=2))
    except Exception:
        pass
    print("Joint latent recovery: P(V,K|U) from observed (Y,K~)")
    print("=" * 100)

    all_summary: Dict[str, Any] = {}
    for split in ("train", "valid", "test"):
        report_path = priv_dir / f"{METHOD}_{split}_reports.jsonl"
        if not report_path.exists():
            raise FileNotFoundError(report_path)
        starts = Counter()
        counts = Counter()
        per_u: Dict[str, Counter] = defaultdict(Counter)
        types = Counter()
        for i, r in enumerate(iter_jsonl(report_path), 1):
            typ = str(r["event_type"])
            types[typ] += 1
            if typ == "start":
                starts[str(r["x_noisy"])] += 1
            elif typ == "count":
                counts[int(r["x_noisy"])] += 1
            elif typ == "transition":
                u, y, kt = str(r["u"]), str(r["y"]), int(r["k_noisy"])
                if u not in succ or y not in succ[u]:
                    raise RuntimeError(f"Report outside public N(u): {u}->{y}")
                if not 1 <= kt <= cfg["K"]:
                    raise RuntimeError(f"Noisy bucket outside 1..K: {kt}")
                per_u[u][(y, kt)] += 1
            else:
                raise RuntimeError(f"Unknown event_type={typ}")
            if args.progress_every > 0 and i % args.progress_every == 0:
                print(f"[{split}] reports={i:,}, contexts={len(per_u):,}")

        start_dist = sparse_start_inverse(starts, len(succ), cfg["eps_start"])
        count_dist = clipped_rr_inverse(counts, range(1, cfg["L_max"] + 1), cfg["eps_count"])
        trans: Dict[str, Any] = {}
        residuals: List[float] = []
        fallback_contexts = 0
        for j, (u, c) in enumerate(per_u.items(), 1):
            domain = [str(x) for x in succ.get(u, [])]
            if not domain:
                raise RuntimeError(f"Empty public N(u): {u}")
            dist, diag = recover_joint(
                c, domain, cfg["eps_event_list"], cfg["eps_bucket"], args.mass_weight
            )
            if diag.get("fallback"):
                fallback_contexts += 1
            if diag.get("nnls_residual") is not None:
                residuals.append(float(diag["nnls_residual"]))
            trans[u] = {
                "domain_size": len(domain),
                "num_reports": int(sum(c.values())),
                "distribution": dist,
                "recovery_diagnostic": diag,
            }
            if args.progress_every > 0 and j % args.progress_every == 0:
                print(f"[{split}] recovered contexts={j:,}/{len(per_u):,}")

        start_obj = {"domain_size": len(succ), "num_reports": int(sum(starts.values())), "distribution": start_dist}
        count_obj = {"domain_size": cfg["L_max"], "num_reports": int(sum(counts.values())), "distribution": count_dist}
        save_json(rec_dir / f"{METHOD}_{split}_start.json", start_obj)
        save_json(rec_dir / f"{METHOD}_{split}_count.json", count_obj)
        save_json(rec_dir / f"{METHOD}_{split}_transition_context.json", trans)
        all_summary[split] = {
            "report_counts": dict(types),
            "num_transition_contexts": len(trans),
            "joint_recovery": True,
            "fallback_contexts": fallback_contexts,
            "mean_nnls_residual": (sum(residuals) / len(residuals) if residuals else None),
        }
        print(
            f"[{split}] DONE contexts={len(trans):,} transitions={types['transition']:,} "
            f"fallback_contexts={fallback_contexts}"
        )

    out = {
        "dataset_name": dcfg.get("dataset_name"),
        "dataset_variant": dcfg.get("dataset_variant"),
        "exp_tag": ecfg.get("exp_tag"),
        "method": METHOD,
        "recovery": "joint latent P(V,K|U) NNLS; marginalize K",
        "server_only_inputs": True,
        "uses_private_debug": False,
        "splits": all_summary,
    }
    save_json(rec_dir / f"{METHOD}_recovery_summary.json", out)
    print("[recover_adapted_nuldp] DONE")


if __name__ == "__main__":
    main()
