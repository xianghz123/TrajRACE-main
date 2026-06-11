# trajrace/risk_utils.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-



from __future__ import annotations

import math
from collections import deque
from typing import Any, Dict, List, Tuple, Optional


# =============================================================================
# 基础工具
# =============================================================================

def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def _clip01(x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    return x


def _mean(vals: List[float]) -> float:
    if not vals:
        return 0.0
    return float(sum(vals) / len(vals))


def _quantile_sorted(sorted_vals: List[float], q: float) -> float:
    
    if not sorted_vals:
        return 0.0

    q = min(1.0, max(0.0, float(q)))
    n = len(sorted_vals)
    if n == 1:
        return float(sorted_vals[0])

    idx = q * (n - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return float(sorted_vals[lo])

    w = idx - lo
    return float((1.0 - w) * sorted_vals[lo] + w * sorted_vals[hi])


# =============================================================================
# 图结构
# =============================================================================

def build_segment_graph_from_successor_cache(successor_cache: Dict[str, List[str]]) -> Dict[str, List[str]]:
    
    graph: Dict[str, List[str]] = {}
    for u, vs in successor_cache.items():
        if isinstance(vs, list):
            graph[u] = list(vs)
        else:
            graph[u] = []
    return graph


# =============================================================================
# 风险分量
# =============================================================================

def _endpoint_risk(
    pos: int,
    traj_len: int,
    sigma_e: float,
) -> Tuple[float, int, int]:
    
    T = max(1, traj_len - 1)
    sigma = max(float(sigma_e), 1e-6)

    d_start = max(0, pos - 1)
    d_end = max(0, T - pos)

    risk_start = math.exp(-float(d_start) / sigma)
    risk_end = math.exp(-float(d_end) / sigma)
    phi = max(risk_start, risk_end)

    near_start = 1 if d_start <= max(1, int(math.ceil(sigma))) else 0
    near_end = 1 if d_end <= max(1, int(math.ceil(sigma))) else 0

    return _clip01(phi), near_start, near_end


def _stay_risk(
    delta_t: int,
    pos: int,
    traj_len: int,
    T_stay: float,
    sigma_s: float,
) -> float:
    
    dt = max(0.0, float(delta_t))
    T0 = max(float(T_stay), 1e-6)
    sigma = max(float(sigma_s), 1e-6)

    
    stay_strength = 1.0 - math.exp(-dt / T0)

    
    T = max(1, traj_len - 1)
    d_start = max(0, pos - 1)
    d_end = max(0, T - pos)
    endpoint_boost = max(
        math.exp(-float(d_start) / sigma),
        math.exp(-float(d_end) / sigma)
    )

    phi = 0.7 * stay_strength + 0.3 * stay_strength * endpoint_boost
    return _clip01(phi)


def _deg_risk(
    u: Any,
    successor_cache: Dict[str, List[str]],
) -> Tuple[float, int]:
    
    cand = successor_cache.get(u, [])
    candidate_size = len(cand)

    if candidate_size <= 0:
        return 1.0, 1
    if candidate_size == 1:
        return 1.0, 1

   
    phi = 1.0 / float(candidate_size)
    return _clip01(phi), int(candidate_size)


# =============================================================================
# 单条轨迹风险计算
# =============================================================================

def compute_transition_risks_for_record(
    event_record: Dict[str, Any],
    successor_cache: Dict[str, List[str]],
    risk_cfg: Dict[str, Any],
    distance_cache: Dict[str, float],
    segment_graph: Dict[str, List[str]],
) -> List[Dict[str, Any]]:
    
    traj_id = str(event_record["traj_id"])
    transitions = event_record.get("transition_events", [])

    lambda_e = float(risk_cfg["lambda_e"])
    lambda_s = float(risk_cfg["lambda_s"])
    lambda_d = float(risk_cfg["lambda_d"])

    sigma_e = float(risk_cfg["sigma_e"])
    sigma_s = float(risk_cfg["sigma_s"])
    T_stay = float(risk_cfg["T_stay"])

    out: List[Dict[str, Any]] = []

    for tr in transitions:
        pos = _safe_int(tr.get("pos", 0))
        u = tr.get("u")
        v = tr.get("v")
        traj_len = _safe_int(tr.get("traj_len", 0))
        delta_t = max(0, _safe_int(tr.get("delta_t", 0)))

        phi_endpoint, near_start_flag, near_end_flag = _endpoint_risk(
            pos=pos,
            traj_len=traj_len,
            sigma_e=sigma_e,
        )

        phi_stay = _stay_risk(
            delta_t=delta_t,
            pos=pos,
            traj_len=traj_len,
            T_stay=T_stay,
            sigma_s=sigma_s,
        )

        phi_deg, candidate_size = _deg_risk(
            u=u,
            successor_cache=successor_cache,
        )

        R_t = (
            lambda_e * phi_endpoint
            + lambda_s * phi_stay
            + lambda_d * phi_deg
        )
        R_t = _clip01(R_t)

        out.append({
            "traj_id": traj_id,
            "pos": pos,
            "u": u,
            "v": v,
            "traj_len": traj_len,
            "delta_t": delta_t,

            "phi_endpoint": float(phi_endpoint),
            "phi_stay": float(phi_stay),
            "phi_deg": float(phi_deg),
            "R_t": float(R_t),

            # b_t 先占位，后续 summarize_target_buckets() 会统一回写
            "b_t": 0,

            # 诊断字段
            "candidate_size": int(candidate_size),
            "near_start_flag": int(near_start_flag),
            "near_end_flag": int(near_end_flag),
        })

    return out


# =============================================================================
# 稳健分桶
# =============================================================================

def _target_bucket_counts(n: int, K: int) -> List[int]:
    
    if n <= 0:
        return [0] * K

    if K == 1:
        return [n]

    if K == 3:
        raw = [0.15 * n, 0.35 * n, 0.50 * n]
    else:
        raw = [n / K for _ in range(K)]

    counts = [int(math.floor(x)) for x in raw]

    
    for i in range(min(n, K)):
        counts[i] = max(counts[i], 1)

    diff = n - sum(counts)

    
    if diff > 0:
        for i in range(diff):
            counts[i % K] += 1
    elif diff < 0:
        diff = -diff
        
        for _ in range(diff):
            for j in range(K - 1, -1, -1):
                lower_bound = 1 if n >= K else 0
                if counts[j] > lower_bound:
                    counts[j] -= 1
                    break

    return counts


def _assign_buckets_by_rank(risk_items: List[Dict[str, Any]], K: int) -> Tuple[List[int], Dict[str, Any]]:
    
    n = len(risk_items)
    if n == 0:
        return [], {"strategy": "empty", "thresholds": []}

    scores = [float(item.get("R_t", 0.0)) for item in risk_items]
    order = sorted(range(n), key=lambda i: scores[i], reverse=True)

    counts = _target_bucket_counts(n=n, K=K)

    assignments = [K] * n
    start = 0
    for bucket_id, cnt in enumerate(counts, start=1):
        end = min(n, start + cnt)
        for idx in order[start:end]:
            assignments[idx] = bucket_id
        start = end

    
    for idx in order[start:]:
        assignments[idx] = K

    
    sorted_scores = sorted(scores)
    if K == 3:
        # 与 15/35/50 近似对应的分位
        th_high = _quantile_sorted(sorted_scores, 0.85)
        th_mid = _quantile_sorted(sorted_scores, 0.50)
        thresholds = [float(th_high), float(th_mid)]
    else:
        thresholds = []
        for j in range(1, K):
            q = 1.0 - float(sum(counts[:j])) / max(1, n)
            thresholds.append(float(_quantile_sorted(sorted_scores, q)))

    meta = {
        "strategy": "rank_quantile",
        "thresholds": thresholds,
        "counts": counts,
    }
    return assignments, meta


# =============================================================================
# 分桶汇总（会原地回写 risk_items 的 b_t）
# =============================================================================

def summarize_target_buckets(
    risk_items: List[Dict[str, Any]],
    K: int,
) -> Dict[str, Any]:
    
    K = max(1, int(K))

    if not risk_items:
        return {
            str(b): {
                "count": 0,
                "avg_R_t": 0.0,
                "avg_phi_endpoint": 0.0,
                "avg_phi_stay": 0.0,
                "avg_phi_deg": 0.0,
                "avg_delta_t": 0.0,
            }
            for b in range(1, K + 1)
        }

    assignments, _meta = _assign_buckets_by_rank(risk_items, K=K)

    for item, b_t in zip(risk_items, assignments):
        item["b_t"] = int(b_t)

    summary: Dict[str, Any] = {}

    for b in range(1, K + 1):
        sub = [x for x in risk_items if int(x.get("b_t", 0)) == b]

        summary[str(b)] = {
            "count": len(sub),
            "avg_R_t": _mean([_safe_float(x.get("R_t", 0.0)) for x in sub]),
            "avg_phi_endpoint": _mean([_safe_float(x.get("phi_endpoint", 0.0)) for x in sub]),
            "avg_phi_stay": _mean([_safe_float(x.get("phi_stay", 0.0)) for x in sub]),
            "avg_phi_deg": _mean([_safe_float(x.get("phi_deg", 0.0)) for x in sub]),
            "avg_delta_t": _mean([_safe_float(x.get("delta_t", 0.0)) for x in sub]),
        }

    return summary