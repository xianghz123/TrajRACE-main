#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""Server-only recovery for the canonical uniform comparator.

This file is intentionally separate from the already-validated risk-aware
recover_statistics.py so the Phase-5 implementation does not need to be
changed again during rebuttal. It consumes ONLY uniform privatized reports
and PUBLIC domains.
"""

import argparse, json, math, os, sys, time
from collections import Counter, defaultdict
from typing import Any, Dict, Iterator, Optional, Sequence

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from trajrace.versioning_utils import apply_versioning, pretty_version_summary

DEFAULT_DATASET_CONFIG = "configs/dataset.yaml"
DEFAULT_EXP_CONFIG = "configs/exp_main.yaml"


def resolve_path(p: Optional[str]) -> Optional[str]:
    if p is None: return None
    return p if os.path.isabs(p) else os.path.abspath(os.path.join(PROJECT_ROOT, p))

def load_yaml(p: str) -> Dict[str, Any]:
    import yaml
    with open(p, "r", encoding="utf-8") as f: x = yaml.safe_load(f)
    if not isinstance(x, dict): raise ValueError("YAML root must be a mapping")
    return x

def load_json(p: str) -> Any:
    with open(p, "r", encoding="utf-8") as f: return json.load(f)

def save_json(p: str, x: Any) -> None:
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f: json.dump(x, f, ensure_ascii=False, indent=2)

def iter_jsonl(p: str) -> Iterator[Dict[str, Any]]:
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line: yield json.loads(line)

def rr_params(eps: float, d: int):
    if d <= 0: raise ValueError("RR domain must be non-empty")
    if d == 1: return 1.0, 0.0
    ee = math.exp(float(eps)); return ee/(ee+d-1), 1.0/(ee+d-1)

def clipped_rr_inverse(counter: Counter, domain: Sequence[Any], eps: float) -> Dict[str, float]:
    domain=list(domain); n=sum(counter.values()); d=len(domain)
    if n <= 0: return {}
    p,q=rr_params(eps,d); den=p-q
    est={}
    for x in domain:
        raw=(counter.get(x,0)/n-q)/den if abs(den)>1e-15 else counter.get(x,0)/n
        if raw>0: est[str(x)]=float(raw)
    s=sum(est.values())
    if s<=0:
        est={str(x): counter.get(x,0)/n for x in domain if counter.get(x,0)>0}; s=sum(est.values())
    return {k:v/s for k,v in est.items()} if s>0 else {}

def sparse_start_inverse(counter: Counter, public_domain_size: int, eps: float) -> Dict[str,float]:
    n=sum(counter.values()); p,q=rr_params(eps,public_domain_size); den=p-q
    if n<=0: return {}
    est={}
    for x,c in counter.items():
        raw=(c/n-q)/den if abs(den)>1e-15 else c/n
        if raw>0: est[str(x)]=float(raw)
    s=sum(est.values())
    if s<=0: est={str(x): c/n for x,c in counter.items()}; s=sum(est.values())
    return {k:v/s for k,v in est.items()} if s>0 else {}

def derive_uniform_eps(exp: Dict[str,Any]) -> float:
    if "eps_event_uniform" in exp: return float(exp["eps_event_uniform"])
    B=float(exp["B_total"]); es=float(exp["eps_start"]); ec=float(exp["eps_count"])
    eb=float(exp["eps_bucket"]); L=int(exp["L_max"])
    return (B-es-ec)/(L-1)-eb

def parse_args():
    p=argparse.ArgumentParser(); p.add_argument("--dataset_config",default=DEFAULT_DATASET_CONFIG)
    p.add_argument("--exp_config",default=DEFAULT_EXP_CONFIG); return p.parse_args()

def main():
    a=parse_args(); dp,ep=resolve_path(a.dataset_config),resolve_path(a.exp_config); assert dp and ep
    rd,re=load_yaml(dp),load_yaml(ep); d,e=apply_versioning(rd,re)
    succ_path=resolve_path(d["successor_cache_path"]); priv=resolve_path(e["privatized_dir"]); rec=resolve_path(e["recovered_dir"])
    assert succ_path and priv and rec; os.makedirs(rec,exist_ok=True)
    succ=load_json(succ_path); eps_u=derive_uniform_eps(e); L=int(e["L_max"])
    print("="*90); print("[recover_uniform_statistics]")
    print(json.dumps(pretty_version_summary(rd,re),indent=2,ensure_ascii=False))
    print(f"eps_event_uniform={eps_u:.12f}"); print("="*90)
    all_summary={}
    for split in ("train","valid","test"):
        path=os.path.join(priv,f"uniform_{split}_reports.jsonl")
        if not os.path.exists(path): raise FileNotFoundError(path)
        starts=Counter(); counts=Counter(); per_u=defaultdict(Counter); types=Counter()
        for r in iter_jsonl(path):
            typ=r["event_type"]; types[typ]+=1
            if typ=="start": starts[str(r["x_noisy"])]+=1
            elif typ=="count": counts[int(r["x_noisy"])]+=1
            elif typ=="transition":
                u,y=str(r["u"]),str(r["y"])
                if u not in succ or y not in succ[u]: raise RuntimeError(f"Report outside public N(u): {u}->{y}")
                per_u[u][y]+=1
            else: raise RuntimeError(f"Unknown event_type={typ}")
        start_dist=sparse_start_inverse(starts,len(succ),float(e["eps_start"]))
        count_dist=clipped_rr_inverse(counts,range(1,L+1),float(e["eps_count"]))
        trans={}
        for u,c in per_u.items():
            domain=succ.get(u,[])
            if not domain: raise RuntimeError(f"Empty public N(u) for reported context {u}")
            dist=clipped_rr_inverse(c,domain,eps_u)
            trans[u]={"domain_size":len(domain),"num_reports":int(sum(c.values())),"distribution":dist}
        start_obj={"domain_size":len(succ),"num_reports":int(sum(starts.values())),"distribution":start_dist}
        count_obj={"domain_size":L,"num_reports":int(sum(counts.values())),"distribution":count_dist}
        save_json(os.path.join(rec,f"uniform_{split}_start.json"),start_obj)
        save_json(os.path.join(rec,f"uniform_{split}_count.json"),count_obj)
        save_json(os.path.join(rec,f"uniform_{split}_transition_context.json"),trans)
        all_summary[split]={"report_counts":dict(types),"num_transition_contexts":len(trans),"eps_event_uniform":eps_u}
        print(f"[uniform recovery][{split}] contexts={len(trans):,}, transitions={types['transition']:,}")
    out={"dataset_name":d["dataset_name"],"dataset_variant":d["dataset_variant"],"exp_tag":e["exp_tag"],
         "method":"uniform","eps_event_uniform":eps_u,"splits":all_summary,
         "server_only_inputs":True,"public_domains_only":True}
    save_json(os.path.join(rec,"uniform_recovery_summary.json"),out)
    print("[recover_uniform_statistics] DONE")
if __name__=="__main__": main()
