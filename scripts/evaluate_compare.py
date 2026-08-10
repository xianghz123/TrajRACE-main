#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""Canonical TrajRACE smoke comparison.

This is a compact artifact/equivalence evaluation, not the final full paper
RQ suite. It compares risk-aware and uniform mechanisms on:
  * overall/high-risk successor keep rate;
  * weighted JS of recovered P(V|U);
  * synthetic bigram JS;
  * synthetic exact-count distribution JS;
  * PUBLIC road-topology legality ratio;
  * exact target-count realization rate.
"""

import argparse,csv,json,math,os,sys
from collections import Counter,defaultdict
from typing import Any,Dict,Iterator,List,Optional,Sequence,Tuple
PROJECT_ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
if PROJECT_ROOT not in sys.path:sys.path.insert(0,PROJECT_ROOT)
from trajrace.versioning_utils import apply_versioning,pretty_version_summary
DEFAULT_DATASET_CONFIG="configs/dataset.yaml";DEFAULT_EXP_CONFIG="configs/exp_main.yaml"

def resolve_path(p:Optional[str])->Optional[str]:
    if p is None:return None
    return p if os.path.isabs(p) else os.path.abspath(os.path.join(PROJECT_ROOT,p))
def load_yaml(p:str)->Dict[str,Any]:
    import yaml
    with open(p,"r",encoding="utf-8") as f:x=yaml.safe_load(f)
    if not isinstance(x,dict):raise ValueError("YAML root must be mapping")
    return x
def load_json(p:str)->Any:
    with open(p,"r",encoding="utf-8") as f:return json.load(f)
def save_json(p:str,x:Any)->None:
    os.makedirs(os.path.dirname(p),exist_ok=True)
    with open(p,"w",encoding="utf-8") as f:json.dump(x,f,ensure_ascii=False,indent=2)
def iter_jsonl(p:str)->Iterator[Dict[str,Any]]:
    with open(p,"r",encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line:yield json.loads(line)
def save_csv(p:str,rows:List[Dict[str,Any]],fields:List[str])->None:
    os.makedirs(os.path.dirname(p),exist_ok=True)
    with open(p,"w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def normalize(d:Dict[Any,float])->Dict[Any,float]:
    x={k:max(0.0,float(v)) for k,v in d.items()};s=sum(x.values());return {k:v/s for k,v in x.items()} if s>0 else {}
def js(p:Dict[Any,float],q:Dict[Any,float])->float:
    sup=set(p)|set(q)
    if not sup:return 0.0
    p=normalize({k:p.get(k,0.0) for k in sup});q=normalize({k:q.get(k,0.0) for k in sup});m={k:.5*(p.get(k,0)+q.get(k,0)) for k in sup}
    def kl(a,b):return sum(av*math.log(av/b[k]) for k,av in a.items() if av>0 and b.get(k,0)>0)
    return .5*kl(p,m)+.5*kl(q,m)
def true_transition(events:Sequence[Dict[str,Any]]):
    c=defaultdict(Counter);n=Counter()
    for r in events:
        for t in r["transition_events"]:u,v=str(t["u"]),str(t["v"]);c[u][v]+=1;n[u]+=1
    return {u:{v:k/sum(cc.values()) for v,k in cc.items()} for u,cc in c.items()},dict(n)
def recovered_transition(p:str)->Dict[str,Dict[str,float]]:
    x=load_json(p);return {str(u):{str(v):float(z) for v,z in info.get("distribution",{}).items()} for u,info in x.items()}
def weighted_transition_js(t:Dict[str,Dict[str,float]],n:Dict[str,int],h:Dict[str,Dict[str,float]])->float:
    total=sum(n.values());return sum(n[u]*js(t[u],h.get(u,{})) for u in t)/total if total else 0.0
def bigram_prob(records:Sequence[Dict[str,Any]])->Dict[Tuple[str,str],float]:
    c=Counter()
    for r in records:
        s=[str(x) for x in r.get("segments",[])]
        for a,b in zip(s[:-1],s[1:]):c[(a,b)]+=1
    z=sum(c.values());return {k:v/z for k,v in c.items()} if z else {}
def count_prob(records:Sequence[Dict[str,Any]])->Dict[int,float]:
    c=Counter(len(r.get("segments",[])) for r in records);z=sum(c.values());return {k:v/z for k,v in c.items()} if z else {}
def public_legal_ratio(records:Sequence[Dict[str,Any]],succ:Dict[str,List[str]])->float:
    good=total=0
    for r in records:
        s=[str(x) for x in r.get("segments",[])]
        for a,b in zip(s[:-1],s[1:]):total+=1;good+=int(b in succ.get(a,[]))
    return good/total if total else 1.0
def parse_args():
    p=argparse.ArgumentParser();p.add_argument("--dataset_config",default=DEFAULT_DATASET_CONFIG);p.add_argument("--exp_config",default=DEFAULT_EXP_CONFIG);return p.parse_args()
def main():
    a=parse_args();dp,ep=resolve_path(a.dataset_config),resolve_path(a.exp_config);assert dp and ep
    rd,re=load_yaml(dp),load_yaml(ep);d,e=apply_versioning(rd,re)
    exp_root=resolve_path(e["experiment_root"]);priv=resolve_path(e["privatized_dir"]);recdir=resolve_path(e["recovered_dir"]);syndir=resolve_path(e["synthetic_dir"]);succp=resolve_path(d["successor_cache_path"]);testp=resolve_path(d["event_test"])
    assert exp_root and priv and recdir and syndir and succp and testp
    outdir=os.path.join(exp_root,"evaluation");os.makedirs(outdir,exist_ok=True);succ=load_json(succp);test=list(iter_jsonl(testp));t_dist,t_n=true_transition(test);t_big=bigram_prob(test);t_count=count_prob(test)
    methods=[str(x) for x in e.get("methods",["riskaware","uniform"])];recovery_mode=str(e.get("recovery_mode","context_bucket_mixture"));rows=[]
    print("="*90);print("[evaluate_compare]");print(json.dumps(pretty_version_summary(rd,re),indent=2,ensure_ascii=False));print("="*90)
    for m in methods:
        summ_path=os.path.join(priv,f"{m}_summary.json")
        if not os.path.exists(summ_path):raise FileNotFoundError(summ_path)
        sm=load_json(summ_path);test_sm=sm["splits"]["test"]
        keep_by=test_sm.get("keep_rate_by_target_bucket",{});high=float(keep_by.get("1",0.0));overall=float(test_sm.get("keep_rate_overall",0.0))
        component="transition_joint" if (m=="riskaware" and recovery_mode=="joint_latent_recovery") else "transition_context"
        rp=os.path.join(recdir,f"{m}_test_{component}.json");sp=os.path.join(syndir,f"{m}_test_synthetic.jsonl");ssp=os.path.join(syndir,f"{m}_test_synthetic_summary.json")
        for p in (rp,sp,ssp):
            if not os.path.exists(p):raise FileNotFoundError(p)
        rh=recovered_transition(rp);syn=list(iter_jsonl(sp));syn_sm=load_json(ssp)
        row={"method":m,"high_risk_keep_rate":high,"keep_rate_overall":overall,
             "transition_js":weighted_transition_js(t_dist,t_n,rh),"synthetic_bigram_js":js(t_big,bigram_prob(syn)),
             "synthetic_count_js":js(t_count,count_prob(syn)),"public_legal_ratio":public_legal_ratio(syn,succ),
             "exact_count_rate":float(syn_sm.get("exact_count_rate",0.0)),"early_stop_rate":float(syn_sm.get("early_stop_rate",0.0)),
             "num_synthetic_trajectories":len(syn),"transition_recovery_component":component}
        rows.append(row);print(f"[{m}] high-risk-keep={high:.6f}, trans-JS={row['transition_js']:.6f}, bigram-JS={row['synthetic_bigram_js']:.6f}, legal={row['public_legal_ratio']:.6f}, exact={row['exact_count_rate']:.6f}")
    fields=list(rows[0].keys()) if rows else []
    save_csv(os.path.join(outdir,"main_compare.csv"),rows,fields)
    report={"dataset_name":d["dataset_name"],"dataset_variant":d["dataset_variant"],"exp_tag":e["exp_tag"],"results":rows,
            "notes":{"high_risk_keep_rate":"Lower is stronger protection for target bucket 1.","transition_js":"Lower is better.","synthetic_bigram_js":"Lower is better.","synthetic_count_js":"Lower is better.","public_legal_ratio":"Must be 1.0 under the public-topology hard constraint.","exact_count_rate":"Higher is better."}}
    save_json(os.path.join(outdir,"main_compare.json"),report)
    # Hard artifact gates, not superiority claims.
    for r in rows:
        if r["public_legal_ratio"] < 1.0-1e-12:raise RuntimeError(f"Synthetic public-topology legality failed for {r['method']}")
        if r["num_synthetic_trajectories"] != len(test):raise RuntimeError(f"Synthetic/test size mismatch for {r['method']}")
    print(f"[evaluate_compare] DONE -> {outdir}")
if __name__=="__main__":main()
