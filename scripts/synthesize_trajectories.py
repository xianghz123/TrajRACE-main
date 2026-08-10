#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""Canonical TrajRACE Phase-6 synthesis.

Synthesizes SBS-scale road-segment sequences from recovered start, exact-count,
and conditional successor distributions. Every generated transition is forced
to belong to the fixed PUBLIC successor set N(u). No observed/private bigram is
used as a legality oracle.
"""

import argparse, json, os, random, sys, time
from collections import defaultdict
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

PROJECT_ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
if PROJECT_ROOT not in sys.path: sys.path.insert(0,PROJECT_ROOT)
from trajrace.versioning_utils import apply_versioning, pretty_version_summary
DEFAULT_DATASET_CONFIG="configs/dataset.yaml"; DEFAULT_EXP_CONFIG="configs/exp_main.yaml"

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
def write_jsonl(p:str,items:Sequence[Dict[str,Any]])->None:
    os.makedirs(os.path.dirname(p),exist_ok=True)
    with open(p,"w",encoding="utf-8") as f:
        for x in items:f.write(json.dumps(x,ensure_ascii=False)+"\n")
def count_jsonl(p:str)->int:
    n=0
    with open(p,"r",encoding="utf-8") as f:
        for line in f:
            if line.strip():n+=1
    return n

def normalize(d:Dict[str,float])->Dict[str,float]:
    x={str(k):max(0.0,float(v)) for k,v in d.items()}; s=sum(x.values())
    return {k:v/s for k,v in x.items()} if s>0 else {}
def sample(d:Dict[str,float],rng:random.Random)->Optional[str]:
    d=normalize(d)
    if not d:return None
    r=rng.random(); a=0.0; last=None
    for k,p in d.items():
        a+=p; last=k
        if r<=a:return k
    return last

def load_scalar(p:str)->Tuple[Dict[str,float],int]:
    x=load_json(p);return normalize(x.get("distribution",{})),int(x.get("num_reports",0))
def load_transition(p:str)->Dict[str,Dict[str,Any]]:
    x=load_json(p); out={}
    for u,info in x.items():
        out[str(u)]={"num_reports":int(info.get("num_reports",0)),
                     "domain_size":int(info.get("domain_size",0)),
                     "distribution":normalize(info.get("distribution",{}))}
    return out
def merge_scalar(parts:List[Tuple[Dict[str,float],int]])->Dict[str,float]:
    acc=defaultdict(float); total=0.0
    for d,w in parts:
        w=max(0,int(w))
        if w<=0:continue
        for k,p in normalize(d).items():acc[k]+=w*p
        total+=w
    if total<=0:return {}
    return normalize({k:v/total for k,v in acc.items()})
def merge_transition(parts:List[Dict[str,Dict[str,Any]]])->Dict[str,Dict[str,Any]]:
    acc=defaultdict(lambda:defaultdict(float)); weights=defaultdict(float); ds=defaultdict(int)
    for part in parts:
        for u,info in part.items():
            w=max(0,int(info.get("num_reports",0))); d=normalize(info.get("distribution",{}))
            if w<=0:continue
            for v,p in d.items():acc[u][v]+=w*p
            weights[u]+=w; ds[u]=max(ds[u],int(info.get("domain_size",len(d))))
    out={}
    for u,a in acc.items():
        out[u]={"num_reports":int(weights[u]),"domain_size":ds[u],
                "distribution":normalize({v:s/weights[u] for v,s in a.items()})}
    return out

def global_successor_prior(trans:Dict[str,Dict[str,Any]])->Dict[str,float]:
    a=defaultdict(float); total=0.0
    for info in trans.values():
        w=max(1,int(info.get("num_reports",1))); total+=w
        for v,p in normalize(info.get("distribution",{})).items():a[v]+=w*p
    return normalize({v:s/total for v,s in a.items()}) if total>0 else {}
def legal_candidate_distribution(u:str,trans:Dict[str,Dict[str,Any]],succ:Dict[str,List[str]],global_prior:Dict[str,float],
                                 remaining_after:int)->Tuple[Dict[str,float],str]:
    legal=[str(v) for v in succ.get(u,[])]
    if not legal:return {},"dead_end"
    legal_set=set(legal)
    local=normalize(trans.get(u,{}).get("distribution",{})) if u in trans else {}
    local={v:p for v,p in local.items() if v in legal_set}
    if remaining_after>0:
        continuable={v:p for v,p in local.items() if succ.get(v,[])}
        if continuable:local=continuable
    if local:return normalize(local),"local"
    prior={v:global_prior.get(v,0.0) for v in legal if global_prior.get(v,0.0)>0}
    if remaining_after>0:
        cont={v:p for v,p in prior.items() if succ.get(v,[])}
        if cont:prior=cont
    if prior:return normalize(prior),"public_global_restricted"
    fallback=[v for v in legal if remaining_after<=0 or succ.get(v,[])]
    if not fallback:fallback=legal
    return {v:1.0/len(fallback) for v in fallback},"public_uniform"

def generate_one(syn_id:str,start_dist:Dict[str,float],count_dist:Dict[str,float],trans:Dict[str,Dict[str,Any]],
                 succ:Dict[str,List[str]],global_prior:Dict[str,float],Lmax:int,rng:random.Random)->Tuple[Dict[str,Any],Dict[str,Any]]:
    start=sample(start_dist,rng); count_label=sample(count_dist,rng)
    if start is None:return {"syn_id":syn_id,"segments":[]},{"target_count":0,"actual_count":0,"exact":0,"early_stop":1,"backoff_steps":0}
    try:target=int(count_label) if count_label is not None else 1
    except Exception:target=1
    target=max(1,min(Lmax,target))
    segments=[str(start)]; backoff=0; early=0
    while len(segments)<target:
        u=segments[-1]; rem=target-(len(segments)+1)
        cand,source=legal_candidate_distribution(u,trans,succ,global_prior,rem)
        if not cand:early=1;break
        if source!="local":backoff+=1
        v=sample(cand,rng)
        if v is None or v not in succ.get(u,[]):raise RuntimeError("Internal synthesis legality failure")
        segments.append(v)
    exact=int(len(segments)==target)
    if not exact:early=1
    return {"syn_id":syn_id,"segments":segments,"target_count":target},{"target_count":target,"actual_count":len(segments),"exact":exact,"early_stop":early,"backoff_steps":backoff}

def parse_args():
    p=argparse.ArgumentParser();p.add_argument("--dataset_config",default=DEFAULT_DATASET_CONFIG);p.add_argument("--exp_config",default=DEFAULT_EXP_CONFIG)
    p.add_argument("--progress_every",type=int,default=250);return p.parse_args()
def main():
    a=parse_args();dp,ep=resolve_path(a.dataset_config),resolve_path(a.exp_config);assert dp and ep
    rd,re=load_yaml(dp),load_yaml(ep);d,e=apply_versioning(rd,re)
    recdir=resolve_path(e["recovered_dir"]);syndir=resolve_path(e["synthetic_dir"]);succp=resolve_path(d["successor_cache_path"]); assert recdir and syndir and succp
    os.makedirs(syndir,exist_ok=True);succ=load_json(succp);methods=[str(x) for x in e.get("methods",["riskaware","uniform"])]
    splits=[str(x) for x in e.get("synthesis_splits",["train","valid"])]
    for s in splits:
        if s not in {"train","valid"}:raise ValueError("synthesis_splits may contain only train/valid; test leakage is forbidden")
    mode=str(e.get("synthetic_num_mode","match_test_size"));test_event=resolve_path(d["event_test"]);assert test_event
    if mode=="match_test_size": n_syn=count_jsonl(test_event)
    elif mode=="fixed": n_syn=int(e.get("synthetic_num",1000))
    else:
        try:n_syn=int(mode)
        except Exception:raise ValueError(f"Unsupported synthetic_num_mode={mode}")
    Lmax=int(e["L_max"]);base_seed=int(e.get("random_seed",42));recovery_mode=str(e.get("recovery_mode","context_bucket_mixture"))
    print("="*90);print("[synthesize_trajectories]");print(json.dumps(pretty_version_summary(rd,re),indent=2,ensure_ascii=False));print(f"methods={methods}, synthesis_splits={splits}, synthetic_num={n_syn}");print("="*90)
    overview={"dataset_name":d["dataset_name"],"dataset_variant":d["dataset_variant"],"exp_tag":e["exp_tag"],"synthetic_num":n_syn,"methods":{}}
    for mi,method in enumerate(methods):
        starts=[];counts=[];transparts=[]
        component="transition_joint" if (method=="riskaware" and recovery_mode=="joint_latent_recovery") else "transition_context"
        for split in splits:
            sp=os.path.join(recdir,f"{method}_{split}_start.json");cp=os.path.join(recdir,f"{method}_{split}_count.json");tp=os.path.join(recdir,f"{method}_{split}_{component}.json")
            for p in (sp,cp,tp):
                if not os.path.exists(p):raise FileNotFoundError(f"Missing recovered input: {p}")
            starts.append(load_scalar(sp));counts.append(load_scalar(cp));transparts.append(load_transition(tp))
        sd=merge_scalar(starts);cd=merge_scalar(counts);tr=merge_transition(transparts);gp=global_successor_prior(tr)
        # Never sample an invalid public start.
        sd=normalize({s:p for s,p in sd.items() if s in succ})
        if not sd:raise RuntimeError(f"Recovered start distribution empty for {method}")
        rng=random.Random(base_seed+7001+mi*100003);records=[];diags=[]
        t0=time.perf_counter()
        for i in range(n_syn):
            rec,diag=generate_one(f"{method}_syn_{i:06d}",sd,cd,tr,succ,gp,Lmax,rng);records.append(rec);diags.append(diag)
            if a.progress_every>0 and (i+1)%a.progress_every==0:print(f"[synthesis][{method}] {i+1:,}/{n_syn:,}")
        path=os.path.join(syndir,f"{method}_test_synthetic.jsonl");write_jsonl(path,records)
        exact=sum(x["exact"] for x in diags)/len(diags) if diags else 0.0;early=sum(x["early_stop"] for x in diags)/len(diags) if diags else 0.0
        avg_t=sum(x["target_count"] for x in diags)/len(diags) if diags else 0.0;avg_a=sum(x["actual_count"] for x in diags)/len(diags) if diags else 0.0
        avg_b=sum(x["backoff_steps"] for x in diags)/len(diags) if diags else 0.0
        sm={"method":method,"synthetic_output_path":path,"num_synthetic_trajectories":len(records),"exact_count_rate":exact,"early_stop_rate":early,"avg_target_count":avg_t,"avg_actual_count":avg_a,"avg_public_backoff_steps":avg_b,"synthesis_splits":splits,"transition_recovery_component":component,"elapsed_sec":time.perf_counter()-t0,"public_successor_hard_constraint":True}
        save_json(os.path.join(syndir,f"{method}_test_synthetic_summary.json"),sm);overview["methods"][method]=sm
        print(f"[synthesis][{method}] DONE exact={exact:.6f}, early_stop={early:.6f}")
    save_json(os.path.join(syndir,"synthetic_overview.json"),overview);print("[synthesize_trajectories] DONE")
if __name__=="__main__":main()
