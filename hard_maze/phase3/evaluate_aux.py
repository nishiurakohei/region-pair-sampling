"""Aggregate Phase 3-A pair-connectivity learnability and distance diagnostics."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

P3=Path(__file__).resolve().parent; ROOT=P3.parent
OUT=ROOT/"results_phase3"/"aux_only"; FIG=OUT/"figures"


def auroc(y,p):
    order=np.argsort(p,kind="mergesort"); ranks=np.empty(len(p),float); sp=p[order]; i=0
    while i<len(p):
        j=i+1
        while j<len(p) and sp[j]==sp[i]: j+=1
        ranks[order[i:j]]=(i+1+j)/2; i=j
    n1=int(y.sum()); n0=len(y)-n1
    return float((ranks[y==1].sum()-n1*(n1+1)/2)/(n1*n0))


def score(y,p):
    p=np.clip(p.astype(float),1e-7,1-1e-7); pred=p>.5
    tpr=np.mean(pred[y==1]); tnr=np.mean(~pred[y==0])
    return {"accuracy":float(np.mean(pred==y)),"balanced_accuracy":float((tpr+tnr)/2),
            "connected_recall":float(tpr),"disconnected_recall":float(tnr),
            "auroc":auroc(y,p),"bce":float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p))),
            "brier":float(np.mean((p-y)**2))}


def aggregate(rows):
    return {k:{"mean":float(np.mean([r[k] for r in rows])),"std":float(np.std([r[k] for r in rows]))}
            for k in rows[0]}


def binned(y,p,d,edges):
    out=[]
    for lo,hi in zip(edges[:-1],edges[1:]):
        mask=(d>=lo)&(d<hi)
        out.append({"range":f"{lo}-{hi-1 if hi<999 else '+'}","count":int(mask.sum()),
                    "accuracy":float(np.mean((p[mask]>.5)==y[mask]))})
    return out


def main():
    cfg=json.loads((P3/"config.json").read_text()); k=cfg["primary_k"]
    fixed=np.load(ROOT/"results_phase3"/"pairs"/f"fixed_test_k{max(cfg['k_values'])}.npz",allow_pickle=False)
    grid=fixed["grid_distance"][:,:k].ravel(); graph=fixed["graph_distance"][:,:k].ravel()
    out={"config":cfg,"models":{} }; rows=[]; curves={}
    for model in cfg["models"]:
        seeds=[]; curves[model]={}
        for seed in cfg["aux_only_seeds"]:
            z=np.load(OUT/f"pred_{model}_s{seed}.npz",allow_pickle=False)
            y,p=z["pair_targets"].ravel().astype(int),z["pair_prob"].ravel()
            m=score(y,p); seeds.append(m); rows.append({"model":model,"seed":seed,**m})
            curves[model][seed]={"grid":binned(y,p,grid,[5,9,14,999]),
                "graph_connected":binned(y[y==1],p[y==1],graph[y==1],[5,11,21,999])}
        out["models"][model]={"per_seed":seeds,"aggregate":aggregate(seeds),"distance":curves[model]}
    out["constant_baseline"]={"accuracy":.5,"balanced_accuracy":.5,"auroc":.5,"bce":float(np.log(2)),"brier":.25}
    (OUT/"metrics.json").write_text(json.dumps(out,indent=2))
    with (OUT/"per_seed_metrics.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
    FIG.mkdir(exist_ok=True)
    labels=["5–8","9–13","14+"]
    fig,axes=plt.subplots(1,2,figsize=(10,4.5))
    for ax,model in zip(axes,cfg["models"]):
        vals=np.array([[x["accuracy"] for x in curves[model][s]["grid"]] for s in cfg["aux_only_seeds"]])
        ax.errorbar(labels,vals.mean(0),yerr=vals.std(0),fmt="o-",capsize=3)
        ax.axhline(.5,color="gray",ls="--"); ax.set_ylim(.45,1); ax.set_title(model.upper())
        ax.set_xlabel("grid Manhattan distance"); ax.set_ylabel("auxiliary accuracy"); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(FIG/"fig1_accuracy_vs_grid_distance.png",dpi=150); plt.close(fig)
    labels2=["5–10","11–20","21+"]
    fig,axes=plt.subplots(1,2,figsize=(10,4.5))
    for ax,model in zip(axes,cfg["models"]):
        vals=np.array([[x["accuracy"] for x in curves[model][s]["graph_connected"]] for s in cfg["aux_only_seeds"]])
        ax.errorbar(labels2,vals.mean(0),yerr=vals.std(0),fmt="o-",capsize=3)
        ax.axhline(.5,color="gray",ls="--"); ax.set_ylim(.45,1); ax.set_title(model.upper())
        ax.set_xlabel("connected graph distance"); ax.set_ylabel("auxiliary accuracy"); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(FIG/"fig2_accuracy_vs_graph_distance.png",dpi=150); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4.5)); target=fixed["targets"][:,:k].ravel()
    ax.hist(grid[target==1],bins=np.arange(5,30),histtype="step",density=True,lw=2,label="connected")
    ax.hist(grid[target==0],bins=np.arange(5,30),histtype="step",density=True,lw=2,label="disconnected")
    ax.set_xlabel("grid Manhattan distance"); ax.set_ylabel("density"); ax.legend(); fig.tight_layout()
    fig.savefig(FIG/"fig3_pair_distance_distribution.png",dpi=150); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4.5))
    for ax,model in zip(axes,cfg["models"]):
        for seed in cfg["aux_only_seeds"]:
            z=np.load(OUT/f"pred_{model}_s{seed}.npz",allow_pickle=False); h=json.loads(str(z["history"]))
            ax.plot([x["epoch"] for x in h],[x["val_pair_bce"] for x in h],label=f"seed {seed}")
        ax.axhline(np.log(2),color="gray",ls="--"); ax.set_title(model.upper()); ax.set_xlabel("epoch")
        ax.set_ylabel("validation pair BCE"); ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(FIG/"fig4_training_curves.png",dpi=150); plt.close(fig)
    best=max(cfg["models"],key=lambda m:out["models"][m]["aggregate"]["accuracy"]["mean"])
    best_acc=out["models"][best]["aggregate"]["accuracy"]["mean"]; proceed=best_acc>.65
    lines=[]
    for model in cfg["models"]:
        a=out["models"][model]["aggregate"]
        lines.append(f"| {model.upper()} | {a['accuracy']['mean']:.3f} ± {a['accuracy']['std']:.3f} | {a['auroc']['mean']:.3f} | {a['bce']['mean']:.3f} | {a['brier']['mean']:.3f} |")
    report=f"""# Phase 3-A — Random-pair Connectivity Auxiliary-only

## Decision

Best auxiliary accuracy: **{best.upper()} {best_acc:.3f}**. Decision: **{'PROCEED to Phase 3-B' if proceed else 'PAUSE full Phase 3-B sweep'}**.

| model | accuracy | AUROC | BCE | Brier |
|---|---:|---:|---:|---:|
{chr(10).join(lines)}

Pairs exclude start/goal and identical nodes, use grid distance ≥{cfg['d_min']}, and are exactly balanced on fixed validation/test sets. Figure 1 and Figure 2 show distance-stratified performance. The threshold for proceeding was auxiliary accuracy >0.65.
"""
    (OUT/"aux_only_report.md").write_text(report)
    (OUT/"decision.json").write_text(json.dumps({"proceed":proceed,"best_model":best,"best_accuracy":best_acc},indent=2))
    print(json.dumps({"proceed":proceed,"best_model":best,"best_accuracy":best_acc},indent=2))


if __name__=="__main__": main()
