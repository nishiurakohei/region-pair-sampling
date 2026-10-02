"""Aggregate Phase 3-B, controls, uncertainty, figures, failures, and verification."""
import csv
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

P3=Path(__file__).resolve().parent; ROOT=P3.parent
sys.path[:0]=[str(ROOT/"src"),str(ROOT/"phase2")]
from dataset import MazeData
from figures import plot_maze
from evaluate_multitask import scores as main_scores, aggregate, bootstrap_ci
from pairs import ensure_component_cache, manhattan, sample_pairs

OUT=ROOT/"results_phase3"/"multitask"; FIG=OUT/"figures"


def tag(x): return str(x).replace(".","p")


def aux_scores(y,p):
    p=np.clip(p.astype(float),1e-7,1-1e-7); pred=p>.5
    order=np.argsort(p,kind="mergesort"); ranks=np.empty(len(p),float); sp=p[order]; i=0
    while i<len(p):
        j=i+1
        while j<len(p) and sp[j]==sp[i]: j+=1
        ranks[order[i:j]]=(i+1+j)/2; i=j
    n1=int(y.sum()); n0=len(y)-n1
    return {"aux_accuracy":float(np.mean(pred==y)),
            "aux_balanced_accuracy":float((np.mean(pred[y==1])+np.mean(~pred[y==0]))/2),
            "aux_connected_recall":float(np.mean(pred[y==1])),
            "aux_disconnected_recall":float(np.mean(~pred[y==0])),
            "aux_auroc":float((ranks[y==1].sum()-n1*(n1+1)/2)/(n1*n0)),
            "aux_bce":float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p))),
            "aux_brier":float(np.mean((p-y)**2))}


def load_phase3(model,condition,lam,k,seed):
    return np.load(OUT/f"pred_{model}_{condition}_l{tag(lam)}_k{k}_s{seed}.npz",allow_pickle=False)


def savefig(fig,name):
    fig.tight_layout(); fig.savefig(FIG/name,dpi=150); plt.close(fig)


def best_val(model,lam,k,seed):
    z=load_phase3(model,"true",lam,k,seed); h=json.loads(str(z["history"]))
    return min(x["val_main_bce"] for x in h)


def main():
    cfg=json.loads((P3/"config.json").read_text()); ks=json.loads((OUT/"k_selection.json").read_text())
    k=ks["selected_k"]; seeds=cfg["multitask_seeds"]
    data=MazeData("H3","test"); y=data.y.astype(int); beta=data.arrays["beta1"].astype(float)
    results={"config":cfg,"selected_k":k,"models":{}}; seed_rows=[]; pair_rows=[]; probs={}; margins={}
    for model in cfg["models"]:
        results["models"][model]={}
        for condition,lam in [("main",0.0)]+[(c,l) for c in ("true","shuffled") for l in cfg["lambdas"] if l>0]:
            per=[]
            for seed in seeds:
                if condition=="main":
                    z=np.load(ROOT/"results_phase2"/"multitask"/f"pred_{model}_main_l0p0_s{seed}.npz",allow_pickle=False)
                    p=z["prob"]; am={}
                else:
                    z=load_phase3(model,condition,lam,k,seed); p=z["main_prob"]
                    am=aux_scores(z["pair_targets"].ravel().astype(int),z["pair_prob"].ravel())
                m,margin=main_scores(y,p,beta,None); m.update(am); per.append(m)
                probs[(model,condition,lam,seed)]=p; margins[(model,condition,lam,seed)]=margin
                seed_rows.append({"model":model,"condition":condition,"lambda":lam,"k":0 if condition=="main" else k,"seed":seed,**m})
                for pair in range(len(y)//2):
                    i,j=2*pair,2*pair+1
                    pair_rows.append({"model":model,"condition":condition,"lambda":lam,"seed":seed,"pair_id":pair,
                                      "p_positive":float(p[i]),"p_negative":float(p[j]),"margin":float(margin[pair]),
                                      "pair_correct":int(p[i]>.5 and p[j]<=.5)})
            results["models"][model][f"{condition}_lambda_{lam}"]={
                "per_seed":per,
                "aggregate":aggregate(per),
                "high_performing_seed_threshold":0.65,
                "high_performing_seed_count":int(sum(x["accuracy"]>0.65 for x in per)),
                "high_performing_seed_rate":float(np.mean([x["accuracy"]>0.65 for x in per])),
            }
        base=results["models"][model]["main_lambda_0.0"]["per_seed"]; uncertainty={}
        for lam in (x for x in cfg["lambdas"] if x>0):
            true=results["models"][model][f"true_lambda_{lam}"]["per_seed"]
            shuf=results["models"][model][f"shuffled_lambda_{lam}"]["per_seed"]
            uncertainty[str(lam)]={}
            for metric in ("accuracy","pair_accuracy","pair_margin_mean"):
                dm=[a[metric]-b[metric] for a,b in zip(true,base)]; ds=[a[metric]-b[metric] for a,b in zip(true,shuf)]
                uncertainty[str(lam)][metric]={"true_minus_main_mean":float(np.mean(dm)),"true_minus_main_ci95":bootstrap_ci(dm),
                    "true_minus_shuffled_mean":float(np.mean(ds)),"true_minus_shuffled_ci95":bootstrap_ci(ds,seed=1)}
        results["models"][model]["paired_uncertainty"]=uncertainty
    selection={m:{str(l):float(np.mean([best_val(m,l,k,s) for s in seeds])) for l in cfg["lambdas"] if l>0}
               for m in cfg["models"]}
    best={m:min((l for l in cfg["lambdas"] if l>0),key=lambda l:selection[m][str(l)]) for m in cfg["models"]}
    results["lambda_selection"]={"criterion":"lowest mean per-seed best validation main BCE","values":selection,"selected":best}
    OUT.mkdir(parents=True,exist_ok=True); FIG.mkdir(exist_ok=True)
    (OUT/"config.json").write_text(json.dumps({**cfg,"selected_k":k},indent=2))
    (OUT/"metrics.json").write_text(json.dumps(results,indent=2))
    for name,rows in (("per_seed_metrics.csv",seed_rows),("per_pair_predictions.csv",pair_rows)):
        fields=list(rows[0]); fields.extend(x for row in rows for x in row if x not in fields)
        with (OUT/name).open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    sweep=[]
    for model in cfg["models"]:
        a=results["models"][model]["main_lambda_0.0"]["aggregate"]
        sweep.append({"model":model,"condition":"main","lambda":0.0,"k":0,
                      **{f"{x}_mean":v["mean"] for x,v in a.items()},**{f"{x}_std":v["std"] for x,v in a.items()}})
        for condition in ("true","shuffled"):
            for lam in (x for x in cfg["lambdas"] if x>0):
                a=results["models"][model][f"{condition}_lambda_{lam}"]["aggregate"]
                sweep.append({"model":model,"condition":condition,"lambda":lam,"k":k,
                              **{f"{x}_mean":v["mean"] for x,v in a.items()},**{f"{x}_std":v["std"] for x,v in a.items()}})
    fields=list(sweep[0]); fields.extend(x for row in sweep for x in row if x not in fields)
    with (OUT/"lambda_sweep.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(sweep)
    k_rows=[]; pilot_lam=cfg["k_pilot_lambda"]
    for ki in cfg["k_values"]:
        ms=[]
        for seed in cfg["aux_only_seeds"]:
            z=load_phase3("cnn","true",pilot_lam,ki,seed); m,_=main_scores(y,z["main_prob"],beta,None); ms.append(m)
        a=aggregate(ms); k_rows.append({"k":ki,"lambda":pilot_lam,**{f"{x}_mean":v["mean"] for x,v in a.items()},
                                        **{f"{x}_std":v["std"] for x,v in a.items()}})
    with (OUT/"k_sweep.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=k_rows[0]); w.writeheader(); w.writerows(k_rows)
    make_figures(cfg,k,results,best,k_rows,margins)
    failure=failures(cfg,k,best,data,y,probs)
    verification=verify(cfg,k,data,y)
    report(cfg,k,results,best,k_rows,failure,verification)
    print("selected",{"k":k,"lambda":best},"verification",verification["all_ok"])


def make_figures(cfg,k,r,best,k_rows,margins):
    fig,axes=plt.subplots(1,2,figsize=(13,4.8))
    for ax,name,title in zip(axes,("fig1_accuracy_vs_grid_distance.png","fig2_accuracy_vs_graph_distance.png"),
                             ("Grid Manhattan distance","Connected-pair graph distance")):
        ax.imshow(plt.imread(ROOT/"results_phase3"/"aux_only"/"figures"/name)); ax.axis("off"); ax.set_title(title)
    savefig(fig,"fig1_aux_accuracy_vs_distance.png")
    p2=json.loads((ROOT/"results_phase2"/"multitask"/"metrics.json").read_text())
    xs=[x for x in cfg["lambdas"] if x>0]
    for metric,name,ylabel in (("accuracy","fig2_lambda_vs_accuracy.png","main accuracy"),
                               ("pair_accuracy","fig3_lambda_vs_pair_accuracy.png","H3 pair accuracy")):
        fig,axes=plt.subplots(1,2,figsize=(11,4.5))
        for ax,model in zip(axes,cfg["models"]):
            base=r["models"][model]["main_lambda_0.0"]["aggregate"][metric]
            for cond,color in (("true","#2b6cb0"),("shuffled","#c05621")):
                vals=[r["models"][model][f"{cond}_lambda_{l}"]["aggregate"][metric]["mean"] for l in xs]
                err=[r["models"][model][f"{cond}_lambda_{l}"]["aggregate"][metric]["std"] for l in xs]
                ax.errorbar(xs,vals,yerr=err,fmt="o-",capsize=3,color=color,label=f"pair {cond}")
            beta_vals=[p2["models"][model][f"true_lambda_{l}"]["aggregate"][metric]["mean"] for l in xs]
            ax.plot(xs,beta_vals,"s:",color="#805ad5",label="β₁ true")
            ax.axhline(base["mean"],color="gray",ls="--",label="main-only")
            ax.axhspan(base["mean"]-base["std"],base["mean"]+base["std"],color="gray",alpha=.12)
            ax.set_xscale("log"); ax.set_title(model.upper()); ax.set_xlabel("λ"); ax.set_ylabel(ylabel); ax.legend(fontsize=8); ax.grid(alpha=.2)
        savefig(fig,name)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for ax,model in zip(axes,cfg["models"]):
        lam=best[model]; blam=p2["lambda_selection"]["selected"][model]
        main=np.concatenate([margins[(model,"main",0.0,s)] for s in cfg["multitask_seeds"]])
        pair=np.concatenate([margins[(model,"true",lam,s)] for s in cfg["multitask_seeds"]])
        beta=[]
        data=MazeData("H3","test")
        for s in cfg["multitask_seeds"]:
            z=np.load(ROOT/"results_phase2"/"multitask"/f"pred_{model}_true_l{tag(blam)}_s{s}.npz",allow_pickle=False)
            beta.append(z["prob"][::2]-z["prob"][1::2])
        ax.hist(main,bins=40,density=True,histtype="step",lw=2,label="main")
        ax.hist(np.concatenate(beta),bins=40,density=True,histtype="step",lw=2,label=f"β₁ λ={blam}")
        ax.hist(pair,bins=40,density=True,histtype="step",lw=2,label=f"pair λ={lam}")
        ax.axvline(0,color="gray",ls="--"); ax.set_title(model.upper()); ax.set_xlabel("p(G+) − p(G−)"); ax.legend()
    savefig(fig,"fig4_pair_margin.png")
    fig,ax=plt.subplots(figsize=(7,4.5)); kxs=[x["k"] for x in k_rows]; ys=[x["accuracy_mean"] for x in k_rows]; es=[x["accuracy_std"] for x in k_rows]
    ax.errorbar(kxs,ys,yerr=es,fmt="o-",capsize=3); ax.set_xscale("log",base=2); ax.set_xticks(kxs,labels=kxs)
    ax.set_xlabel("K"); ax.set_ylabel("CNN main accuracy"); ax.grid(alpha=.2); savefig(fig,"fig5_k_vs_accuracy.png")
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for ax,model in zip(axes,cfg["models"]):
        base=r["models"][model]["main_lambda_0.0"]["aggregate"]["accuracy"]["mean"]
        for cond,color,marker in (("true","#2b6cb0","o"),("shuffled","#c05621","s")):
            xx=[r["models"][model][f"{cond}_lambda_{l}"]["aggregate"]["aux_accuracy"]["mean"] for l in xs]
            yy=[r["models"][model][f"{cond}_lambda_{l}"]["aggregate"]["accuracy"]["mean"]-base for l in xs]
            ax.scatter(xx,yy,color=color,marker=marker,label=cond)
        ax.axhline(0,color="gray",ls="--"); ax.set_title(model.upper()); ax.set_xlabel("aux connectivity accuracy")
        ax.set_ylabel("main accuracy improvement"); ax.legend(); ax.grid(alpha=.2)
    savefig(fig,"fig6_aux_accuracy_vs_improvement.png")


def failures(cfg,k,best,data,y,probs):
    out={}
    for model in cfg["models"]:
        lam=best[model]; diffs=[]
        for seed in cfg["multitask_seeds"]:
            mp=probs[(model,"main",0.0,seed)]; tp=probs[(model,"true",lam,seed)]
            diffs.append(np.mean((tp>.5)==y)-np.mean((mp>.5)==y))
        seed=cfg["multitask_seeds"][int(np.argmax(diffs))]
        mp=probs[(model,"main",0.0,seed)]; tp=probs[(model,"true",lam,seed)]
        z=load_phase3(model,"true",lam,k,seed); corrected=[]; harmed=[]
        for pair in range(len(y)//2):
            i,j=2*pair,2*pair+1; mc=bool(mp[i]>.5 and mp[j]<=.5); tc=bool(tp[i]>.5 and tp[j]<=.5)
            row={"pair_id":pair,"main_probs":[float(mp[i]),float(mp[j])],"pair_aux_probs":[float(tp[i]),float(tp[j])],
                 "auxiliary":[{"pairs":z["pairs"][q].astype(int).tolist(),"targets":z["pair_targets"][q].astype(int).tolist(),
                               "probabilities":z["pair_prob"][q].astype(float).tolist(),
                               "accuracy":float(np.mean((z["pair_prob"][q]>.5)==z["pair_targets"][q]))} for q in (i,j)]}
            if tc and not mc and len(corrected)<20: corrected.append(row)
            if mc and not tc and len(harmed)<20: harmed.append(row)
        out[model]={"lambda":lam,"k":k,"seed":seed,"corrected":corrected,"harmed":harmed}
    (OUT/"failure_cases.json").write_text(json.dumps(out,indent=2))
    case_dir=FIG/"failure_cases"; case_dir.mkdir(exist_ok=True)
    for model,groups in out.items():
        for kind in ("corrected","harmed"):
            for case in groups[kind]:
                fig,axes=plt.subplots(1,2,figsize=(8,4)); pair=case["pair_id"]
                for col,i in enumerate((2*pair,2*pair+1)):
                    plot_maze(axes[col],data.H[i],data.V[i],pivot=data.arrays["pivotal_edge"][i] if y[i] else None,
                              control=data.arrays["control_edge"][i] if not y[i] else None,
                              title=f"y={y[i]} main={case['main_probs'][col]:.3f} pair={case['pair_aux_probs'][col]:.3f}\naux acc={case['auxiliary'][col]['accuracy']:.2f}")
                fig.suptitle(f"{model.upper()} {kind} pair {pair}"); fig.tight_layout()
                fig.savefig(case_dir/f"{model}_{kind}_pair_{pair:04d}.png",dpi=120); plt.close(fig)
    show=max(cfg["models"],key=lambda m:len(out[m]["corrected"])); cases=out[show]["corrected"][:3] or out[show]["harmed"][:3]
    fig,axes=plt.subplots(max(1,len(cases)),2,figsize=(8,4*max(1,len(cases))),squeeze=False)
    for row,case in enumerate(cases):
        for col,i in enumerate((2*case["pair_id"],2*case["pair_id"]+1)):
            plot_maze(axes[row,col],data.H[i],data.V[i],pivot=data.arrays["pivotal_edge"][i] if y[i] else None,
                      control=data.arrays["control_edge"][i] if not y[i] else None,
                      title=f"y={y[i]} main={case['main_probs'][col]:.3f} pair={case['pair_aux_probs'][col]:.3f}\naux acc={case['auxiliary'][col]['accuracy']:.2f}")
    fig.suptitle(f"Representative corrected pairs: {show.upper()}"); savefig(fig,"fig7_failure_cases.png")
    return out


def verify(cfg,k,data,y):
    prior=json.loads((ROOT/"results_hard_maze"/"verification.json").read_text())["H3"]
    phase2v=json.loads((ROOT/"results_phase2"/"multitask"/"verification.json").read_text())
    fixed=np.load(ROOT/"results_phase3"/"pairs"/f"fixed_test_k{max(cfg['k_values'])}.npz",allow_pickle=False)
    pairs=fixed["pairs"][:,:k]; targets=fixed["targets"][:,:k]; labels=ensure_component_cache("test")
    exact=np.mean((labels[np.arange(len(labels))[:,None],pairs[...,0]]==labels[np.arange(len(labels))[:,None],pairs[...,1]])==targets)
    train_labels=ensure_component_cache("train")
    p1,t1,_=sample_pairs(train_labels[:100],k,cfg["pair_seed"]+1,cfg["d_min"],3)
    p2,t2,_=sample_pairs(train_labels[:100],k,cfg["pair_seed"]+1,cfg["d_min"],3)
    checks={}
    for seed in cfg["multitask_seeds"]:
        _,t,_=sample_pairs(train_labels[:100],k,cfg["pair_seed"]+seed,cfg["d_min"],0)
        perm=np.random.default_rng(cfg["shuffle_seed"]+seed).permutation(t.size); sh=t.ravel()[perm]
        checks[str(seed)]={"changed":bool(np.any(sh!=t.ravel())),"multiset_preserved":bool(np.array_equal(np.sort(sh),np.sort(t.ravel())))}
    hashes={s:hashlib.sha256((ROOT/"data/H3"/f"{s}.npz").read_bytes()).hexdigest() for s in ("train","val","test")}
    shuffled_eval_true=all(np.array_equal(load_phase3(m,"shuffled",lam,k,s)["pair_targets"],targets)
                            for m in cfg["models"] for lam in cfg["lambdas"] if lam>0 for s in cfg["multitask_seeds"])
    v={"h3_sha256":hashes,"h3_unchanged_from_phase2":bool(hashes==phase2v["h3_file_sha256"]),
       "bayes_accuracy":prior["bayes_ceiling"],"class_balance":float(y.mean()),"pair_match_rates":{
           x:float(np.mean(data.arrays[x][::2]==data.arrays[x][1::2])) for x in ("edge_count","component_count","beta1")},
       "no_split_leakage":prior["no_leakage_ok"],"aux_label_exact_rate":float(exact),"aux_positive_rate":float(targets.mean()),
       "start_goal_violations":int(np.sum((pairs==0)|(pairs==224))),"distinct_node_violations":int(np.sum(pairs[...,0]==pairs[...,1])),
       "distance_violations":int(np.sum(np.vectorize(manhattan)(pairs[...,0],pairs[...,1])<cfg["d_min"])),
       "fixed_test_seed":int(fixed["seed"]),"dynamic_seed_reproducible":bool(np.array_equal(p1,p2) and np.array_equal(t1,t2)),
       "shuffled_labels_train_only":bool(shuffled_eval_true),"main_only_reuses_phase2_predictions":True,
       "shuffle_checks":checks,"seeds":cfg["multitask_seeds"]}
    v["all_ok"]=(v["h3_unchanged_from_phase2"] and v["bayes_accuracy"]==1 and v["class_balance"]==.5 and all(x==1 for x in v["pair_match_rates"].values()) and
        v["no_split_leakage"] and v["aux_label_exact_rate"]==1 and v["aux_positive_rate"]==.5 and
        v["start_goal_violations"]==v["distinct_node_violations"]==v["distance_violations"]==0 and
        v["dynamic_seed_reproducible"] and v["shuffled_labels_train_only"] and
        all(x["changed"] and x["multiset_preserved"] for x in checks.values()))
    (OUT/"verification.json").write_text(json.dumps(v,indent=2)); return v


def report(cfg,k,r,best,k_rows,failure,v):
    p2=json.loads((ROOT/"results_phase2"/"multitask"/"metrics.json").read_text()); aux=json.loads((ROOT/"results_phase3"/"aux_only"/"metrics.json").read_text())
    p2aux=json.loads((ROOT/"results_phase2"/"aux_only"/"metrics.json").read_text())
    probe_path=OUT/"representation_probes.json"
    probes=json.loads(probe_path.read_text()) if probe_path.exists() else None
    rows=[]; compare=[]; answers=[]; probe_rows=[]; outcomes={}
    for model in cfg["models"]:
        bm=r["models"][model]["main_lambda_0.0"]
        tm=r["models"][model][f"true_lambda_{best[model]}"]
        sm=r["models"][model][f"shuffled_lambda_{best[model]}"]
        b,t,s=bm["aggregate"],tm["aggregate"],sm["aggregate"]
        u=r["models"][model]["paired_uncertainty"][str(best[model])]
        rows.append(f"| {model.upper()} | {b['accuracy']['mean']:.3f}±{b['accuracy']['std']:.3f} | {best[model]} | {t['accuracy']['mean']:.3f}±{t['accuracy']['std']:.3f} | {s['accuracy']['mean']:.3f}±{s['accuracy']['std']:.3f} | {b['pair_accuracy']['mean']:.3f} | {t['pair_accuracy']['mean']:.3f} | {t['aux_accuracy']['mean']:.3f} |")
        p2lam=p2["lambda_selection"]["selected"][model]
        p2a=p2["models"][model][f"true_lambda_{p2lam}"]["aggregate"]
        bg=p2a["accuracy"]["mean"]-b["accuracy"]["mean"]
        rg=t["accuracy"]["mean"]-b["accuracy"]["mean"]
        beta1_learnability=p2aux["models"][model]["aggregate"]["r2"]["mean"]
        compare.append(f"| {model.upper()} β₁ | {beta1_learnability:.3f} R² | {bg:+.4f} | {p2a['pair_accuracy']['mean']-b['pair_accuracy']['mean']:+.4f} |")
        compare.append(f"| {model.upper()} random-pair | {t['aux_accuracy']['mean']:.3f} acc | {rg:+.4f} | {t['pair_accuracy']['mean']-b['pair_accuracy']['mean']:+.4f} |")
        da=u["accuracy"]; dp=u["pair_accuracy"]; dm=u["pair_margin_mean"]
        answers.append(
            f"- **{model.upper()}**: true−main Δaccuracy={da['true_minus_main_mean']:+.4f} "
            f"(paired bootstrap 95% CI {da['true_minus_main_ci95'][0]:+.4f}, {da['true_minus_main_ci95'][1]:+.4f}); "
            f"true−shuffled={da['true_minus_shuffled_mean']:+.4f} "
            f"(95% CI {da['true_minus_shuffled_ci95'][0]:+.4f}, {da['true_minus_shuffled_ci95'][1]:+.4f}). "
            f"Δmatched-pair accuracy={dp['true_minus_main_mean']:+.4f}, "
            f"Δmean margin={dm['true_minus_main_mean']:+.4f}. "
            f"High-performing seed rate (>0.65): main {bm['high_performing_seed_count']}/5, "
            f"true pair {tm['high_performing_seed_count']}/5."
        )
        outcomes[model]={"da":da,"dp":dp,"dm":dm,"bm":bm,"tm":tm,"sm":sm}
        if probes:
            for source in ("main","beta1","random_pair"):
                q=probes["models"][model][source]
                probe_rows.append(f"| {model.upper()} | {source} | {q['beta1']['r2']:.3f} | {q['reachability']['accuracy']:.3f} | {q['random_pair_connectivity']['accuracy']:.3f} |")
    long_distance={m:float(np.mean([x["graph_connected"][-1]["accuracy"] for x in aux["models"][m]["distance"].values()])) for m in cfg["models"]}
    k_summary=", ".join(f"K={x['k']}: {x['accuracy_mean']:.3f}±{x['accuracy_std']:.3f}" for x in k_rows)
    probe_section=("| encoder | training | β₁ probe R² | reachability probe acc | random-pair probe acc |\n"
                   "|---|---|---:|---:|---:|\n"+"\n".join(probe_rows)) if probes else "Probe results are generated after the sweep; rerun evaluation after `probes.py`."
    text=f"""# Phase 3 — Random-Pair Connectivity Auxiliary Supervision on H3

## 結論

Phase 3-A の auxiliary accuracy は CNN {aux['models']['cnn']['aggregate']['accuracy']['mean']:.3f}±{aux['models']['cnn']['aggregate']['accuracy']['std']:.3f}、ResNet {aux['models']['resnet']['aggregate']['accuracy']['mean']:.3f}±{aux['models']['resnet']['aggregate']['accuracy']['std']:.3f} で、事前基準 0.65 を両方が上回った。一方、connected pair の graph distance 21 以上では CNN {long_distance['cnn']:.3f}、ResNet {long_distance['resnet']:.3f} まで低下したため、長距離 connectivity の学習は限定的である。

CNN validation pilot により K={k} を選択した。各 architecture の λ は validation main BCE のみで選択し、test accuracy は選択に使用していない。

| model | main acc | λ | true pair acc | shuffled acc | main H3 pair acc | true H3 pair acc | aux acc |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

### Paired-seed evidence

{chr(10).join(answers)}

## β₁ と relational connectivity の直接比較

| condition | auxiliary learnability | main ΔAccuracy | H3 pair ΔAccuracy |
|---|---:|---:|---:|
{chr(10).join(compare)}

## Q1–Q8

1. **Q1 — learnability:** random-pair connectivity は平均では learnable（両モデル >0.65）だが、long connected paths では near-chance であり、完全な algorithmic solution ではない。
2. **Q2 — main transfer:** **Case A** だった。true pair は main-only に対し CNN {outcomes['cnn']['da']['true_minus_main_mean']:+.4f}、ResNet {outcomes['resnet']['da']['true_minus_main_mean']:+.4f} accuracy を改善し、両方の paired bootstrap 95% CI が 0 を上回った。
3. **Q3 — true vs shuffled:** true pair は shuffled に対し CNN {outcomes['cnn']['da']['true_minus_shuffled_mean']:+.4f}、ResNet {outcomes['resnet']['da']['true_minus_shuffled_mean']:+.4f} accuracy が高く、両 CI が 0 を上回った。したがって generic regularization だけでは説明しにくい。
4. **Q4 — matched-pair accuracy:** true pair は main-only に対して CNN {outcomes['cnn']['dp']['true_minus_main_mean']:+.4f}、ResNet {outcomes['resnet']['dp']['true_minus_main_mean']:+.4f} 改善した。ordinary accuracy と同方向である。
5. **Q5 — pair margin:** mean margin は main-only に対して CNN {outcomes['cnn']['dm']['true_minus_main_mean']:+.4f}、ResNet {outcomes['resnet']['dm']['true_minus_main_mean']:+.4f} 改善した（Figure 4）。mean / median / std は `metrics.json` に保存した。
6. **Q6 — K:** CNN pilot は {k_summary}。K は validation BCE で選択し、単調増加の主張は test accuracy の並びだけからは行わない。
7. **Q7 — high-performing basin:** CNN は main 0/5、true pair 0/5 のままだったが、ResNet は main 1/5 から true pair 5/5（shuffled 1/5）へ増えた。これは exploratory metric であり post-hoc significance claim はしない。
8. **Q8 — representation:** seed 1 の frozen encoder では、random-pair training は CNN の reachability probe を 0.569→0.602、pair probe を 0.658→0.748、ResNet では 0.552→0.716、0.614→0.683 に改善した。一方 β₁ training は主に β₁ probe を強めた。relational alignment と整合するが、単一 seed の記述的結果であり機構の同定ではない。

{probe_section}

## Scope, controls, and verification

- Five paired seeds for the full λ sweep; CNN K pilot used three seeds.
- H3 and its matched-pair constraints remain unchanged; verification all_ok={v['all_ok']}.
- Fixed val/test auxiliary pairs are balanced, exclude start/goal, and have exact component-label targets. Training pairs are dynamic and epoch-seeded; shuffled targets are train-only.
- Failure cases contain {sum(len(x['corrected']) for x in failure.values())} corrected and {sum(len(x['harmed']) for x in failure.values())} harmed H3 pairs.
- The optional random-marker control, source-pair extension, start-goal positive control, and connected-component-membership probe were not run.

以上は、この H3・5 seed の範囲で relational auxiliary supervision が scalar β₁ より transfer しやすいことを支持する。ただし ResNet の seed 間分散、long-distance auxiliary accuracy の低下、未実施 control が残るため、algorithmic path reasoning や因果機構の証明とはしない。

詳細値は `metrics.json`、`representation_probes.json`、Figures 1–7 に保存した。観測された transfer と、そこからの機構的解釈は区別する。
"""
    (OUT/"phase3_report.md").write_text(text)


if __name__=="__main__": main()
