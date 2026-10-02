"""Complete Stage 0 size/iteration coverage, checkpointing after each size."""
import argparse, csv, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from stage0_reproduce import HERE, evaluate, load_model, make_data


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--config",type=Path,default=HERE/"configs"/"stage0_full_tail.json")
    ap.add_argument("--out",type=Path,default=HERE.parent/"results_external"/"stage0_reproduction"/"full_tail");a=ap.parse_args()
    cfg=json.loads(a.config.read_text());a.out.mkdir(parents=True,exist_ok=True);(a.out/"config.json").write_text(json.dumps(cfg,indent=2))
    device=torch.device("mps" if torch.backends.mps.is_available() else "cpu");model=load_model(device)
    for size in cfg["maze_sizes"]:
        target=a.out/f"size_{size}.json"
        if target.exists(): print("skip completed",size,flush=True);continue
        xs=[];ys=[];hashes=[]
        for seed in cfg["seeds"]:
            x,y,h=make_data(size,cfg["num_mazes_per_seed"],cfg["percolation"],cfg["deadend_start"],seed,HERE/"data")
            xs.append(x);ys.append(y);hashes.append(h)
        scores=evaluate(model,torch.cat(xs),torch.cat(ys),cfg["iterations"],len(xs),device)
        rows=[]
        for it,(acc,correct,seconds) in scores.items():
            for seed,ok,digest in zip(cfg["seeds"],correct,hashes):
                rows.append({"seed":seed,"maze_size":size,"iterations":it,"num_mazes":1,"accuracy":float(ok),"seconds_batch":seconds,"dataset_sha256":digest})
            print({"maze_size":size,"iterations":it,"accuracy":acc,"seconds":seconds},flush=True)
        target.write_text(json.dumps(rows,indent=2))
    rows=[]
    for size in cfg["maze_sizes"]: rows.extend(json.loads((a.out/f"size_{size}.json").read_text()))
    with (a.out/"per_seed_metrics.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    summary=[]
    for size in cfg["maze_sizes"]:
        for it in cfg["iterations"]:
            vals=[r["accuracy"] for r in rows if r["maze_size"]==size and r["iterations"]==it]
            summary.append({"maze_size":size,"iterations":it,"accuracy_mean":sum(vals)/len(vals),"correct":int(sum(vals)),"n":len(vals)})
    (a.out/"metrics.json").write_text(json.dumps({"device":str(device),"summary":summary},indent=2))
    screen=json.loads((a.out.parent/"metrics.json").read_text())["summary"]
    old=[r for r in screen if r["axis"]=="maze_size"]
    combined=[{"maze_size":r["maze_size"],"iterations":r["iterations"],"accuracy_mean":r["accuracy_mean"]} for r in old]+summary
    figdir=a.out.parent/"figures"
    fig,ax=plt.subplots(figsize=(8,5))
    for it in [30,100,300,1000,3000]:
        q=sorted((r for r in combined if r["iterations"]==it),key=lambda r:r["maze_size"])
        if q: ax.plot([r["maze_size"] for r in q],[r["accuracy_mean"] for r in q],"o-",label=f"{it} iter")
    ax.set(xlabel="Maze size",ylabel="Exact solution accuracy",ylim=(-.03,1.03));ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(figdir/"full_maze_size_vs_accuracy.png",dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5))
    for size in sorted({r["maze_size"] for r in combined}):
        q=sorted((r for r in combined if r["maze_size"]==size),key=lambda r:r["iterations"])
        ax.plot([r["iterations"] for r in q],[r["accuracy_mean"] for r in q],"o-",label=f"{size}×{size}")
    ax.set(xlabel="Iterations",ylabel="Exact solution accuracy",ylim=(-.03,1.03));ax.set_xscale("log");ax.grid(alpha=.25);ax.legend(ncol=2,fontsize=8);fig.tight_layout();fig.savefig(figdir/"full_iterations_vs_accuracy.png",dpi=160);plt.close(fig)

if __name__=="__main__": main()
