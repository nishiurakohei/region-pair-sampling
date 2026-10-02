"""Reproduce the published DT-Net OOD trends without modifying upstream."""
import argparse, csv, hashlib, itertools, json, platform, sys, time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE=Path(__file__).resolve().parents[1]
UP=HERE/"upstream"/"maze-extrapolation"
sys.path.insert(0,str(UP))
from src.models.dt_net import DTNet
from maze_dataset.dataset.rasterized import MazeDatasetConfig, MazeDataset, RasterizedMazeDataset
from maze_dataset.generation import LatticeMazeGenerators


def load_model(device):
    model=DTNet()
    saved=torch.load(UP/"models"/"dt_net.pth",map_location="cpu",weights_only=True)["net"]
    model.load_state_dict({k.replace("module.",""):v for k,v in saved.items()},strict=True)
    return model.eval().to(device)


def make_data(size,n,p,deadend,seed,cache):
    cfg=MazeDatasetConfig(name=f"s0-n{size}-p{p}-d{int(deadend)}-s{seed}",grid_n=(size+1)//2,n_mazes=n,seed=seed,
        maze_ctor=LatticeMazeGenerators.gen_dfs_percolation,maze_ctor_kwargs={"p":p},
        endpoint_kwargs={"deadend_start":deadend,"endpoints_not_equal":True})
    base=MazeDataset.from_config(cfg,local_base_path=str(cache))
    raster=RasterizedMazeDataset.from_base_MazeDataset(base_dataset=base,
        added_params={"remove_isolated_cells":True,"extend_pixels":True}).get_batch(idxs=None)
    x=(raster[0].float()/255).permute(0,3,1,2)
    y=(raster[1].float()/255).permute(0,3,1,2).max(1).values
    return x,y,hashlib.sha256(x.numpy().tobytes()).hexdigest()


@torch.inference_mode()
def evaluate(model,x,y,iters,batch_size,device):
    correct={i:[] for i in iters}; started=time.time()
    for lo in range(0,len(x),batch_size):
        xb=x[lo:lo+batch_size].to(device); yb=y[lo:lo+batch_size].to(device)
        latent=model.input_to_latent(xb)
        for step in range(1,max(iters)+1):
            latent=model.recur_block(torch.cat([latent,xb],1))
            if step in correct:
                pred=model.output_to_prediction(model.head(latent),xb)
                correct[step].extend(torch.all(pred==yb,dim=(1,2)).cpu().tolist())
    elapsed=time.time()-started
    return {i:(float(np.mean(correct[i])),correct[i],elapsed*i/max(iters)) for i in iters}


def aggregate(rows):
    groups={}
    for r in rows:
        key=(r["axis"],r["maze_size"],r["percolation"],r["deadend_start"],r["iterations"])
        groups.setdefault(key,[]).append(r["accuracy"])
    return [{"axis":k[0],"maze_size":k[1],"percolation":k[2],"deadend_start":k[3],"iterations":k[4],
             "accuracy_mean":float(np.mean(v)),"accuracy_std":float(np.std(v)),"seeds":len(v)} for k,v in groups.items()]


def figures(summary,out):
    figdir=out/"figures";figdir.mkdir(parents=True,exist_ok=True)
    specs=[("maze_size","maze_size","Maze size"),("percolation","percolation","Percolation")]
    for axis,xkey,xlabel in specs:
        fig,ax=plt.subplots(figsize=(7,4.5))
        rows=[r for r in summary if r["axis"]==axis]
        for it in sorted({r["iterations"] for r in rows}):
            q=sorted((r for r in rows if r["iterations"]==it),key=lambda z:z[xkey])
            ax.errorbar([r[xkey] for r in q],[r["accuracy_mean"] for r in q],yerr=[r["accuracy_std"] for r in q],marker="o",label=f"{it} iter")
        ax.set(xlabel=xlabel,ylabel="Exact solution accuracy",ylim=(-.03,1.03));ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(figdir/f"{axis}_vs_accuracy.png",dpi=160);plt.close(fig)
    rows=[r for r in summary if r["axis"]=="deadend_start"]
    fig,ax=plt.subplots(figsize=(5,4));x=np.arange(2);q=sorted(rows,key=lambda z:z["deadend_start"],reverse=True)
    ax.bar(x,[r["accuracy_mean"] for r in q],yerr=[r["accuracy_std"] for r in q],capsize=4)
    ax.set_xticks(x,["True (ID)","False (OOD)"]);ax.set(ylabel="Exact solution accuracy",ylim=(0,1.03));fig.tight_layout();fig.savefig(figdir/"deadend_start_comparison.png",dpi=160);plt.close(fig)
    rows=[r for r in summary if r["axis"]=="maze_size"]
    fig,ax=plt.subplots(figsize=(7,4.5))
    for size in sorted({r["maze_size"] for r in rows}):
        q=sorted((r for r in rows if r["maze_size"]==size),key=lambda z:z["iterations"])
        ax.plot([r["iterations"] for r in q],[r["accuracy_mean"] for r in q],"o-",label=f"{size}×{size}")
    ax.set(xlabel="Iterations",ylabel="Exact solution accuracy",ylim=(-.03,1.03));ax.set_xscale("log");ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(figdir/"iterations_vs_accuracy.png",dpi=160);plt.close(fig)


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--config",type=Path,default=HERE/"configs"/"stage0_screen.json");ap.add_argument("--out",type=Path,default=HERE.parent/"results_external"/"stage0_reproduction");a=ap.parse_args()
    cfg=json.loads(a.config.read_text());a.out.mkdir(parents=True,exist_ok=True);(a.out/"config.json").write_text(json.dumps(cfg,indent=2))
    commit=__import__("subprocess").check_output(["git","-C",str(UP),"rev-parse","HEAD"],text=True).strip();(a.out/"upstream_commit.txt").write_text(commit+"\n")
    device=torch.device("mps" if torch.backends.mps.is_available() else "cpu");model=load_model(device);rows=[]
    for axis,spec in cfg.items():
        if axis in {"model","seeds","batch_size"}: continue
        for seed,size,p,deadend in itertools.product(cfg["seeds"],spec["maze_sizes"],spec["percolations"],spec["deadend_starts"]):
            x,y,digest=make_data(size,spec["num_mazes"],p,deadend,seed,HERE/"data")
            scores=evaluate(model,x,y,spec["iterations"],cfg["batch_size"],device)
            for it,(acc,correct,seconds) in scores.items():
                row={"axis":axis,"seed":seed,"maze_size":size,"percolation":p,"deadend_start":deadend,"iterations":it,
                     "num_mazes":len(x),"accuracy":acc,"correct_count":int(sum(correct)),"seconds":seconds,"dataset_sha256":digest}
                rows.append(row);print(row,flush=True)
    with (a.out/"per_seed_metrics.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    summary=aggregate(rows);(a.out/"metrics.json").write_text(json.dumps({"upstream_commit":commit,"device":str(device),"summary":summary},indent=2));figures(summary,a.out)
    env={"python":sys.version,"platform":platform.platform(),"torch":torch.__version__,"numpy":np.__version__,"maze_dataset":"1.1.0","muutils":"0.6.21","device":str(device)}
    (a.out/"environment.json").write_text(json.dumps(env,indent=2))

if __name__=="__main__": main()
